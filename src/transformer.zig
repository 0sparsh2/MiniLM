const std = @import("std");
const micro_kernel = @import("micro_kernel.zig");
const stochastic = @import("stochastic.zig");
const thread_pool = @import("thread_pool.zig");

const binary_kernel = @import("binary_kernel.zig");

/// A Linear layer that exclusively uses 1-bit weights during the forward pass
/// and popcount arithmetic (dotProductBinary).
pub const BitLinear = struct {
    in_features: usize,
    out_features: usize,
    
    // Weights are 1-bit (-1, 1), packed into u64
    weights: []u64,
    
    // Shadow weights for the backward pass, stored as i16.
    shadow_weights: []i16,

    pub fn init(allocator: std.mem.Allocator, in_features: usize, out_features: usize) !BitLinear {
        std.debug.assert(in_features % 64 == 0); // Must be multiple of 64
        
        const packed_dim = in_features / 64;
        // The ArenaAllocator provides excellent contiguous cache packing.
        // If we strictly want 64-byte alignment, we would need to pass std.mem.Alignment.fromByteUnits(64)
        // depending on the exact Zig compiler version. For now, ArenaAllocator gives us ~95% of the benefit.
        const weights = try allocator.alloc(u64, packed_dim * out_features);
        const shadow_weights = try allocator.alloc(i16, in_features * out_features);
        
        @memset(weights, 0);
        @memset(shadow_weights, 0);
        
        return .{
            .in_features = in_features,
            .out_features = out_features,
            .weights = weights,
            .shadow_weights = shadow_weights,
        };
    }

    pub fn deinit(self: *BitLinear, allocator: std.mem.Allocator) void {
        allocator.free(self.weights);
        allocator.free(self.shadow_weights);
    }

    /// Forward pass expects packed u64 inputs!
    pub fn forward(self: *BitLinear, x_packed: []const u64, out: []i32) void {
        const packed_dim = self.in_features / 64;
        std.debug.assert(x_packed.len == packed_dim);
        std.debug.assert(out.len == self.out_features);
        
        for (0..self.out_features) |i| {
            const row_start = i * packed_dim;
            const w_row = self.weights[row_start .. row_start + packed_dim];
            out[i] = binary_kernel.dotProductBinary(w_row, x_packed);
        }
    }

    pub const ForwardCtx = struct {
        weights: []const u64,
        x_packed: []const u64,
        out: []i32,
        packed_dim: usize,
        start_row: usize,
        end_row: usize,
    };

    fn processForwardBatch(ctx: *ForwardCtx) void {
        for (ctx.start_row..ctx.end_row) |i| {
            const row_start = i * ctx.packed_dim;
            const w_row = ctx.weights[row_start .. row_start + ctx.packed_dim];
            ctx.out[i] = binary_kernel.dotProductBinary(w_row, ctx.x_packed);
        }
    }

    /// Multi-threaded forward pass across multiple CPU cores.
    pub fn forwardThreaded(self: *BitLinear, x_packed: []const u64, out: []i32, pool: *const thread_pool.ThreadPool) !void {
        const packed_dim = self.in_features / 64;
        std.debug.assert(x_packed.len == packed_dim);
        std.debug.assert(out.len == self.out_features);
        
        if (pool.num_threads <= 1 or self.out_features < pool.num_threads) {
            self.forward(x_packed, out);
            return;
        }

        // Zero-allocation stack buffer for context structs
        var stack_buf: [2048]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&stack_buf);
        const allocator = fba.allocator();

        var contexts = try allocator.alloc(ForwardCtx, pool.num_threads);
        
        var rows_per_thread = self.out_features / pool.num_threads;
        if (rows_per_thread == 0) rows_per_thread = 1;

        var start_idx: usize = 0;
        for (0..pool.num_threads) |i| {
            var end_idx = start_idx + rows_per_thread;
            if (i == pool.num_threads - 1 or end_idx > self.out_features) {
                end_idx = self.out_features;
            }
            
            contexts[i] = .{
                .weights = self.weights,
                .x_packed = x_packed,
                .out = out,
                .packed_dim = packed_dim,
                .start_row = start_idx,
                .end_row = end_idx,
            };
            
            start_idx = end_idx;
        }

        try pool.execute(ForwardCtx, contexts, processForwardBatch);
    }

    /// Localized weight update for Forward-Forward / Local Learning algorithms.
    /// This removes the need for global backpropagation and massive memory chains.
    /// It applies the gradient immediately to the shadow weights.
    pub fn localUpdate(
        self: *BitLinear, 
        in_packed: []const u64, 
        local_error: []const i32, 
        shift_scale: u4, 
        rng: *stochastic.VectorXorshift32,
        allocator: std.mem.Allocator
    ) !void {
        var gradients = try allocator.alloc(i8, self.shadow_weights.len);
        defer allocator.free(gradients);
        @memset(gradients, 0);

        for (0..self.out_features) |out_idx| {
            const row_start = out_idx * self.in_features;
            const err = local_error[out_idx];
            
            var grad = -err;
            if (grad > 127) grad = 127;
            if (grad < -128) grad = -128;
            
            const grad_pos = @as(i8, @intCast(grad));
            const grad_neg = -grad_pos;
            
            for (0..self.in_features) |in_idx| {
                const block_idx = in_idx / 64;
                const bit_idx = @as(u6, @intCast(in_idx % 64));
                const bit = (in_packed[block_idx] >> bit_idx) & 1;
                
                gradients[row_start + in_idx] = if (bit == 1) grad_pos else grad_neg;
            }
        }
        
        stochastic.stochasticUpdate(self.shadow_weights, gradients, shift_scale, rng);
        self.syncWeights();
    }
    
    /// Synchronize standard weights from shadow weights
    pub fn syncWeights(self: *BitLinear) void {
        const shadow_i8 = std.heap.page_allocator.alloc(i8, self.shadow_weights.len) catch unreachable;
        defer std.heap.page_allocator.free(shadow_i8);
        for (self.shadow_weights, 0..) |v, i| shadow_i8[i] = if (v >= 0) 1 else -1;
        binary_kernel.packBinary(shadow_i8, self.weights);
    }
    
    pub fn save(self: *BitLinear, writer: anytype) !void {
        // We only need to save the final packed binary weights (inference only).
        // Since weights is []u64, we write them directly.
        try writer.writeAll(std.mem.sliceAsBytes(self.weights));
    }
    
    pub fn load(self: *BitLinear, reader: anytype) !void {
        _ = try reader.readAll(std.mem.sliceAsBytes(self.weights));
    }
};

/// A simple MLP block using our BitLinear layers and a dummy ReLU activation.
pub const BitMLP = struct {
    fc1: BitLinear,
    fc2: BitLinear,
    
    // Intermediate activation buffer
    hidden: []i32,
    hidden_packed: []u64, // Quantized back to 1-bit for the next layer

    pub fn init(allocator: std.mem.Allocator, in_features: usize, hidden_features: usize) !BitMLP {
        std.debug.assert(hidden_features % 64 == 0);
        const fc1 = try BitLinear.init(allocator, in_features, hidden_features);
        const fc2 = try BitLinear.init(allocator, hidden_features, in_features);
        const hidden = try allocator.alloc(i32, hidden_features);
        const hidden_packed = try allocator.alloc(u64, hidden_features / 64);
        
        return .{
            .fc1 = fc1,
            .fc2 = fc2,
            .hidden = hidden,
            .hidden_packed = hidden_packed,
        };
    }

    pub fn deinit(self: *BitMLP, allocator: std.mem.Allocator) void {
        self.fc1.deinit(allocator);
        self.fc2.deinit(allocator);
        allocator.free(self.hidden);
        allocator.free(self.hidden_packed);
    }

    pub fn forward(self: *BitMLP, allocator: std.mem.Allocator, x_packed: []const u64, out: []i32, pool: *const thread_pool.ThreadPool) !void {
        _ = allocator; // For future dynamic allocations if needed
        
        // First linear layer
        try self.fc1.forwardThreaded(x_packed, self.hidden, pool);
        
        // Binarization (Sign activation function)
        // >= 0 becomes bit 1, < 0 becomes bit 0
        @memset(self.hidden_packed, 0);
        for (self.hidden, 0..) |val, i| {
            if (val >= 0) {
                const block_idx = i / 64;
                const bit_idx = @as(u6, @intCast(i % 64));
                self.hidden_packed[block_idx] |= (@as(u64, 1) << bit_idx);
            }
        }
        
        // Second linear layer
        try self.fc2.forwardThreaded(self.hidden_packed, out, pool);
    }
    
    pub fn save(self: *BitMLP, writer: anytype) !void {
        try self.fc1.save(writer);
        try self.fc2.save(writer);
    }
    
    pub fn load(self: *BitMLP, reader: anytype) !void {
        try self.fc1.load(reader);
        try self.fc2.load(reader);
    }
};
