const std = @import("std");
const binary_kernel = @import("binary_kernel.zig");
const thread_pool = @import("thread_pool.zig");
const stochastic = @import("stochastic.zig");

/// A simplified 1-Bit Selective State-Space Model (inspired by Mamba / RWKV).
/// Replaces O(N^2) Self-Attention with an O(N) binary RNN state.
/// This processes sequences purely via bitwise state accumulation.
pub const BinarySSM = struct {
    embed_dim: usize,
    
    // Binary Projections (weights are packed as u64)
    w_in: []u64,
    w_out: []u64,
    
    // Shadow weights for stochastic learning (i16)
    shadow_in: []i16,
    shadow_out: []i16,
    
    // Persistent state for inference
    inference_state: []u64,

    pub fn init(allocator: std.mem.Allocator, embed_dim: usize) !BinarySSM {
        const packed_dim = embed_dim / 64;
        
        const w_in = try allocator.alloc(u64, packed_dim * embed_dim);
        const w_out = try allocator.alloc(u64, packed_dim * embed_dim);
        const shadow_in = try allocator.alloc(i16, embed_dim * embed_dim);
        const shadow_out = try allocator.alloc(i16, embed_dim * embed_dim);
        const inference_state = try allocator.alloc(u64, packed_dim);
        
        @memset(w_in, 0);
        @memset(w_out, 0);
        @memset(shadow_in, 0);
        @memset(shadow_out, 0);
        @memset(inference_state, 0);
        
        return .{
            .embed_dim = embed_dim,
            .w_in = w_in,
            .w_out = w_out,
            .shadow_in = shadow_in,
            .shadow_out = shadow_out,
            .inference_state = inference_state,
        };
    }

    pub fn deinit(self: *BinarySSM, allocator: std.mem.Allocator) void {
        allocator.free(self.w_in);
        allocator.free(self.w_out);
        allocator.free(self.shadow_in);
        allocator.free(self.shadow_out);
        allocator.free(self.inference_state);
    }
    
    pub fn resetCache(self: *BinarySSM) void {
        @memset(self.inference_state, 0);
    }

    pub fn syncWeights(self: *BinarySSM) void {
        const shadow_in_i8 = std.heap.page_allocator.alloc(i8, self.shadow_in.len) catch unreachable;
        defer std.heap.page_allocator.free(shadow_in_i8);
        for (self.shadow_in, 0..) |v, i| shadow_in_i8[i] = if (v >= 0) 1 else -1;
        binary_kernel.packBinary(shadow_in_i8, self.w_in);
        
        const shadow_out_i8 = std.heap.page_allocator.alloc(i8, self.shadow_out.len) catch unreachable;
        defer std.heap.page_allocator.free(shadow_out_i8);
        for (self.shadow_out, 0..) |v, i| shadow_out_i8[i] = if (v >= 0) 1 else -1;
        binary_kernel.packBinary(shadow_out_i8, self.w_out);
    }
    
    pub fn randomize(self: *BinarySSM, rng: *stochastic.VectorXorshift32) void {
        for (0..self.shadow_in.len) |i| self.shadow_in[i] = @as(i16, @intCast(rng.next()[0] % 50)) - 25;
        for (0..self.shadow_out.len) |i| self.shadow_out[i] = @as(i16, @intCast(rng.next()[0] % 50)) - 25;
        self.syncWeights();
    }

    /// O(N) Forward pass that linearly accumulates the state vector.
    pub fn forward(
        self: *BinarySSM, 
        allocator: std.mem.Allocator, 
        x_i8: []const i8, 
        out_i32: []i32, 
        pool: *const thread_pool.ThreadPool
    ) !void {
        _ = pool; // Single thread for this PoC
        
        const seq_len = x_i8.len / self.embed_dim;
        const packed_dim = self.embed_dim / 64;
        
        var state_vector: []u64 = undefined;
        var free_state = false;
        
        if (seq_len > 1) {
            // Training: Allocate fresh state for the sequence
            state_vector = try allocator.alloc(u64, packed_dim);
            @memset(state_vector, 0);
            free_state = true;
        } else {
            // Inference: Use persistent state buffer
            state_vector = self.inference_state;
        }
        defer if (free_state) allocator.free(state_vector);

        const current_x_packed = try allocator.alloc(u64, packed_dim);
        defer allocator.free(current_x_packed);

        const current_h = try allocator.alloc(i32, self.embed_dim);
        defer allocator.free(current_h);
        
        const current_h_i8 = try allocator.alloc(i8, self.embed_dim);
        defer allocator.free(current_h_i8);
        
        const current_h_packed = try allocator.alloc(u64, packed_dim);
        defer allocator.free(current_h_packed);

        for (0..seq_len) |t| {
            const token_x = x_i8[t * self.embed_dim .. (t + 1) * self.embed_dim];
            const token_o = out_i32[t * self.embed_dim .. (t + 1) * self.embed_dim];
            
            binary_kernel.packBinary(token_x, current_x_packed);
            
            // 1. Input Projection: x_t * W_in
            for (0..self.embed_dim) |d| {
                const w_row = self.w_in[d * packed_dim .. (d + 1) * packed_dim];
                current_h[d] = binary_kernel.dotProductBinary(w_row, current_x_packed);
            }
            
            // 2. State Accumulation: State = State XOR (Input > 0)
            // This is a highly simplified binary associative scan.
            for (0..self.embed_dim) |d| {
                current_h_i8[d] = if (current_h[d] > 0) 1 else -1;
            }
            binary_kernel.packBinary(current_h_i8, current_h_packed);
            
            for (0..packed_dim) |d| {
                // XNOR the incoming state with the past state
                state_vector[d] = ~(state_vector[d] ^ current_h_packed[d]);
            }
            
            // 3. Output Projection: State * W_out
            for (0..self.embed_dim) |d| {
                const w_row = self.w_out[d * packed_dim .. (d + 1) * packed_dim];
                token_o[d] = binary_kernel.dotProductBinary(w_row, state_vector);
            }
        }
    }
    
    pub fn save(self: *BinarySSM, writer: anytype) !void {
        try writer.writeAll(std.mem.sliceAsBytes(self.w_in));
        try writer.writeAll(std.mem.sliceAsBytes(self.w_out));
    }
    
    pub fn load(self: *BinarySSM, reader: anytype) !void {
        _ = try reader.readAll(std.mem.sliceAsBytes(self.w_in));
        _ = try reader.readAll(std.mem.sliceAsBytes(self.w_out));
    }
};
