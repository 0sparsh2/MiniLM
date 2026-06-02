const std = @import("std");
const transformer = @import("transformer.zig");
const moe = @import("moe.zig");
const mamba = @import("mamba.zig");
const attention = @import("attention.zig");
const thread_pool = @import("thread_pool.zig");
const stochastic = @import("stochastic.zig");
const binary_kernel = @import("binary_kernel.zig");

pub const BitMaxNorm = struct {
    /// In-place Absolute Maximum Normalization.
    pub fn forward(in_out: []i32, out_i8: []i8) void {
        std.debug.assert(in_out.len == out_i8.len);
        
        var max_abs: i32 = 0;
        for (in_out) |v| {
            const abs_v = if (v < 0) -v else v;
            if (abs_v > max_abs) max_abs = abs_v;
        }

        if (max_abs == 0) {
            @memset(out_i8, 0);
            return;
        }

        if (max_abs <= 127) {
            for (in_out, 0..) |v, i| out_i8[i] = @as(i8, @intCast(v));
        } else {
            for (in_out, 0..) |v, i| {
                const scaled = @divTrunc(v * 127, max_abs);
                out_i8[i] = @as(i8, @intCast(scaled));
            }
        }
    }
};

pub const TransformerBlock = struct {
    ssm: mamba.BinarySSM,
    mlp: moe.SparseMoE,
    local_head: transformer.BitLinear,
    embed_dim: usize,

    pub fn init(allocator: std.mem.Allocator, embed_dim: usize, vocab_size: usize, max_seq_len: usize) !TransformerBlock {
        _ = max_seq_len; // Not needed for SSM
        const hidden_dim = embed_dim * 4;
        const num_experts = 64; // Hyper-sparse 64-way routing
        return .{
            .ssm = try mamba.BinarySSM.init(allocator, embed_dim),
            .mlp = try moe.SparseMoE.init(allocator, embed_dim, hidden_dim, num_experts),
            .local_head = try transformer.BitLinear.init(allocator, embed_dim, vocab_size),
            .embed_dim = embed_dim,
        };
    }

    pub fn deinit(self: *TransformerBlock, allocator: std.mem.Allocator) void {
        self.ssm.deinit(allocator);
        self.mlp.deinit(allocator);
        self.local_head.deinit(allocator);
    }

    pub fn syncWeights(self: *TransformerBlock) void {
        self.ssm.syncWeights();
        self.mlp.syncWeights();
        self.local_head.syncWeights();
    }

    pub fn randomize(self: *TransformerBlock, rng: *stochastic.VectorXorshift32) void {
        self.ssm.randomize(rng);
        self.mlp.randomize(rng);
        
        for (0..self.local_head.shadow_weights.len) |i| self.local_head.shadow_weights[i] = @as(i16, @intCast(rng.next()[0] % 50)) - 25;
        self.syncWeights();
    }

    pub fn forward(
        self: *TransformerBlock, 
        allocator: std.mem.Allocator, 
        x_i32: []i32, 
        pool: *const thread_pool.ThreadPool,
        target_char_id: u8,
        vocab_size: usize,
        rng: *stochastic.VectorXorshift32,
        out_pred: *usize,
        out_loss: *i64,
        shift_scale: u4
    ) !void {
        const seq_len = x_i32.len / self.embed_dim;
        const packed_dim = self.embed_dim / 64;
        
        const x_i8 = try allocator.alloc(i8, x_i32.len);
        defer allocator.free(x_i8);
        
        const x_packed = try allocator.alloc(u64, seq_len * packed_dim);
        defer allocator.free(x_packed);
        
        const ssm_out = try allocator.alloc(i32, x_i32.len);
        defer allocator.free(ssm_out);
        
        const mlp_out = try allocator.alloc(i32, x_i32.len);
        defer allocator.free(mlp_out);

        // 1. Norm + SSM (Mamba equivalent)
        for (0..seq_len) |t| {
            BitMaxNorm.forward(
                x_i32[t * self.embed_dim .. (t + 1) * self.embed_dim],
                x_i8[t * self.embed_dim .. (t + 1) * self.embed_dim]
            );
            binary_kernel.packBinary(
                x_i8[t * self.embed_dim .. (t + 1) * self.embed_dim],
                x_packed[t * packed_dim .. (t + 1) * packed_dim]
            );
        }
        
        try self.ssm.forward(allocator, x_i8, ssm_out, pool);
        
        // Residual Add 1
        for (0..x_i32.len) |i| x_i32[i] += ssm_out[i];

        // 2. Norm + Sparse MoE
        for (0..seq_len) |t| {
            BitMaxNorm.forward(
                x_i32[t * self.embed_dim .. (t + 1) * self.embed_dim],
                x_i8[t * self.embed_dim .. (t + 1) * self.embed_dim]
            );
            binary_kernel.packBinary(
                x_i8[t * self.embed_dim .. (t + 1) * self.embed_dim],
                x_packed[t * packed_dim .. (t + 1) * packed_dim]
            );
        }
        
        try self.mlp.forward(allocator, x_packed, mlp_out, pool);
        
        // Residual Add 2
        for (0..x_i32.len) |i| x_i32[i] += mlp_out[i];

        // 3. Greedy Local Learning Update
        // Extract the last token context (which represents the sequence)
        const last_token_ctx = x_i32[(seq_len - 1) * self.embed_dim .. seq_len * self.embed_dim];
        
        // Binarize and pack for Local Classifier Head
        const last_token_i8 = try allocator.alloc(i8, self.embed_dim);
        defer allocator.free(last_token_i8);
        BitMaxNorm.forward(last_token_ctx, last_token_i8);
        
        const last_token_packed = try allocator.alloc(u64, packed_dim);
        defer allocator.free(last_token_packed);
        binary_kernel.packBinary(last_token_i8, last_token_packed);

        // Local Forward Pass
        const local_out = try allocator.alloc(i32, vocab_size);
        defer allocator.free(local_out);
        try self.local_head.forwardThreaded(last_token_packed, local_out, pool);

        // Local Error and Prediction
        const local_error = try allocator.alloc(i32, vocab_size);
        defer allocator.free(local_error);
        
        var max_val: i32 = -999999;
        var max_idx: u16 = 0;
        var loss: i64 = 0;
        
        for (0..vocab_size) |v| {
            const current_val = local_out[v];
            var diff: i32 = 0;
            
            if (v == target_char_id) {
                // Correct token: Push logit up to +32
                if (current_val < 32) {
                    diff = current_val - 32;
                }
            } else {
                // Incorrect tokens: Push logit down to -32
                if (current_val > -32) {
                    diff = current_val + 32;
                }
            }
            
            loss += @as(i64, diff) * @as(i64, diff);
            local_error[v] = diff;
            
            if (current_val > max_val) {
                max_val = current_val;
                max_idx = @as(u16, @intCast(v));
            }
        }
        
        out_pred.* = max_idx;
        out_loss.* = loss;

        // Execute local stochastic update on the Local Head
        try self.local_head.localUpdate(last_token_packed, local_error, shift_scale, rng, allocator);
        
        // Note: For true deep intermediate learning, we would also backpropagate the
        // error *locally* into the Attention and MLP weights of THIS block.
        // For the PoC, ensuring the local head exists forces the block's residual
        // state to be meaningful for sequence prediction at this layer depth.
    }
    
    pub fn resetCache(self: *TransformerBlock) void {
        self.ssm.resetCache();
    }
    
    /// Inference Forward Pass using KV Cache
    pub fn inferenceForward(
        self: *TransformerBlock, 
        allocator: std.mem.Allocator, 
        token_i32: []i32, 
        pool: *const thread_pool.ThreadPool
    ) !void {
        const token_i8 = try allocator.alloc(i8, self.embed_dim);
        defer allocator.free(token_i8);
        
        const packed_dim = self.embed_dim / 64;
        const token_packed = try allocator.alloc(u64, packed_dim);
        defer allocator.free(token_packed);
        
        const ssm_out = try allocator.alloc(i32, self.embed_dim);
        defer allocator.free(ssm_out);
        
        const mlp_out = try allocator.alloc(i32, self.embed_dim);
        defer allocator.free(mlp_out);

        // 1. Norm + SSM (State-Space)
        BitMaxNorm.forward(token_i32, token_i8);
        try self.ssm.forward(allocator, token_i8, ssm_out, pool);
        for (0..self.embed_dim) |i| token_i32[i] += ssm_out[i];

        // 2. Norm + MLP
        BitMaxNorm.forward(token_i32, token_i8);
        binary_kernel.packBinary(token_i8, token_packed);
        try self.mlp.forward(allocator, token_packed, mlp_out, pool);
        for (0..self.embed_dim) |i| token_i32[i] += mlp_out[i];
    }
    
    pub fn save(self: *TransformerBlock, writer: anytype) !void {
        try self.ssm.save(writer);
        try self.mlp.save(writer);
        try self.local_head.save(writer);
    }
    
    pub fn load(self: *TransformerBlock, reader: anytype) !void {
        try self.ssm.load(reader);
        try self.mlp.load(reader);
        try self.local_head.load(reader);
    }
};
