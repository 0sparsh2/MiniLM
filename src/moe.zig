const std = @import("std");
const transformer = @import("transformer.zig");
const thread_pool = @import("thread_pool.zig");
const stochastic = @import("stochastic.zig");

/// A Mixture-of-Experts layer that routes a token to a single specific BitMLP expert.
/// This drastically reduces compute overhead by only activating a fraction of the network.
pub const SparseMoE = struct {
    num_experts: usize,
    embed_dim: usize,
    
    // The Router is just a simple linear layer that outputs logits for each expert.
    router: transformer.BitLinear,
    
    // Array of expert MLPs
    experts: []transformer.BitMLP,
    
    // Intermediate buffer for routing
    router_out: []i32,

    pub fn init(allocator: std.mem.Allocator, embed_dim: usize, hidden_dim: usize, num_experts: usize) !SparseMoE {
        const router = try transformer.BitLinear.init(allocator, embed_dim, num_experts);
        const experts = try allocator.alloc(transformer.BitMLP, num_experts);
        const router_out = try allocator.alloc(i32, num_experts);
        
        for (0..num_experts) |i| {
            experts[i] = try transformer.BitMLP.init(allocator, embed_dim, hidden_dim);
        }
        
        return .{
            .num_experts = num_experts,
            .embed_dim = embed_dim,
            .router = router,
            .experts = experts,
            .router_out = router_out,
        };
    }

    pub fn deinit(self: *SparseMoE, allocator: std.mem.Allocator) void {
        self.router.deinit(allocator);
        for (0..self.num_experts) |i| {
            self.experts[i].deinit(allocator);
        }
        allocator.free(self.experts);
        allocator.free(self.router_out);
    }
    
    pub fn syncWeights(self: *SparseMoE) void {
        self.router.syncWeights();
        for (0..self.num_experts) |i| {
            self.experts[i].fc1.syncWeights();
            self.experts[i].fc2.syncWeights();
        }
    }
    
    pub fn randomize(self: *SparseMoE, rng: *stochastic.VectorXorshift32) void {
        // Randomize router
        for (0..self.router.shadow_weights.len) |i| {
            self.router.shadow_weights[i] = @as(i16, @intCast(rng.next()[0] % 50)) - 25;
        }
        self.router.syncWeights();
        
        // Randomize experts
        for (0..self.num_experts) |e| {
            var fc1 = &self.experts[e].fc1;
            var fc2 = &self.experts[e].fc2;
            
            for (0..fc1.shadow_weights.len) |i| {
                fc1.shadow_weights[i] = @as(i16, @intCast(rng.next()[0] % 50)) - 25;
            }
            fc1.syncWeights();
            
            for (0..fc2.shadow_weights.len) |i| {
                fc2.shadow_weights[i] = @as(i16, @intCast(rng.next()[0] % 50)) - 25;
            }
            fc2.syncWeights();
        }
    }

    /// Forward pass executing only the Top-1 Expert for maximum CPU speed.
    pub fn forward(
        self: *SparseMoE, 
        allocator: std.mem.Allocator, 
        x_packed: []const u64, 
        out_i32: []i32, 
        pool: *const thread_pool.ThreadPool
    ) !void {
        const packed_dim = self.embed_dim / 64;
        const seq_len = x_packed.len / packed_dim;
        std.debug.assert(out_i32.len == seq_len * self.embed_dim);
        
        @memset(out_i32, 0);

        for (0..seq_len) |t| {
            const token_packed = x_packed[t * packed_dim .. (t + 1) * packed_dim];
            const token_out = out_i32[t * self.embed_dim .. (t + 1) * self.embed_dim];
            
            // 1. Run Router
            self.router.forward(token_packed, self.router_out);
            
            // 2. Pick Top-1 Expert
            var max_val: i32 = -999999;
            var best_expert: usize = 0;
            
            for (0..self.num_experts) |i| {
                if (self.router_out[i] > max_val) {
                    max_val = self.router_out[i];
                    best_expert = i;
                }
            }
            
            // 3. Execute only the chosen expert!
            try self.experts[best_expert].forward(allocator, token_packed, token_out, pool);
        }
    }
    
    pub fn save(self: *SparseMoE, writer: anytype) !void {
        try self.router.save(writer);
        for (0..self.num_experts) |i| {
            try self.experts[i].save(writer);
        }
    }
    
    pub fn load(self: *SparseMoE, reader: anytype) !void {
        try self.router.load(reader);
        for (0..self.num_experts) |i| {
            try self.experts[i].load(reader);
        }
    }
};
