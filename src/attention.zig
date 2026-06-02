const std = @import("std");
const transformer = @import("transformer.zig");
const thread_pool = @import("thread_pool.zig");

pub const BitAttention = struct {
    wq: transformer.BitLinear,
    wk: transformer.BitLinear,
    wv: transformer.BitLinear,
    wo: transformer.BitLinear,
    embed_dim: usize,
    
    // KV Cache for instantaneous inference
    max_seq_len: usize,
    cache_len: usize,
    cache_K: []i8,
    cache_V: []i8,

    pub fn init(allocator: std.mem.Allocator, embed_dim: usize, max_seq_len: usize) !BitAttention {
        const cache_K = try allocator.alloc(i8, max_seq_len * embed_dim);
        const cache_V = try allocator.alloc(i8, max_seq_len * embed_dim);
        @memset(cache_K, 0);
        @memset(cache_V, 0);

        return .{
            .wq = try transformer.BitLinear.init(allocator, embed_dim, embed_dim),
            .wk = try transformer.BitLinear.init(allocator, embed_dim, embed_dim),
            .wv = try transformer.BitLinear.init(allocator, embed_dim, embed_dim),
            .wo = try transformer.BitLinear.init(allocator, embed_dim, embed_dim),
            .embed_dim = embed_dim,
            .max_seq_len = max_seq_len,
            .cache_len = 0,
            .cache_K = cache_K,
            .cache_V = cache_V,
        };
    }

    pub fn deinit(self: *BitAttention, allocator: std.mem.Allocator) void {
        self.wq.deinit(allocator);
        self.wk.deinit(allocator);
        self.wv.deinit(allocator);
        self.wo.deinit(allocator);
        allocator.free(self.cache_K);
        allocator.free(self.cache_V);
    }

    pub fn syncWeights(self: *BitAttention) void {
        self.wq.syncWeights();
        self.wk.syncWeights();
        self.wv.syncWeights();
        self.wo.syncWeights();
    }
    
    pub fn resetCache(self: *BitAttention) void {
        self.cache_len = 0;
    }

    /// Training Forward Pass (Sliding Window, No Cache)
    pub fn forward(self: *BitAttention, allocator: std.mem.Allocator, x: []const i8, out: []i32, pool: *const thread_pool.ThreadPool) !void {
        const seq_len = x.len / self.embed_dim;
        std.debug.assert(out.len == seq_len * self.embed_dim);

        var q_i32 = try allocator.alloc(i32, seq_len * self.embed_dim);
        defer allocator.free(q_i32);
        var k_i32 = try allocator.alloc(i32, seq_len * self.embed_dim);
        defer allocator.free(k_i32);
        var v_i32 = try allocator.alloc(i32, seq_len * self.embed_dim);
        defer allocator.free(v_i32);

        for (0..seq_len) |t| {
            const token_x = x[t * self.embed_dim .. (t + 1) * self.embed_dim];
            try self.wq.forwardThreaded(token_x, q_i32[t * self.embed_dim .. (t + 1) * self.embed_dim], pool);
            try self.wk.forwardThreaded(token_x, k_i32[t * self.embed_dim .. (t + 1) * self.embed_dim], pool);
            try self.wv.forwardThreaded(token_x, v_i32[t * self.embed_dim .. (t + 1) * self.embed_dim], pool);
        }

        const q_i8 = try allocator.alloc(i8, q_i32.len);
        defer allocator.free(q_i8);
        const k_i8 = try allocator.alloc(i8, k_i32.len);
        defer allocator.free(k_i8);
        const v_i8 = try allocator.alloc(i8, v_i32.len);
        defer allocator.free(v_i8);

        quantizeTernary(q_i32, q_i8);
        quantizeTernary(k_i32, k_i8);
        quantizeTernary(v_i32, v_i8);

        var attn_out_i32 = try allocator.alloc(i32, seq_len * self.embed_dim);
        defer allocator.free(attn_out_i32);
        @memset(attn_out_i32, 0);

        for (0..seq_len) |t_q| {
            const q_t = q_i8[t_q * self.embed_dim .. (t_q + 1) * self.embed_dim];
            
            for (0..t_q + 1) |t_k| {
                const k_t = k_i8[t_k * self.embed_dim .. (t_k + 1) * self.embed_dim];
                
                var score: i32 = 0;
                for (0..self.embed_dim) |d| {
                    score += @as(i32, q_t[d]) * @as(i32, k_t[d]);
                }
                
                // ReLU Max-Scaling instead of Softmax
                if (score > 0) {
                    const v_t = v_i8[t_k * self.embed_dim .. (t_k + 1) * self.embed_dim];
                    for (0..self.embed_dim) |d| {
                        attn_out_i32[t_q * self.embed_dim + d] += score * @as(i32, v_t[d]);
                    }
                }
            }
        }

        const attn_out_i8 = try allocator.alloc(i8, attn_out_i32.len);
        defer allocator.free(attn_out_i8);
        quantizeTernary(attn_out_i32, attn_out_i8);

        for (0..seq_len) |t| {
            const in_token = attn_out_i8[t * self.embed_dim .. (t + 1) * self.embed_dim];
            const out_token = out[t * self.embed_dim .. (t + 1) * self.embed_dim];
            try self.wo.forwardThreaded(in_token, out_token, pool);
        }
    }
    
    /// Inference Forward Pass (Single Token with KV Caching)
    pub fn inferenceForward(self: *BitAttention, allocator: std.mem.Allocator, token_x: []const i8, out: []i32, pool: *const thread_pool.ThreadPool) !void {
        std.debug.assert(token_x.len == self.embed_dim);
        std.debug.assert(out.len == self.embed_dim);
        
        if (self.cache_len >= self.max_seq_len) {
            std.debug.print("KV CACHE FULL! Sequence length exceeded max.\n", .{});
            return;
        }

        // 1. Calculate Q, K, V for just this ONE token
        const q_i32 = try allocator.alloc(i32, self.embed_dim);
        defer allocator.free(q_i32);
        const k_i32 = try allocator.alloc(i32, self.embed_dim);
        defer allocator.free(k_i32);
        const v_i32 = try allocator.alloc(i32, self.embed_dim);
        defer allocator.free(v_i32);

        try self.wq.forwardThreaded(token_x, q_i32, pool);
        try self.wk.forwardThreaded(token_x, k_i32, pool);
        try self.wv.forwardThreaded(token_x, v_i32, pool);

        const q_i8 = try allocator.alloc(i8, self.embed_dim);
        defer allocator.free(q_i8);
        const k_i8 = try allocator.alloc(i8, self.embed_dim);
        defer allocator.free(k_i8);
        const v_i8 = try allocator.alloc(i8, self.embed_dim);
        defer allocator.free(v_i8);

        quantizeTernary(q_i32, q_i8);
        quantizeTernary(k_i32, k_i8);
        quantizeTernary(v_i32, v_i8);
        
        // 2. Append K and V to the cache
        @memcpy(self.cache_K[self.cache_len * self.embed_dim .. (self.cache_len + 1) * self.embed_dim], k_i8);
        @memcpy(self.cache_V[self.cache_len * self.embed_dim .. (self.cache_len + 1) * self.embed_dim], v_i8);
        
        self.cache_len += 1;

        // 3. Compute Attention ONLY against the Cache
        var attn_out_i32 = try allocator.alloc(i32, self.embed_dim);
        defer allocator.free(attn_out_i32);
        @memset(attn_out_i32, 0);

        for (0..self.cache_len) |t_k| {
            const cache_k_t = self.cache_K[t_k * self.embed_dim .. (t_k + 1) * self.embed_dim];
            
            var score: i32 = 0;
            for (0..self.embed_dim) |d| {
                score += @as(i32, q_i8[d]) * @as(i32, cache_k_t[d]);
            }
            
            if (score > 0) {
                const cache_v_t = self.cache_V[t_k * self.embed_dim .. (t_k + 1) * self.embed_dim];
                for (0..self.embed_dim) |d| {
                    attn_out_i32[d] += score * @as(i32, cache_v_t[d]);
                }
            }
        }

        // 4. Output Projection
        const attn_out_i8 = try allocator.alloc(i8, self.embed_dim);
        defer allocator.free(attn_out_i8);
        quantizeTernary(attn_out_i32, attn_out_i8);

        try self.wo.forwardThreaded(attn_out_i8, out, pool);
    }

    fn quantizeTernary(in: []const i32, out: []i8) void {
        for (in, 0..) |val, i| {
            var q: i8 = 0;
            if (val > 0) q = 1;
            if (val < 0) q = -1;
            out[i] = q;
        }
    }
};
