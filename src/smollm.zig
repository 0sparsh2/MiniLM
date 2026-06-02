const std = @import("std");
const binary_kernel = @import("binary_kernel.zig");

// ============================================================
// Config
// ============================================================
const Header = extern struct {
    magic: u32,
    version: u32,
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    rms_norm_eps: f32,
};

const BlockQ4_0 = extern struct {
    d: f32,
    qs: [16]u8,
};

const LayerWeights = struct {
    input_layernorm: []f32,
    post_attention_layernorm: []f32,
    q_proj: []BlockQ4_0,
    k_proj: []BlockQ4_0,
    v_proj: []BlockQ4_0,
    o_proj: []BlockQ4_0,
    gate_proj: []BlockQ4_0,
    up_proj: []BlockQ4_0,
    down_proj: []BlockQ4_0,
};

const Model = struct {
    header: Header,
    embed_tokens: []BlockQ4_0,
    layers: []LayerWeights,
    norm_weight: []f32,
    lm_head: []BlockQ4_0,
    // Vocabulary strings for decoding tokens
    vocab: [][]u8,
};

// ============================================================
// File Reading Helpers
// ============================================================
const FdReader = struct {
    fd: std.posix.fd_t,
    pub const ReadError = std.posix.ReadError;
    pub fn readAll(self: @This(), buf: []u8) ReadError!usize {
        var total: usize = 0;
        while (total < buf.len) {
            const n = try std.posix.read(self.fd, buf[total..]);
            if (n == 0) break;
            total += n;
        }
        return total;
    }
    pub fn readNoEof(self: @This(), buf: []u8) !void {
        var total: usize = 0;
        while (total < buf.len) {
            const n = try std.posix.read(self.fd, buf[total..]);
            if (n == 0) return error.UnexpectedEof;
            total += n;
        }
    }
};

fn readF32(reader: FdReader) !f32 {
    var buf: [4]u8 = undefined;
    try reader.readNoEof(&buf);
    return @bitCast(buf);
}

fn readSliceF32(reader: FdReader, allocator: std.mem.Allocator, count: usize) ![]f32 {
    const data = try allocator.alloc(f32, count);
    try reader.readNoEof(std.mem.sliceAsBytes(data));
    return data;
}

fn readSliceI8(reader: FdReader, allocator: std.mem.Allocator, count: usize) ![]i8 {
    const data = try allocator.alloc(i8, count);
    try reader.readNoEof(std.mem.sliceAsBytes(data));
    return data;
}

fn readSliceQ4_0(reader: FdReader, allocator: std.mem.Allocator, num_weights: usize) ![]BlockQ4_0 {
    std.debug.assert(num_weights % 32 == 0);
    const num_blocks = num_weights / 32;
    const data = try allocator.alloc(BlockQ4_0, num_blocks);
    try reader.readNoEof(std.mem.sliceAsBytes(data));
    return data;
}

fn readSliceU64(reader: FdReader, allocator: std.mem.Allocator, rows: usize, cols: usize) ![]u64 {
    const total_bits = rows * cols;
    const num_u64 = (total_bits + 63) / 64;
    const data = try allocator.alloc(u64, num_u64);
    try reader.readNoEof(std.mem.sliceAsBytes(data));
    return data;
}

// ------------------------------------------------------------
// Load a newline-separated tokenizer vocabulary using raw POSIX calls.
// Returns a slice where vocab[i] is the UTF-8 string for token ID i.
// Avoids std.fs (restructured in Zig 0.16) and std.ArrayList (now unmanaged).
fn loadVocab(allocator: std.mem.Allocator, path: []const u8) ![][]u8 {
    // Open file with a null-terminated path.
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);
    const fd = try std.posix.openatZ(std.posix.AT.FDCWD, path_z, .{ .ACCMODE = .RDONLY }, 0);
    defer _ = std.c.close(fd);

    // Read the entire file into an allocator-owned buffer, growing as needed.
    var buf: []u8 = try allocator.alloc(u8, 0);
    var total: usize = 0;
    while (true) {
        // Grow buffer by 64 KB if needed.
        if (total == buf.len) {
            buf = try allocator.realloc(buf, buf.len + 65536);
        }
        const n = try std.posix.read(fd, buf[total..]);
        if (n == 0) break;
        total += n;
    }
    const file_data = buf[0..total];
    defer allocator.free(buf);

    // Count newlines to pre-allocate the vocab slice.
    var num_lines: usize = 0;
    for (file_data) |c| if (c == '\n') { num_lines += 1; };
    // Handle files with no trailing newline.
    if (total > 0 and file_data[total - 1] != '\n') num_lines += 1;

    const vocab = try allocator.alloc([]u8, num_lines);
    var idx: usize = 0;
    var it = std.mem.splitScalar(u8, file_data, '\n');
    while (it.next()) |line| {
        // Skip trailing empty segment after final newline.
        if (line.len == 0 and it.peek() == null) break;
        vocab[idx] = try allocator.dupe(u8, line);
        idx += 1;
    }
    return vocab[0..idx];
}


// ============================================================
// Model Loading
// ============================================================
fn loadModel(allocator: std.mem.Allocator, bin_path: [:0]const u8, vocab_path: []const u8) !Model {
    const fd = try std.posix.openatZ(std.posix.AT.FDCWD, bin_path, .{ .ACCMODE = .RDONLY }, 0);
    const reader = FdReader{ .fd = fd };

    // Parse header
    var header: Header = undefined;
    try reader.readNoEof(std.mem.asBytes(&header));
    if (header.magic != 0x42495453) return error.InvalidMagic;
    if (header.version != 9) return error.InvalidVersion;

    const H = header.hidden_size;
    const I = header.intermediate_size;
    const V = header.vocab_size;
    const L = header.num_hidden_layers;
    const NH = header.num_attention_heads;
    const NKV = header.num_key_value_heads;
    const head_dim = H / NH;
    const kv_dim = head_dim * NKV;

    std.debug.print("SmolLM-135M | Layers: {} | Dim: {} | Vocab: {} | Heads: {}/{}\n",
        .{L, H, V, NH, NKV});

    // Q4_0 Embeddings
    const embed_tokens = try readSliceQ4_0(reader, allocator, V * H);

    // Load layers
    var layers = try allocator.alloc(LayerWeights, L);
    for (0..L) |i| {
        layers[i] = .{
            .input_layernorm = try readSliceF32(reader, allocator, H),
            .post_attention_layernorm = try readSliceF32(reader, allocator, H),
            .q_proj = try readSliceQ4_0(reader, allocator, H * H),
            .k_proj = try readSliceQ4_0(reader, allocator, kv_dim * H),
            .v_proj = try readSliceQ4_0(reader, allocator, kv_dim * H),
            .o_proj = try readSliceQ4_0(reader, allocator, H * H),
            .gate_proj = try readSliceQ4_0(reader, allocator, I * H),
            .up_proj = try readSliceQ4_0(reader, allocator, I * H),
            .down_proj = try readSliceQ4_0(reader, allocator, H * I),
        };
        if (i % 10 == 0) std.debug.print("  Loaded layer {d}\n", .{i});
    }

    const norm_weight = try readSliceF32(reader, allocator, H);
    
    // Q4_0 LM Head
    const lm_head = try readSliceQ4_0(reader, allocator, V * H);

    const vocab = try loadVocab(allocator, vocab_path);

    return Model{
        .header = header,
        .embed_tokens = embed_tokens,
        .layers = layers,
        .norm_weight = norm_weight,
        .lm_head = lm_head,
        .vocab = vocab,
    };
}

// ============================================================
// Math Primitives
// ============================================================

/// RMSNorm: normalize x by its RMS, then scale by weight
fn rmsNorm(x: []const f32, weight: []const f32, out: []f32, eps: f32) void {
    var sum: f32 = 0.0;
    for (x) |v| sum += v * v;
    const rms = @sqrt(sum / @as(f32, @floatFromInt(x.len)) + eps);
    const scale = 1.0 / rms;
    for (0..x.len) |i| out[i] = x[i] * scale * weight[i];
}

/// Q4_0 block-wise matrix-vector multiply
fn q4_0MatMul(
    w_blocks: []const BlockQ4_0,
    x: []const f32,
    out: []f32,
    rows: usize,
    in_dim: usize,
) void {
    const blocks_per_row = in_dim / 32;
    for (0..rows) |i| {
        var sum: f32 = 0.0;
        const row_blocks = w_blocks[i * blocks_per_row .. (i + 1) * blocks_per_row];
        
        for (0..blocks_per_row) |b| {
            const block = row_blocks[b];
            const scale = block.d;
            var block_sum: f32 = 0.0;
            const x_block = x[b * 32 .. (b + 1) * 32];
            
            for (0..16) |c| {
                const packed_byte = block.qs[c];
                const even = (packed_byte >> 4) & 0x0F;
                const odd = packed_byte & 0x0F;
                
                const w_even = @as(f32, @floatFromInt(even)) - 8.0;
                const w_odd = @as(f32, @floatFromInt(odd)) - 8.0;
                
                block_sum += w_even * x_block[c * 2] + w_odd * x_block[c * 2 + 1];
            }
            sum += block_sum * scale;
        }
        out[i] = sum;
    }
}



/// Softmax in-place
fn softmax(x: []f32) void {
    var max: f32 = x[0];
    for (x[1..]) |v| if (v > max) { max = v; };
    var sum: f32 = 0.0;
    for (x) |*v| { v.* = @exp(v.* - max); sum += v.*; }
    for (x) |*v| v.* /= sum;
}

/// SiLU activation: x * sigmoid(x)
inline fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

// Rotary Positional Embedding (RoPE)
fn applyRope(x: []f32, pos: usize, head_dim: usize) void {
    const half = head_dim / 2;
    for (0..half) |i| {
        const power = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim));
        const freq: f32 = 1.0 / @exp(@log(@as(f32, 10000.0)) * power);
        const angle: f32 = @as(f32, @floatFromInt(pos)) * freq;
        const cos_a = @cos(angle);
        const sin_a = @sin(angle);
        
        const x0 = x[i];
        const x1 = x[i + half];
        
        x[i] = x0 * cos_a - x1 * sin_a;
        x[i + half] = x0 * sin_a + x1 * cos_a;
    }
}

// ============================================================
// Forward Pass Core Functions
// ============================================================


fn forward(
    model: *const Model,
    allocator: std.mem.Allocator,
    token: u32,
    pos: usize,
    max_seq_len: usize,
    k_cache: []f32,
    v_cache: []f32,
    logits: []f32,
) !void {
    const H = model.header.hidden_size;
    const I = model.header.intermediate_size;
    const V = model.header.vocab_size;
    const NH = model.header.num_attention_heads;
    const NKV = model.header.num_key_value_heads;
    const head_dim = H / NH;
    const kv_dim = head_dim * NKV;
    const seq_len = pos + 1;

    // Allocations
    const x = try allocator.alloc(f32, H);
    defer allocator.free(x);
    const xn = try allocator.alloc(f32, H);
    defer allocator.free(xn);

    const q = try allocator.alloc(f32, H);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, kv_dim);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, kv_dim);
    defer allocator.free(v);
    const attn_out = try allocator.alloc(f32, H);
    defer allocator.free(attn_out);

    const gate = try allocator.alloc(f32, I);
    defer allocator.free(gate);
    const up_val = try allocator.alloc(f32, I);
    defer allocator.free(up_val);
    const mlp_out = try allocator.alloc(f32, H);
    defer allocator.free(mlp_out);

    var attn_score = try allocator.alloc(f32, seq_len);
    defer allocator.free(attn_score);

    // Embed token from Q4_0 embedding table
    const blocks_per_row = H / 32;
    const embed_offset = token * blocks_per_row;
    
    for (0..blocks_per_row) |b| {
        const block = model.embed_tokens[embed_offset + b];
        const scale = block.d;
        for (0..16) |c| {
            const packed_byte = block.qs[c];
            const even = (packed_byte >> 4) & 0x0F;
            const odd = packed_byte & 0x0F;
            x[b * 32 + c * 2] = (@as(f32, @floatFromInt(even)) - 8.0) * scale;
            x[b * 32 + c * 2 + 1] = (@as(f32, @floatFromInt(odd)) - 8.0) * scale;
        }
    }

    // Loop over transformer layers
    for (model.layers, 0..) |layer, li| {
        // --- Pre-Attention RMSNorm ---
        rmsNorm(x, layer.input_layernorm, xn, model.header.rms_norm_eps);

        // --- Q, K, V Projections (Q4_0) ---
        q4_0MatMul(layer.q_proj, xn, q, H, H);
        q4_0MatMul(layer.k_proj, xn, k, kv_dim, H);
        q4_0MatMul(layer.v_proj, xn, v, kv_dim, H);

        // --- RoPE on Q and K ---
        for (0..NH) |h| applyRope(q[h * head_dim ..][0..head_dim], pos, head_dim);
        for (0..NKV) |h| applyRope(k[h * head_dim ..][0..head_dim], pos, head_dim);

        // --- Store KV in cache ---
        const layer_offset = li * max_seq_len * kv_dim;
        const cache_pos = layer_offset + pos * kv_dim;
        @memcpy(k_cache[cache_pos .. cache_pos + kv_dim], k);
        @memcpy(v_cache[cache_pos .. cache_pos + kv_dim], v);

        // --- Attention: dot Q with each cached K, softmax, accumulate V ---
        @memset(attn_out, 0.0);
        for (0..NH) |h| {
            const q_head = q[h * head_dim ..][0..head_dim];
            const kv_group = h / (NH / NKV); // GQA grouping
            const scale_attn = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

            for (0..seq_len) |t| {
                const k_head = k_cache[layer_offset + t * kv_dim + kv_group * head_dim ..][0..head_dim];
                var dot: f32 = 0.0;
                for (0..head_dim) |d| dot += q_head[d] * k_head[d];
                attn_score[t] = dot * scale_attn;
            }
            softmax(attn_score[0..seq_len]);

            // Accumulate V
            const attn_h = attn_out[h * head_dim ..][0..head_dim];
            for (0..seq_len) |t| {
                const v_head = v_cache[layer_offset + t * kv_dim + kv_group * head_dim ..][0..head_dim];
                for (0..head_dim) |d| attn_h[d] += attn_score[t] * v_head[d];
            }
        }

        // --- Output Projection (Q4_0) ---
        const o_out = try allocator.alloc(f32, H);
        defer allocator.free(o_out);
        q4_0MatMul(layer.o_proj, attn_out, o_out, H, H);
        // Residual
        for (0..H) |d| x[d] += o_out[d];

        // --- Post-Attention RMSNorm ---
        rmsNorm(x, layer.post_attention_layernorm, xn, model.header.rms_norm_eps);

        // --- SwiGLU MLP ---
        q4_0MatMul(layer.gate_proj, xn, gate, I, H);
        q4_0MatMul(layer.up_proj, xn, up_val, I, H);
        for (0..I) |d| gate[d] = silu(gate[d]) * up_val[d];
        
        q4_0MatMul(layer.down_proj, gate, mlp_out, H, I);
        // Residual
        for (0..H) |d| x[d] += mlp_out[d];

    }

    // Final RMSNorm
    rmsNorm(x, model.norm_weight, xn, model.header.rms_norm_eps);

    // LM Head: reuse embed_tokens (Q4_0)
    for (0..V) |tok| {
        var dot: f32 = 0.0;
        const lm_blocks_per_row = H / 32;
        const offset = tok * lm_blocks_per_row;
        
        for (0..lm_blocks_per_row) |b| {
            const block = model.lm_head[offset + b];
            const scale = block.d;
            var block_sum: f32 = 0.0;
            const xn_block = xn[b * 32 .. (b + 1) * 32];
            
            for (0..16) |c| {
                const packed_byte = block.qs[c];
                const even = (packed_byte >> 4) & 0x0F;
                const odd = packed_byte & 0x0F;
                const w_even = (@as(f32, @floatFromInt(even)) - 8.0);
                const w_odd = (@as(f32, @floatFromInt(odd)) - 8.0);
                block_sum += xn_block[c * 2] * w_even + xn_block[c * 2 + 1] * w_odd;
            }
            dot += block_sum * scale;
        }
        logits[tok] = dot;
    }
}

// Temperature sampling helper
fn sample(logits: []f32, temperature: f32, rng: *std.Random.DefaultPrng) u32 {
    if (temperature > 0.0) {
        for (logits) |*lv| lv.* /= temperature;
        softmax(logits);
        var cumulative: f32 = 0.0;
        const r = rng.random().float(f32);
        for (logits, 0..) |p, i| {
            cumulative += p;
            if (cumulative >= r) return @intCast(i);
        }
        return @intCast(logits.len - 1);
    } else {
        // Greedy
        var best_tok: u32 = 0;
        var best_logit: f32 = logits[0];
        for (logits[1..], 1..) |lv, i| {
            if (lv > best_logit) { best_logit = lv; best_tok = @intCast(i); }
        }
        return best_tok;
    }
}

// ============================================================
// Simple byte-level tokenizer (matches our BPE approach)
// ============================================================

pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(init.arena.allocator());
    defer arena.deinit();
    const allocator = arena.allocator();

    std.debug.print("--- SmolLM-135M 1-Bit CPU Inference ---\n", .{});
    std.debug.print("Loading model weights (40MB)...\n", .{});

    var args = init.minimal.args;
    var iter = args.iterate();
    _ = iter.next(); // skip executable name
    
    var bin_path: [:0]const u8 = "model_q4_0.bin";
    var vocab_path: []const u8 = "scripts/smollm_vocab.txt";
    
    if (iter.next()) |b_path| {
        bin_path = b_path;
    }
    if (iter.next()) |v_path| {
        vocab_path = v_path;
    }

    const model = try loadModel(allocator, bin_path, vocab_path);
    std.debug.print("Model loaded! Ready to generate.\n", .{});

    // Outer loop for interactive persistent engine
    var rng = std.Random.DefaultPrng.init(42);
    while (true) {
        var arena_prompt = std.heap.ArenaAllocator.init(allocator);
        defer arena_prompt.deinit();
        const p_alloc = arena_prompt.allocator();

        // Read prompt token IDs from stdin.
        // Protocol: 4 bytes (u32 LE) = number of tokens, then N*4 bytes (u32 LE each).
        var count_buf: [4]u8 = undefined;
        var total_read: usize = 0;
        while (total_read < 4) {
            const n = try std.posix.read(std.posix.STDIN_FILENO, count_buf[total_read..]);
            if (n == 0) return error.UnexpectedEof;
            total_read += n;
        }
        const prompt_len: u32 = std.mem.readInt(u32, &count_buf, .little);

        const max_new_tokens: usize = 200;
        var tokens = try p_alloc.alloc(u32, prompt_len + max_new_tokens + 1);
        for (0..prompt_len) |i| {
            var id_buf: [4]u8 = undefined;
            total_read = 0;
            while (total_read < 4) {
                const n = try std.posix.read(std.posix.STDIN_FILENO, id_buf[total_read..]);
                if (n == 0) return; // EOF
                total_read += n;
            }
            tokens[i] = std.mem.readInt(u32, &id_buf, .little);
        }
        var tok_len: usize = prompt_len;

        const H = model.header.hidden_size;
        const NKV = model.header.num_key_value_heads;
        const head_dim = H / model.header.num_attention_heads;
        const kv_dim = head_dim * NKV;
        const L = model.header.num_hidden_layers;
        const max_seq_len = prompt_len + max_new_tokens + 1;
        
        const k_cache = try p_alloc.alloc(f32, L * max_seq_len * kv_dim);
        const v_cache = try p_alloc.alloc(f32, L * max_seq_len * kv_dim);
        const logits = try p_alloc.alloc(f32, model.header.vocab_size);

        // Prefill prompt
        for (0..prompt_len) |i| {
            try forward(&model, p_alloc, tokens[i], i, max_seq_len, k_cache, v_cache, logits);
        }

        // Generate new tokens.
        for (0..max_new_tokens) |_| {
            const next_tok = sample(logits, 0.7, &rng);
            tokens[tok_len] = next_tok;
            
            // EOS token for SmolLM is 0 or 2.
            if (next_tok == 0 or next_tok == 2) break;
            const token_str = if (next_tok < model.vocab.len) model.vocab[next_tok] else "?";
            _ = std.c.write(std.posix.STDOUT_FILENO, token_str.ptr, token_str.len);
            
            try forward(&model, p_alloc, next_tok, tok_len, max_seq_len, k_cache, v_cache, logits);
            tok_len += 1;
        }
        
        // Signal end of turn with a special byte or just newline
        // We will just write a specific end-of-turn marker 0x04 (EOT)
        const eot: [1]u8 = .{4};
        _ = std.c.write(std.posix.STDOUT_FILENO, &eot, 1);
    }
}
