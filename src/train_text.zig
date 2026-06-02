const std = @import("std");
const tokenizer = @import("tokenizer.zig");
const block = @import("block.zig");
const transformer = @import("transformer.zig");
const stochastic = @import("stochastic.zig");
const thread_pool = @import("thread_pool.zig");

// A sequence training loop testing deep stacking and multi-threading on the CPU Engine
pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const text = @embedFile("tiny_shakespeare.txt");
    
    var tok = try tokenizer.Tokenizer.init(allocator, text);
    defer tok.deinit();
    
    const encoded_text = try allocator.alloc(u8, text.len);
    defer allocator.free(encoded_text);
    tok.encode(text, encoded_text);
    
    const vocab_size = tok.vocab_size;
    
    // Setup 8-Core Thread Pool
    const pool = thread_pool.ThreadPool.init(allocator, 8);
    std.debug.print("--- Initialized 8-Core Thread Pool ---\n", .{});

    // 3. Build Deep Model (4 Transformer Blocks + Classifier Head)
    const seq_len = 16;
    const embed_dim = vocab_size; // Kept at vocab size to avoid an embedding lookup matrix for PoC
    const num_layers = 4;
    
    var layers = try allocator.alloc(block.TransformerBlock, num_layers);
    defer allocator.free(layers);
    
    var rng = stochastic.VectorXorshift32.init(9999);
    
    for (0..num_layers) |l| {
        layers[l] = try block.TransformerBlock.init(allocator, embed_dim);
        layers[l].randomize(&rng);
    }
    defer for (0..num_layers) |l| layers[l].deinit(allocator);

    var head = try transformer.BitLinear.init(allocator, embed_dim, vocab_size);
    defer head.deinit(allocator);
    for (0..head.shadow_weights.len) |i| head.shadow_weights[i] = 0;
    head.syncWeights();

    // Buffers
    var x_i32 = try allocator.alloc(i32, seq_len * embed_dim);
    defer allocator.free(x_i32);
    
    const head_out = try allocator.alloc(i32, vocab_size);
    defer allocator.free(head_out);
    
    var local_error = try allocator.alloc(i32, vocab_size);
    defer allocator.free(local_error);

    const num_epochs = 1000;
    const start_idx = 0;
    
    for (0..num_epochs) |epoch| {
        // --- PREPARE SEQUENCE BATCH ---
        @memset(x_i32, 0);
        for (0..seq_len) |t| {
            const char_id = encoded_text[start_idx + t];
            // Initialize integer sequence directly into the activation buffer
            x_i32[t * embed_dim + char_id] = 100; // Scaled up to give Norm block some signal
        }
        
        const target_char_id = encoded_text[start_idx + seq_len]; // Next character
        
        // --- DEEP FORWARD PASS ---
        for (0..num_layers) |l| {
            try layers[l].forward(allocator, x_i32, &pool);
        }
        
        // We only care about predicting the next token from the LAST contextualized token
        const last_token_ctx = x_i32[(seq_len - 1) * embed_dim .. seq_len * embed_dim];
        
        // Quantize back to i8 for the classification head
        const last_token_i8 = try allocator.alloc(i8, embed_dim);
        defer allocator.free(last_token_i8);
        for (0..embed_dim) |d| {
            const val = last_token_ctx[d];
            var quantized: i8 = 0;
            if (val > 0) quantized = 1;
            if (val < 0) quantized = -1;
            last_token_i8[d] = quantized;
        }
        
        // Classification Head
        try head.forwardThreaded(last_token_i8, head_out, &pool);
        
        // --- COMPUTE LOCAL ERROR ---
        var loss: i64 = 0;
        @memset(local_error, 0);
        for (0..vocab_size) |v| {
            const target_val: i32 = if (v == target_char_id) 1 else 0;
            const current_val = head_out[v];
            const diff = current_val - target_val;
            loss += @as(i64, diff) * @as(i64, diff);
            local_error[v] = diff;
        }
        
        // --- LOCAL UPDATE ---
        try head.localUpdate(last_token_i8, local_error, 4, &rng, allocator);
        
        if (epoch % 100 == 0 or epoch == num_epochs - 1) {
            var max_val: i32 = -999999;
            var max_idx: u8 = 0;
            for (0..vocab_size) |v| {
                if (head_out[v] > max_val) {
                    max_val = head_out[v];
                    max_idx = @as(u8, @intCast(v));
                }
            }
            
            var pred_str: [1]u8 = undefined;
            tok.decode(&[_]u8{max_idx}, &pred_str);
            var targ_str: [1]u8 = undefined;
            tok.decode(&[_]u8{target_char_id}, &targ_str);
            
            std.debug.print("Epoch {d:0>4} | Loss: {d:0>6} | Pred: '{s}' | Target: '{s}'\n", .{epoch, loss, pred_str, targ_str});
        }
    }
    
    std.debug.print("--- Training Complete ---\n", .{});
}
