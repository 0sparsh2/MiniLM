const std = @import("std");
const bpe = @import("bpe.zig");
const block = @import("block.zig");
const transformer = @import("transformer.zig");
const stochastic = @import("stochastic.zig");
const thread_pool = @import("thread_pool.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("--- Loading BPE Tokenizer ---\n", .{});
    const text = @embedFile("wiki_5mb.txt");
    
    var tok = bpe.BPETokenizer.init(allocator);
    defer tok.deinit();
    try tok.train(text, 256);
    
    const vocab_size = tok.vocab_size;
    std.debug.print("Vocab Size: {d}\n", .{vocab_size});

    const pool = thread_pool.ThreadPool.init(allocator, 8);
    
    const embed_dim = 128;
    const num_layers = 4;
    const max_seq_len = 100; // Generate up to 100 tokens
    
    var layers = try allocator.alloc(block.TransformerBlock, num_layers);
    defer allocator.free(layers);
    
    var rng = stochastic.VectorXorshift32.init(123);
    
    for (0..num_layers) |l| {
        layers[l] = try block.TransformerBlock.init(allocator, embed_dim, vocab_size, max_seq_len);
        layers[l].randomize(&rng);
    }
    defer for (0..num_layers) |l| layers[l].deinit(allocator);

    var embed_layer = try transformer.BitLinear.init(allocator, vocab_size, embed_dim);
    defer embed_layer.deinit(allocator);
    for (0..embed_layer.shadow_weights.len) |i| embed_layer.shadow_weights[i] = @as(i16, @intCast(rng.next()[0] % 50)) - 25;
    embed_layer.syncWeights();

    var token_i8 = try allocator.alloc(i8, vocab_size);
    defer allocator.free(token_i8);
    const token_i32 = try allocator.alloc(i32, embed_dim);
    defer allocator.free(token_i32);
    
    // We will use the last block's local head as the final classifier
    const classifier = &layers[num_layers - 1].local_head;
    const logits = try allocator.alloc(i32, vocab_size);
    defer allocator.free(logits);
    
    std.debug.print("--- Starting Instantaneous KV Inference ---\n", .{});
    std.debug.print("Prompt: ", .{});
    
    var current_token_id: u16 = 32; // Just pick a random start token (e.g. space)
    
    for (0..max_seq_len) |_| {
        // 1. One-hot the current token
        @memset(token_i8, 0);
        if (current_token_id < vocab_size) token_i8[current_token_id] = 1;
        
        // 2. Embed
        try embed_layer.forwardThreaded(token_i8, token_i32, &pool);
        
        // 3. Deep Forward Pass using KV Cache
        for (0..num_layers) |l| {
            try layers[l].inferenceForward(allocator, token_i32, &pool);
        }
        
        // 4. Classify (Greedy argmax)
        // Quantize context
        var ctx_i8 = try allocator.alloc(i8, embed_dim);
        defer allocator.free(ctx_i8);
        for (0..embed_dim) |d| {
            var q: i8 = 0;
            if (token_i32[d] > 0) q = 1;
            if (token_i32[d] < 0) q = -1;
            ctx_i8[d] = q;
        }
        
        try classifier.forwardThreaded(ctx_i8, logits, &pool);
        
        var max_val: i32 = -999999;
        var next_token_id: u16 = 0;
        for (0..vocab_size) |v| {
            if (logits[v] > max_val) {
                max_val = logits[v];
                next_token_id = @as(u16, @intCast(v));
            }
        }
        
        // Fallback or randomness could go here. We just pick max.
        current_token_id = next_token_id;
        
        // Decode Token (Hack: we just print the raw ID for now since BPE decoding isn't fully written)
        std.debug.print("[{d}] ", .{current_token_id});
    }
    
    std.debug.print("\n\nGenerated {d} tokens!\n", .{max_seq_len});
}
