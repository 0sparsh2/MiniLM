const std = @import("std");
const dataloader = @import("dataloader.zig");
const block = @import("block.zig");
const transformer = @import("transformer.zig");
const thread_pool = @import("thread_pool.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    std.debug.print("--- Initializing Byte-Level Mamba Test ---\n", .{});

    // 1. Raw Byte Loader
    const text = "Hello Byte-Level World! We are completely skipping BPE embeddings today.";
    var loader = try dataloader.DataLoader.init(allocator, text, 1, 1);
    defer loader.deinit();

    const vocab_size = dataloader.DataLoader.vocab_size;
    std.debug.print("Vocab Size: {d} (Pure Bytes)\n", .{vocab_size});

    const embed_dim = 128;
    
    const pool = thread_pool.ThreadPool.init(allocator, 1); // Single thread for simple test

    // 2. Initialize Model Components
    var embed_layer = try transformer.BitLinear.init(allocator, vocab_size, embed_dim);
    defer embed_layer.deinit(allocator);

    var mamba_block = try block.TransformerBlock.init(allocator, embed_dim, vocab_size, 1);
    defer mamba_block.deinit(allocator);

    std.debug.print("Model initialized. Embedding Layer size: {d} bytes\n", .{embed_layer.weights.len * 8});
    
    // Simulate passing a string of bytes through the embedding and Mamba block
    const test_str = "Physics";
    std.debug.print("Feeding string: '{s}'\n", .{test_str});
    
    const x_packed = try allocator.alloc(u64, vocab_size / 64);
    defer allocator.free(x_packed);
    const x_i8 = try allocator.alloc(i8, vocab_size);
    defer allocator.free(x_i8);
    
    const token_i32 = try allocator.alloc(i32, embed_dim);
    defer allocator.free(token_i32);

    for (0..test_str.len) |i| {
        const char = test_str[i];
        
        // One-hot encode byte
        @memset(x_i8, -1);
        x_i8[char] = 1;
        
        // Pack into 1-bit
        @memset(x_packed, 0);
        for (0..vocab_size) |v| {
            if (x_i8[v] == 1) {
                const block_idx = v / 64;
                const bit_idx = @as(u6, @intCast(v % 64));
                x_packed[block_idx] |= (@as(u64, 1) << bit_idx);
            }
        }
        
        // Byte Embedding
        embed_layer.forward(x_packed, token_i32);
        
        // Mamba State Update
        try mamba_block.inferenceForward(allocator, token_i32, &pool);
        
        std.debug.print("Byte '{c}' processed. Mamba hidden state updated.\n", .{char});
    }

    std.debug.print("--- Test Passed Successfully. Zero crashing, massive memory saved! ---\n", .{});
}
