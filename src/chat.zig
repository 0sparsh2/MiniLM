const std = @import("std");
const dataloader = @import("dataloader.zig");
const block = @import("block.zig");
const transformer = @import("transformer.zig");
const thread_pool = @import("thread_pool.zig");
const binary_kernel = @import("binary_kernel.zig");

pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(init.arena.allocator());
    defer arena.deinit();
    const allocator = arena.allocator();

    std.debug.print("--- Loading BiSamba-CPU Interactive Chat ---\n", .{});
    
    // 1. Raw Bytes setup (Vocab is just 256)
    const text = @embedFile("wiki_5mb.txt");
    var loader = try dataloader.DataLoader.init(allocator, text, 1, 1);
    defer loader.deinit();
    const vocab_size = dataloader.DataLoader.vocab_size;
    std.debug.print("Vocabulary Loaded. Size: {d}\n", .{vocab_size});

    const pool = thread_pool.ThreadPool.init(allocator, 8);
    const embed_dim = 128;
    const num_layers = 4;
    
    // 2. Initialize Model
    var layers = try allocator.alloc(block.TransformerBlock, num_layers);
    defer allocator.free(layers);
    for (0..num_layers) |l| {
        layers[l] = try block.TransformerBlock.init(allocator, embed_dim, vocab_size, 1);
    }
    
    var embed_layer = try transformer.BitLinear.init(allocator, vocab_size, embed_dim);
    defer embed_layer.deinit(allocator);

    // 3. Load Weights using POSIX
    const file_name_z: [*:0]const u8 = "smollm_1bit.bin";
    const fd = std.posix.openatZ(std.posix.AT.FDCWD, file_name_z, .{ .ACCMODE = .RDONLY }, 0) catch |err| {
        std.debug.print("Failed to open smollm_1bit.bin. Have you trained the model yet? Error: {}\n", .{err});
        return;
    };
    
    const FdReader = struct {
        fd: std.posix.fd_t,
        pub const Error = std.posix.ReadError;
        pub fn readAll(self: @This(), buffer: []u8) Error!usize {
            var total_read: usize = 0;
            while (total_read < buffer.len) {
                const n = try std.posix.read(self.fd, buffer[total_read..]);
                if (n == 0) break;
                total_read += n;
            }
            return total_read;
        }
    };
    const reader = FdReader{ .fd = fd };
    
    try embed_layer.load(reader);
    for (0..num_layers) |l| {
        try layers[l].load(reader);
    }
    std.debug.print("Model weights loaded successfully from {s}!\n", .{file_name_z});

    // 4. Interactive Chat Loop
    var input_buf: [1024]u8 = undefined;

    const token_i32 = try allocator.alloc(i32, embed_dim);
    defer allocator.free(token_i32);
    
    const vocab_packed_dim = vocab_size / 64;
    const x_packed = try allocator.alloc(u64, vocab_packed_dim);
    defer allocator.free(x_packed);
    
    var x_i8 = try allocator.alloc(i8, vocab_size);
    defer allocator.free(x_i8);

    while (true) {
        std.debug.print("\nUser> ", .{});
        var len: usize = 0;
        while (len < input_buf.len) {
            var char: [1]u8 = undefined;
            const n = std.posix.read(std.posix.STDIN_FILENO, &char) catch 0;
            if (n == 0) break;
            if (char[0] == '\n') break;
            input_buf[len] = char[0];
            len += 1;
        }
        if (len == 0) break;
        const prompt = input_buf[0..len];
        if (std.mem.eql(u8, prompt, "exit")) break;
        
        // std.debug.print("CPU-GPT> ", .{});
        
        // Reset SSM Cache
        for (0..num_layers) |l| layers[l].resetCache();

        // Encode prompt (Fallback to naive byte encoding if BPE missing)
        var last_token: usize = ' ';
        if (prompt.len > 0) {
            last_token = prompt[prompt.len - 1]; // Extremely naive prompt priming
        }

        // Reset KV Cache / SSM State before generating
        for (0..num_layers) |l| {
            layers[l].resetCache();
        }

        // Extremely simple stochastic sampling proportional to pseudo-logits
        // We use the RNG to add noise to the argmax to prevent looping
        var seed: u32 = @as(u32, @intCast(prompt.len + 1234));
        
        // Generate 50 tokens
        for (0..50) |step| {
            _ = step;
            
            // One-hot encode the token
            @memset(x_i8, -1);
            if (last_token < vocab_size) {
                x_i8[last_token] = 1;
            } else {
                x_i8[0] = 1;
            }
            
            binary_kernel.packBinary(x_i8, x_packed);
            
            // Embed
            embed_layer.forwardThreaded(x_packed, token_i32, &pool) catch unreachable;
            
            // Deep Forward
            for (0..num_layers) |l| {
                layers[l].inferenceForward(allocator, token_i32, &pool) catch unreachable;
            }
            
            // Predict
            const last_token_i8 = try allocator.alloc(i8, embed_dim);
            defer allocator.free(last_token_i8);
            block.BitMaxNorm.forward(token_i32, last_token_i8);
            
            const packed_dim = embed_dim / 64;
            const last_token_packed = try allocator.alloc(u64, packed_dim);
            defer allocator.free(last_token_packed);
            binary_kernel.packBinary(last_token_i8, last_token_packed);
            
            const local_out = try allocator.alloc(i32, vocab_size);
            defer allocator.free(local_out);
            layers[num_layers - 1].local_head.forwardThreaded(last_token_packed, local_out, &pool) catch unreachable;
            
            // Stochastic Argmax (Simple Gumbel-like noise)
            var max_val: i32 = -999999;
            var best_tok: usize = 0;
            
            // Xorshift32 PRNG
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            
            for (0..vocab_size) |v| {
                // Add pseudo-random noise [-31, 31] to break deterministic loops
                const noise = @as(i32, @intCast(seed % 63)) - 31;
                const noisy_val = local_out[v] + noise;
                
                if (noisy_val > max_val) {
                    max_val = noisy_val;
                    best_tok = v;
                }
            }
            
            last_token = best_tok;
            
            // We use ASCII bounds for now until full BPE decode is wired
            const char_out = if (best_tok >= 32 and best_tok <= 126) @as(u8, @intCast(best_tok)) else ' ';
            const char_arr = [_]u8{char_out};
            _ = std.c.write(1, &char_arr, 1);
        }
        _ = std.c.write(1, "\x04", 1);
    }
}
