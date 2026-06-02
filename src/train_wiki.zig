const std = @import("std");
const dataloader = @import("dataloader.zig");
const block = @import("block.zig");
const transformer = @import("transformer.zig");
const stochastic = @import("stochastic.zig");
const thread_pool = @import("thread_pool.zig");
const binary_kernel = @import("binary_kernel.zig");

pub fn main() !void {
    // We use an ArenaAllocator to guarantee that all neural network layers
    // (the entire ~12MB model) are allocated contiguously in physical memory.
    // This dramatically improves the likelihood of L2/L3 cache residency
    // and eliminates memory fragmentation, allowing the CPU to prefetch perfectly.
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    std.debug.print("--- Loading Wikipedia (5MB) ---\n", .{});
    const text = @embedFile("wiki_5mb.txt");

    const batch_size = 8;
    const seq_len = 64;
    var loader = try dataloader.DataLoader.init(allocator, text, seq_len, batch_size);
    defer loader.deinit();
    
    const vocab_size = dataloader.DataLoader.vocab_size;
    std.debug.print("BPE Loaded! Vocab Size: {d} | Train Seq Len: {d} | Val Seq Len: {d}\n", .{vocab_size, loader.train_slice.len, loader.val_slice.len});

    const pool = thread_pool.ThreadPool.init(allocator, 10);
    std.debug.print("--- Initialized 10-Core Thread Pool ---\n", .{});

    const embed_dim = 128;
    const num_layers = 4;
    
    // Initial Hyperparameter (Learning Rate)
    // shift_scale will increase (lowering the LR) as training progresses.
    var shift_scale: u4 = 3;
    
    std.debug.print("\n=== STARTING TRAINING ===\n", .{});
    
    var layers = try allocator.alloc(block.TransformerBlock, num_layers);
    defer allocator.free(layers);
        
        var rng = stochastic.VectorXorshift32.init(9999);
        
        for (0..num_layers) |l| {
            layers[l] = try block.TransformerBlock.init(allocator, embed_dim, vocab_size, seq_len);
            layers[l].randomize(&rng);
        }
        
        var embed_layer = try transformer.BitLinear.init(allocator, vocab_size, embed_dim);
        defer embed_layer.deinit(allocator);
        for (0..embed_layer.shadow_weights.len) |i| embed_layer.shadow_weights[i] = @as(i16, @intCast(rng.next()[0] % 50)) - 25;
        embed_layer.syncWeights();

        var x_i32 = try allocator.alloc(i32, seq_len * embed_dim);
        defer allocator.free(x_i32);
        
        var x_i8 = try allocator.alloc(i8, seq_len * vocab_size);
        defer allocator.free(x_i8);

        const b_x = try allocator.alloc(usize, batch_size * seq_len);
        defer allocator.free(b_x);
        const b_y = try allocator.alloc(usize, batch_size);
        defer allocator.free(b_y);

        const num_steps = 15000; // Train for 15,000 steps (takes ~3 minutes)
        
        for (0..num_steps) |step| {
            // Learning Rate Decay (Increase shift_scale to decrease gradient step)
            if (step == 5_000) shift_scale = 4;
            if (step == 10_000) shift_scale = 5;
            if (step == 13_000) shift_scale = 6;
            loader.nextTrainBatch(b_x, b_y);
            
            var batch_loss: i64 = 0;

            for (0..batch_size) |b| {
                // Pre-allocate 4MB fixed buffer for temporary per-batch allocations
                // This eliminates heap allocation overhead and prevents Arena leaks!
                var fba_buf: [4 * 1024 * 1024]u8 = undefined;
                var fba = std.heap.FixedBufferAllocator.init(&fba_buf);
                const temp_alloc = fba.allocator();
                
                @memset(x_i8, -1);
                for (0..seq_len) |t| {
                    const token_id = b_x[b * seq_len + t];
                    if (token_id < vocab_size) {
                        x_i8[t * vocab_size + token_id] = 1;
                    }
                }
                
                const target_id = b_y[b];
                
                const vocab_packed_dim = vocab_size / 64;
                var x_packed = try temp_alloc.alloc(u64, seq_len * vocab_packed_dim);
                defer temp_alloc.free(x_packed);
                binary_kernel.packBinary(x_i8, x_packed);
                
                // Embed Token
                for (0..seq_len) |t| {
                    const token_packed = x_packed[t * vocab_packed_dim .. (t + 1) * vocab_packed_dim];
                    const token_out = x_i32[t * embed_dim .. (t + 1) * embed_dim];
                    try embed_layer.forwardThreaded(token_packed, token_out, &pool);
                }
                
                var final_loss: i64 = 0;
                var final_pred: usize = 0;
                
                for (0..num_layers) |l| {
                    var layer_loss: i64 = 0;
                    var layer_pred: usize = 0;
                    try layers[l].forward(
                        temp_alloc, 
                        x_i32, 
                        &pool, 
                        @as(u8, @intCast(target_id)), // Assuming vocab < 256
                        vocab_size, 
                        &rng, 
                        &layer_pred, 
                        &layer_loss,
                        shift_scale
                    );
                    
                    if (l == num_layers - 1) {
                        final_loss = layer_loss;
                        final_pred = layer_pred;
                    }
                }
                
                batch_loss += final_loss;
            }
            
            if (step > 0 and step % 100 == 0) {
                const avg_loss = @divTrunc(batch_loss, @as(i64, @intCast(batch_size)));
                std.debug.print("Step {d:0>6} | LR (Shift): {d} | Train Loss: {d:0>6}\n", .{step, shift_scale, avg_loss});
            }
        }
        
        std.debug.print("Training completed\n", .{});
        
        // Save weights
        const file_name_z: [*:0]const u8 = "smollm_1bit.bin";
        const fd = std.posix.openatZ(std.posix.AT.FDCWD, file_name_z, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644) catch |err| {
            std.debug.print("Failed to open file for saving. Error: {}\n", .{err});
            return;
        };
        const FdWriter = struct {
            fd: std.posix.fd_t,
            pub fn writeAll(self: @This(), bytes: []const u8) !void {
                var total_written: usize = 0;
                while (total_written < bytes.len) {
                    const n = std.c.write(self.fd, bytes[total_written..].ptr, bytes.len - total_written);
                    if (n <= 0) return error.WriteFailed;
                    total_written += @as(usize, @intCast(n));
                }
            }
        };
        const writer = FdWriter{ .fd = fd };
        try embed_layer.save(writer);
        for (0..num_layers) |l| {
            try layers[l].save(writer);
        }
        _ = std.c.close(fd);
        std.debug.print("Model weights successfully saved to {s}!\n", .{file_name_z});
        
        for (0..num_layers) |l| layers[l].deinit(allocator);
    
    std.debug.print("--- Process Complete ---\n", .{});
}
