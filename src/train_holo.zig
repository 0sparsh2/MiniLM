const std = @import("std");
const holo_reservoir = @import("holo_reservoir.zig");
const dataloader = @import("dataloader.zig");
const stochastic = @import("stochastic.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const vocab_size = 256;
    // 8192-dimensional vector. 
    // This allows for extreme hash collision avoidance in holographic memory.
    const embed_dim = 8192; 
    
    // Create the architecture
    var model = try holo_reservoir.HoloReservoir.init(allocator, embed_dim, vocab_size);
    defer model.deinit(allocator);

    var rng = stochastic.VectorXorshift32.init(42);
    model.randomize(&rng);
    
    const train_file = @embedFile("wiki_5mb.txt");
    var loader = try dataloader.DataLoader.init(allocator, train_file, 64, 32);
    defer loader.deinit();

    std.debug.print("--- HoloReservoir 8192-Dim Initialized! ---\n", .{});
    std.debug.print("Parameters: Exactly 1 (The Readout Matrix). Matrix Size: 262 KB.\n", .{});

    const batch_size = 32;
    const seq_len = 64;
    
    const b_x = try allocator.alloc(usize, batch_size * seq_len);
    defer allocator.free(b_x);
    const b_y = try allocator.alloc(usize, batch_size);
    defer allocator.free(b_y);

    const num_steps = 15000;
    var shift_scale: u4 = 3;

    for (0..num_steps) |step| {
        if (step == 5_000) shift_scale = 4;
        if (step == 10_000) shift_scale = 5;
        if (step == 13_000) shift_scale = 6;

        loader.nextTrainBatch(b_x, b_y);
        
        var batch_loss: i64 = 0;
        
        // Train sequentially for memory safety and zero OS thread overhead
        for (0..batch_size) |b| {
            // Allocate a fast tiny buffer for this specific batch sequence
            var fba_buf: [1024 * 1024]u8 = undefined;
            var fba = std.heap.FixedBufferAllocator.init(&fba_buf);
            const temp_alloc = fba.allocator();
            
            const sequence_usize = b_x[b * seq_len .. (b + 1) * seq_len];
            const target_id = b_y[b];
            
            const sequence = try temp_alloc.alloc(u8, seq_len);
            for (0..seq_len) |t| sequence[t] = @as(u8, @intCast(sequence_usize[t]));
            
            var pred: usize = 0;
            var loss: i64 = 0;
            
            try model.forwardTraining(temp_alloc, sequence, target_id, &pred, &loss, shift_scale, &rng);
            batch_loss += loss;
        }
        
        if (step % 100 == 0) {
            const avg_loss = @divTrunc(batch_loss, batch_size);
            std.debug.print("Step {d:0>6} | LR (Shift): {d} | Train Loss: {d}\n", .{ step, shift_scale, avg_loss });
        }
    }

    std.debug.print("Training completed\n", .{});

    const out_file = try std.fs.cwd().createFile("holo_1bit.bin", .{});
    defer out_file.close();
    try model.save(out_file.writer());
    std.debug.print("Model weights successfully saved to holo_1bit.bin!\n", .{});
}
