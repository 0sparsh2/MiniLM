const std = @import("std");
const transformer = @import("transformer.zig");
const stochastic = @import("stochastic.zig");

// A script that proves we can update a single isolated layer locally based on 
// localized pseudo-targets without keeping gradients for previous layers in memory.
pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const vocab_size: usize = 16;
    
    // We instantiate a network layer
    var layer1 = try transformer.BitLinear.init(allocator, vocab_size, vocab_size);
    defer layer1.deinit(allocator);

    // Initialize shadow weights negatively so it outputs roughly negative values
    for (0..layer1.shadow_weights.len) |i| {
        layer1.shadow_weights[i] = -50;
    }
    layer1.syncWeights();

    var rng = stochastic.VectorXorshift32.init(55555);

    // 1. Setup a tiny toy dataset
    // We want the layer to map input[1] -> output[2] = 1
    var input = [_]i8{0} ** vocab_size;
    input[1] = 1;
    
    const output = try allocator.alloc(i32, vocab_size);
    defer allocator.free(output);
    
    const local_error = try allocator.alloc(i32, vocab_size);
    defer allocator.free(local_error);

    std.debug.print("--- Starting Phase 4 Localized Training ---\n", .{});
    
    const num_epochs = 1000;
    for (0..num_epochs) |epoch| {
        // --- FORWARD PASS (Local) ---
        layer1.forward(&input, output);
        
        // --- COMPUTE LOCAL ERROR ---
        // For a local target, we pretend we know exactly what THIS layer should output
        // In real Forward-Forward, this comes from a localized 'goodness' function.
        var loss: i64 = 0;
        @memset(local_error, 0);
        
        const target_val: i32 = 1;
        const current_val = output[2];
        const diff = current_val - target_val; // For index 2
        
        loss += @as(i64, diff) * @as(i64, diff);
        local_error[2] = diff;
        
        // --- LOCAL UPDATE ---
        // Apply the gradient locally immediately without touching a backprop graph!
        try layer1.localUpdate(&input, local_error, 1, &rng, allocator);
        
        if (epoch % 100 == 0 or epoch == num_epochs - 1) {
            std.debug.print("Epoch {d:0>3} | Local Loss: {d:0>6}\n", .{epoch, loss});
        }
    }
    
    std.debug.print("------------------------------------------\n", .{});
    std.debug.print("Final Output[2] without backprop: {d}\n", .{output[2]});
}
