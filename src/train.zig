const std = @import("std");
const transformer = @import("transformer.zig");
const stochastic = @import("stochastic.zig");

// A tiny training loop to prove convergence feasibility and measure baseline speed.
pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // 1. Setup a tiny toy dataset
    // We will just try to map an input vector (one-hot encoded 'h') to an output vector ('e').
    // In a real LLM, this is predicting the next token.
    const vocab_size: usize = 16;
    
    // Input token: 'h' (say, index 1)
    var input = [_]i8{0} ** vocab_size;
    input[1] = 1;
    
    // Target token: 'e' (say, index 2)
    var target = [_]i32{0} ** vocab_size;
    target[2] = 1; // Target must be 1 since BitLinear weights are bounded to [-1, 1]
    
    // 2. Initialize a single BitLinear layer (our "model")
    var model = try transformer.BitLinear.init(allocator, vocab_size, vocab_size);
    defer model.deinit(allocator);
    
    // Randomly initialize shadow weights so it has something to start with
    var rng = stochastic.VectorXorshift32.init(12345);
    for (0..model.shadow_weights.len) |i| {
        // Init negatively to force it to learn to become positive
        model.shadow_weights[i] = -50;
    }
    model.syncWeights();

    const output = try allocator.alloc(i32, vocab_size);
    defer allocator.free(output);
    
    var gradients = try allocator.alloc(i8, model.shadow_weights.len);
    defer allocator.free(gradients);

    std.debug.print("--- Starting Phase 1 Baseline Training ---\n", .{});
    
    const num_epochs = 1000;
    for (0..num_epochs) |epoch| {
        // --- FORWARD PASS ---
        model.forward(&input, output);
        
        // --- LOSS CALCULATION (MSE derivative) ---
        var loss: i64 = 0;
        var error_signals = try allocator.alloc(i32, vocab_size);
        defer allocator.free(error_signals);
        
        for (0..vocab_size) |i| {
            const diff = output[i] - target[i];
            loss += @as(i64, diff) * @as(i64, diff);
            error_signals[i] = diff; // Gradient of MSE with respect to output
        }
        
        // --- BACKWARD PASS ---
        // Calculate gradients for weights: grad_W = error_signal * input
        // Since input is [1, 0, 0...], grad_W is just error_signal where input == 1.
        @memset(gradients, 0);
        for (0..vocab_size) |out_idx| {
            const row_start = out_idx * vocab_size;
            const err = error_signals[out_idx];
            
            // Scale down error to fit in i8 gradient
            // Simple clipping/scaling for the proof of concept
            var scaled_err = err;
            if (scaled_err > 127) scaled_err = 127;
            if (scaled_err < -128) scaled_err = -128;
            
            for (0..vocab_size) |in_idx| {
                if (input[in_idx] != 0) {
                    gradients[row_start + in_idx] = @as(i8, @intCast(scaled_err));
                }
            }
        }
        
        // --- WEIGHT UPDATE (Stochastic Rounding) ---
        // shift=3 means effective learning rate scaling
        // We negate gradients when updating to move *against* the error
        for (0..gradients.len) |i| gradients[i] = -gradients[i];
        
        stochastic.stochasticUpdate(model.shadow_weights, gradients, 1, &rng);
        model.syncWeights(); // Apply shadow weights to actual ternary weights
        
        if (epoch % 100 == 0 or epoch == num_epochs - 1) {
            std.debug.print("Epoch {d:0>3} | Loss: {d:0>6}\n", .{epoch, loss});
        }
    }
    
    std.debug.print("------------------------------------------\n", .{});
    std.debug.print("Final output for target index 2: {d} (Target: {d})\n", .{output[2], target[2]});
}
