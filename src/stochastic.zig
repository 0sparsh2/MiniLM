const std = @import("std");

/// A very simple Xorshift PRNG that operates on SIMD vectors for speed.
pub const VectorXorshift32 = struct {
    const VLen = 16;
    const V32 = @Vector(VLen, u32);

    state: V32,

    pub fn init(seed: u32) VectorXorshift32 {
        var initial_state: [VLen]u32 = undefined;
        var current_seed = seed;
        for (0..VLen) |i| {
            // Simple linear congruential generator to seed the xorshift
            current_seed = current_seed *% 1664525 +% 1013904223;
            initial_state[i] = current_seed;
        }
        return .{ .state = initial_state };
    }

    pub fn next(self: *VectorXorshift32) V32 {
        var x = self.state;
        const shift13: @Vector(VLen, u5) = @splat(13);
        const shift17: @Vector(VLen, u5) = @splat(17);
        const shift5: @Vector(VLen, u5) = @splat(5);
        
        x ^= x << shift13;
        x ^= x >> shift17;
        x ^= x << shift5;
        self.state = x;
        return x;
    }
};

/// Stochastically updates an array of INT16 shadow weights using INT8 gradients.
pub fn stochasticUpdate(shadow_weights: []i16, gradients: []const i8, shift: u4, rng: *VectorXorshift32) void {
    std.debug.assert(shadow_weights.len == gradients.len);
    
    const VLen = 16;
    const V16 = @Vector(VLen, i16);
    const V8 = @Vector(VLen, i8);
    
    var i: usize = 0;
    while (i + VLen <= shadow_weights.len) : (i += VLen) {
        const sw: V16 = shadow_weights[i..][0..VLen].*;
        const grad: V8 = gradients[i..][0..VLen].*;
        
        // Convert gradients to i16
        const grad16: V16 = grad;
        
        // Generate random numbers
        const rand_u32 = rng.next();
        
        // We need random numbers between 0 and (1 << shift) - 1.
        // We can get this by masking.
        const mask = (@as(u32, 1) << shift) - 1;
        const mask_vec: @Vector(VLen, u32) = @splat(mask);
        const rand_masked = rand_u32 & mask_vec;
        
        // We want to do: (grad + rand) >> shift, but sign-aware.
        // For negative gradients, we should subtract the random value to round stochastically towards zero.
        // Actually, stochastic rounding of X / S:
        // floor(X/S) + ( (X % S) > rand ? 1 : 0 )
        // A simpler trick for integers:
        // if X > 0: (X + rand) >> shift
        // if X < 0: (X - rand) >> shift (or similar)
        // Let's implement the exact probability:
        
        const rand16: V16 = @intCast(rand_masked);
        
        const v_zero: V16 = @splat(0);
        const shift_vec: @Vector(VLen, u4) = @splat(shift);
        
        // g > 0
        const mask_gt = grad16 > v_zero;
        const update_gt = (grad16 + rand16) >> shift_vec;
        
        // g < 0
        const mask_lt = grad16 < v_zero;
        const neg_grad = v_zero - grad16; // -grad16 can overflow, but gradients are i8 cast to i16, so -grad is safe
        const update_lt = v_zero - ((neg_grad + rand16) >> shift_vec);
        
        var update = @select(i16, mask_gt, update_gt, v_zero);
        update = @select(i16, mask_lt, update_lt, update);
        
        // Write back
        shadow_weights[i..][0..VLen].* = sw +% update;
    }
    
    // Fallback for remaining elements omitted for brevity in this PoC
    while (i < shadow_weights.len) : (i += 1) {
        const g = @as(i16, gradients[i]);
        // Just a simple deterministic update for the remainder for now
        shadow_weights[i] +%= g >> shift;
    }
}

test "stochastic update" {
    var rng = VectorXorshift32.init(42);
    
    var shadow_weights = [_]i16{ 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100 };
    const gradients = [_]i8{ 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10 };
    
    // Shift by 3 means we divide gradients by 8.
    // Gradient is 10. 10 / 8 = 1.25.
    // So 25% of the time it updates by 2, 75% of the time it updates by 1.
    stochasticUpdate(&shadow_weights, &gradients, 3, &rng);
    
    // The sum of updates should be roughly 1.25 * 16 = 20.
    // Total sum should be ~ 1600 + 20 = 1620.
    var sum: i32 = 0;
    for (shadow_weights) |w| {
        sum += w;
    }
    
    // Should be strictly greater than 1616 (if it always rounded down to 1) 
    // and less than 1632 (if it always rounded up to 2).
    try std.testing.expect(sum > 1616 and sum < 1632);
}
