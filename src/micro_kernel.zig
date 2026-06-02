const std = @import("std");

/// Computes the dot product of two INT8 arrays, accumulating into an INT32.
/// Uses Zig's vectorization to leverage SIMD (NEON on ARM64).
pub fn dotProductInt8(a: []const i8, b: []const i8) i32 {
    std.debug.assert(a.len == b.len);
    var sum: i32 = 0;
    
    const VLen = 16; // 128-bit vector holds 16 x i8
    const V8 = @Vector(VLen, i8);
    const V16 = @Vector(VLen, i16);
    const V32 = @Vector(VLen, i32);
    
    var i: usize = 0;
    while (i + VLen <= a.len) : (i += VLen) {
        const va: V8 = a[i..][0..VLen].*;
        const vb: V8 = b[i..][0..VLen].*;
        
        // In Zig 0.11+, extending vectors can be done by assigning or casting.
        // Let's try @as to cast the elements up to i16 to avoid overflow during multiply.
        // Max value of i8 * i8 is 16384, fitting in i16.
        const va16: V16 = va;
        const vb16: V16 = vb;
        
        const v_prod = va16 * vb16;
        
        // Cast to i32 before accumulating to avoid overflow in sum
        const v_prod32: V32 = v_prod;
        
        sum += @reduce(.Add, v_prod32);
    }
    
    // Handle remaining elements
    while (i < a.len) : (i += 1) {
        sum += @as(i32, a[i]) * @as(i32, b[i]);
    }
    
    return sum;
}

test "dotProductInt8 simple" {
    const a = [_]i8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1 };
    const b = [_]i8{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 2 };
    
    const expected: i32 = 136 + 2; // sum(1..16) + 2 = 138
    const result = dotProductInt8(&a, &b);
    try std.testing.expectEqual(expected, result);
}
