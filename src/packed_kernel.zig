const std = @import("std");

/// Computes the dot product of a packed 2-bit ternary array and an INT8 array.
/// `packed_a` contains 4 weights per byte.
/// Bits 0-1: weight 0
/// Bits 2-3: weight 1
/// Bits 4-5: weight 2
/// Bits 6-7: weight 3
/// Value mapping: 00 -> 0, 01 -> 1, 10 -> -1
const UNPACK_LUT = blk: {
    @setEvalBranchQuota(2000);
    var lut: [256][4]i8 = undefined;
    for (0..256) |i| {
        for (0..4) |j| {
            const shift = @as(u3, @intCast(j * 2));
            const w = (i >> shift) & 0x03;
            lut[i][j] = @as(i8, @intCast(w & 1)) - @as(i8, @intCast(w >> 1));
        }
    }
    break :blk lut;
};

pub fn dotProductPackedInt8(packed_a: []const u8, b: []const i8) i32 {
    std.debug.assert(packed_a.len * 4 == b.len);
    var sum: i32 = 0;
    
    // Process 4 bytes of packed weights (16 weights) at a time
    const VLen = 16;
    const V8 = @Vector(VLen, i8);
    const V16 = @Vector(VLen, i16);
    const V32 = @Vector(VLen, i32);
    
    var i: usize = 0;
    var p_i: usize = 0;
    
    while (p_i + 4 <= packed_a.len) : ({ p_i += 4; i += 16; }) {
        // Read 4 bytes = 32 bits = 16 weights
        const p_bytes = packed_a[p_i..][0..4];
        
        // We want to create a V8 of the unpacked weights.
        // There are faster ways with NEON specific intrinsics, but we will rely 
        // on Zig's vector optimizer for this generic representation.
        // Process 4 bytes (16 weights) purely in SIMD vectors
        const p_vec = @Vector(16, u8){
            p_bytes[0], p_bytes[0], p_bytes[0], p_bytes[0],
            p_bytes[1], p_bytes[1], p_bytes[1], p_bytes[1],
            p_bytes[2], p_bytes[2], p_bytes[2], p_bytes[2],
            p_bytes[3], p_bytes[3], p_bytes[3], p_bytes[3],
        };
        
        const shift_vec = @Vector(16, u3){
            0, 2, 4, 6,
            0, 2, 4, 6,
            0, 2, 4, 6,
            0, 2, 4, 6,
        };
        
        const w_shifted = p_vec >> shift_vec;
        const w_masked = w_shifted & @as(@Vector(16, u8), @splat(3));
        const w_i8 = @as(@Vector(16, i8), @bitCast(w_masked));
        
        const w_mapped = (w_i8 & @as(@Vector(16, i8), @splat(1))) - (w_i8 >> @as(@Vector(16, u3), @splat(1)));
        
        const va: V8 = w_mapped;
        const vb: V8 = b[i..][0..VLen].*;
        
        const va16: V16 = va;
        const vb16: V16 = vb;
        
        const v_prod = va16 * vb16;
        const v_prod32: V32 = v_prod;
        
        sum += @reduce(.Add, v_prod32);
    }
    
    // Handle remaining bytes (up to 3)
    while (p_i < packed_a.len) : ({ p_i += 1; i += 4; }) {
        const B = packed_a[p_i];
        for (0..4) |w_idx| {
            if (i + w_idx >= b.len) break;
            const shift = @as(u3, @intCast(w_idx * 2));
            const W = (B >> shift) & 0x03;
            const w_mapped = @as(i32, @intCast(W & 1)) - @as(i32, @intCast(W >> 1));
            sum += w_mapped * @as(i32, b[i + w_idx]);
        }
    }
    
    return sum;
}

test "dotProductPackedInt8 simple" {
    // Let's pack 16 weights
    // weights: 1, -1, 0, 1 | 1, 1, -1, -1 | 0, 0, 0, 0 | 1, 1, 1, 1
    // mappings: 1->01, -1->10, 0->00
    // Byte 0: 01 (1), 10 (-1), 00 (0), 01 (1)
    // 01_00_10_01 in binary = 0x49
    
    // Byte 1: 01 (1), 01 (1), 10 (-1), 10 (-1)
    // 10_10_01_01 in binary = 0xA5
    
    // Byte 2: 00, 00, 00, 00 = 0x00
    
    // Byte 3: 01, 01, 01, 01 = 0x55
    
    const packed_w = [_]u8{ 0x49, 0xA5, 0x00, 0x55 };
    const b = [_]i8{ 
        1, 1, 1, 1, // weights 1, -1, 0, 1 -> sum = 1
        1, 1, 1, 1, // weights 1, 1, -1, -1 -> sum = 0
        1, 1, 1, 1, // weights 0, 0, 0, 0 -> sum = 0
        1, 1, 1, 1  // weights 1, 1, 1, 1 -> sum = 4
    }; // total expected sum = 5
    
    const result = dotProductPackedInt8(&packed_w, &b);
    try std.testing.expectEqual(@as(i32, 5), result);
}
