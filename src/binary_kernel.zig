const std = @import("std");

/// Computes the dot product of two 1-bit packed binary arrays.
/// Both `a` and `b` contain packed weights where each bit represents a parameter:
/// bit 0 -> -1
/// bit 1 -> +1
/// The dot product of two binary vectors is defined as:
/// number_of_matching_bits - number_of_mismatching_bits.
/// Matching bits are found via `a == b`, mismatching via `a != b`.
/// Mathematically: length_in_bits - 2 * popcount(a XOR b).
pub fn dotProductBinary(a: []const u64, b: []const u64) i32 {
    std.debug.assert(a.len == b.len);
    var pop_count_diff: usize = 0;
    
    const VLen = 16; // 1024-bit vector unroll
    const V = @Vector(VLen, u64);
    
    var i: usize = 0;
    while (i + VLen <= a.len) : (i += VLen) {
        const v_a: V = a[i..][0..VLen].*;
        const v_b: V = b[i..][0..VLen].*;
        const xor = v_a ^ v_b;
        const popcounts = @popCount(xor);
        pop_count_diff += @reduce(.Add, popcounts);
    }
    
    while (i < a.len) : (i += 1) {
        pop_count_diff += @popCount(a[i] ^ b[i]);
    }
    
    const total_bits = a.len * 64;
    return @as(i32, @intCast(total_bits)) - 2 * @as(i32, @intCast(pop_count_diff));
}

/// Packs an array of `i8` ternary/binary values {-1, 0, 1} into an array of `u64`.
/// For 1-bit networks, we force 0 to +1 or -1 randomly, or just treat >= 0 as 1 and < 0 as -1.
/// In this kernel, >= 0 is bit=1, < 0 is bit=0.
pub fn packBinary(src: []const i8, dst: []u64) void {
    std.debug.assert(src.len <= dst.len * 64);
    @memset(dst, 0);
    
    for (src, 0..) |val, i| {
        if (val >= 0) {
            const block_idx = i / 64;
            const bit_idx = @as(u6, @intCast(i % 64));
            dst[block_idx] |= (@as(u64, 1) << bit_idx);
        }
    }
}

/// Unpacks an array of `u64` back into an array of `i8` {-1, 1}
pub fn unpackBinary(src: []const u64, dst: []i8) void {
    std.debug.assert(dst.len <= src.len * 64);
    
    for (0..dst.len) |i| {
        const block_idx = i / 64;
        const bit_idx = @as(u6, @intCast(i % 64));
        const bit = (src[block_idx] >> bit_idx) & 1;
        dst[i] = if (bit == 1) 1 else -1;
    }
}

test "binary dot product" {
    // 64 bits of 1s and 64 bits of alternating
    const a = [_]u64{ 0xFFFFFFFFFFFFFFFF }; // All 1s
    const b = [_]u64{ 0x0000000000000000 }; // All 0s
    
    // Total bits = 64. popCount(a ^ b) = 64. Result = 64 - 128 = -64
    try std.testing.expectEqual(@as(i32, -64), dotProductBinary(&a, &b));
    
    const c = [_]u64{ 0xFFFFFFFFFFFFFFFF }; // All 1s
    const d = [_]u64{ 0xFFFFFFFFFFFFFFFF }; // All 1s
    
    // Total bits = 64. popCount(c ^ d) = 0. Result = 64 - 0 = 64
    try std.testing.expectEqual(@as(i32, 64), dotProductBinary(&c, &d));
}

test "pack and unpack" {
    const src = [_]i8{ -1, 1, -1, 1, 1, 1, -1, -1 }; // 8 bits
    var packed_vals: [1]u64 = undefined;
    
    packBinary(&src, &packed_vals);
    
    // Bits (LSB to MSB): 0, 1, 0, 1, 1, 1, 0, 0
    // Binary: 00111010 = 0x3A = 58
    try std.testing.expectEqual(@as(u64, 58), packed_vals[0]);
    
    var unpacked: [8]i8 = undefined;
    unpackBinary(&packed_vals, &unpacked);
    
    for (src, 0..) |v, i| {
        try std.testing.expectEqual(v, unpacked[i]);
    }
}
