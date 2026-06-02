const std = @import("std");
const packed_kernel = @import("packed_kernel.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const vector_len: usize = 1024 * 1024; // 1M elements
    const iterations: usize = 1000;
    
    var data_a = try allocator.alloc(i8, vector_len);
    var data_b = try allocator.alloc(i8, vector_len);
    var packed_a = try allocator.alloc(u8, vector_len / 4);
    
    for (0..vector_len) |i| {
        data_a[i] = @as(i8, @intCast((i % 3) - 1));
        data_b[i] = @as(i8, @intCast((i % 100) - 50));
    }
    
    for (0..packed_a.len) |i| {
        var p: u8 = 0;
        for (0..4) |j| {
            const w = data_a[i * 4 + j];
            var bits: u8 = 0;
            if (w == 1) bits = 1;
            if (w == -1) bits = 2;
            p |= (bits << @as(u3, @intCast(j * 2)));
        }
        packed_a[i] = p;
    }
    
    var sum: i32 = 0;
    for (0..iterations) |_| {
        sum +%= packed_kernel.dotProductPackedInt8(packed_a, data_b);
    }
    std.debug.print("Sum: {}\n", .{sum});
}
