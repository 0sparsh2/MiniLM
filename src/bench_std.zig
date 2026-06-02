const std = @import("std");
const micro_kernel = @import("micro_kernel.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const vector_len: usize = 1024 * 1024; // 1M elements
    const iterations: usize = 1000;
    
    var data_a = try allocator.alloc(i8, vector_len);
    var data_b = try allocator.alloc(i8, vector_len);
    
    for (0..vector_len) |i| {
        data_a[i] = @as(i8, @intCast((i % 3) - 1));
        data_b[i] = @as(i8, @intCast((i % 100) - 50));
    }
    
    var sum: i32 = 0;
    for (0..iterations) |_| {
        sum +%= micro_kernel.dotProductInt8(data_a, data_b);
    }
    std.debug.print("Sum: {}\n", .{sum});
}
