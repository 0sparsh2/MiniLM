const std = @import("std");
const micro_kernel = @import("micro_kernel.zig");
const packed_kernel = @import("packed_kernel.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const vector_len: usize = 1024 * 1024; // 1M elements
    const iterations: usize = 1000;
    
    var data_a = try allocator.alloc(i8, vector_len);
    defer allocator.free(data_a);
    var data_b = try allocator.alloc(i8, vector_len);
    defer allocator.free(data_b);
    var packed_a = try allocator.alloc(u8, vector_len / 4);
    defer allocator.free(packed_a);

    // Initialize with dummy data
    for (0..vector_len) |i| {
        data_a[i] = @as(i8, @intCast((i % 3) - 1)); // -1, 0, 1
        data_b[i] = @as(i8, @intCast((i % 100) - 50));
    }
    
    // Pack data_a into packed_a
    for (0..packed_a.len) |i| {
        var p: u8 = 0;
        for (0..4) |j| {
            const w = data_a[i * 4 + j];
            // mapping back: 0->00, 1->01, -1->10
            var bits: u8 = 0;
            if (w == 1) bits = 1;
            if (w == -1) bits = 2;
            p |= (bits << @as(u3, @intCast(j * 2)));
        }
        packed_a[i] = p;
    }


    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    const mode = if (args.len > 1) args[1] else "standard";
    
    if (std.mem.eql(u8, mode, "standard")) {
        std.debug.print("--- Benchmarking Standard Kernel (INT8) ---\n", .{});
        const start1 = std.time.milliTimestamp();
        var sum1: i32 = 0;
        for (0..iterations) |_| {
            sum1 +%= micro_kernel.dotProductInt8(data_a, data_b);
        }
        const end1 = std.time.milliTimestamp();
        const ops1 = (@as(f64, @floatFromInt(vector_len * iterations)) / @as(f64, @floatFromInt(end1 - start1))) * 1000.0 / 1_000_000_000.0;
        std.debug.print("Standard INT8: {} ms, Ops: {d:.2} GigaOps/sec, Sum: {}\n", .{end1 - start1, ops1, sum1});
    } else {
        std.debug.print("--- Benchmarking Packed Kernel (2-bit) ---\n", .{});
        const start2 = std.time.milliTimestamp();
        var sum2: i32 = 0;
        for (0..iterations) |_| {
            sum2 +%= packed_kernel.dotProductPackedInt8(packed_a, data_b);
        }
        const end2 = std.time.milliTimestamp();
        const ops2 = (@as(f64, @floatFromInt(vector_len * iterations)) / @as(f64, @floatFromInt(end2 - start2))) * 1000.0 / 1_000_000_000.0;
        std.debug.print("Packed 2-bit: {} ms, Ops: {d:.2} GigaOps/sec, Sum: {}\n", .{end2 - start2, ops2, sum2});
    }
}
