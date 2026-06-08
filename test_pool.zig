const std = @import("std");

pub fn main() !void {
    var pool: std.Thread.Pool = undefined;
    try pool.init(.{ .allocator = std.heap.page_allocator, .n_jobs = 4 });
    defer pool.deinit();
    
    var wg: std.Thread.WaitGroup = undefined;
    wg.reset();
    
    std.debug.print("Pool works!\n", .{});
}
