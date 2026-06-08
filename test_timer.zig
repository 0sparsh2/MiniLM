const std = @import("std");
pub fn main() !void {
    std.debug.print("Has std.Timer: {}\n", .{@hasDecl(std, "Timer")});
    std.debug.print("Has std.time.timestamp: {}\n", .{@hasDecl(std.time, "timestamp")});
}
