const std = @import("std");
pub fn main() !void {
    std.debug.print("Has Pool: {}\n", .{@hasDecl(std.Thread, "Pool")});
}
