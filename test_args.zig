const std = @import("std");

pub fn main(init: std.process.Init) !void {
    var args = init.minimal.args;
    var iter = args.iterate();
    while (iter.next()) |arg| {
        std.debug.print("{s}\n", .{arg});
    }
}
