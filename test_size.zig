const std = @import("std");

const BlockQ4_0 = extern struct {
    d: f32,
    qs: [16]u8,
};

pub fn main() void {
    std.debug.print("Size: {}\n", .{@sizeOf(BlockQ4_0)});
}
