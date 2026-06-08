const std = @import("std");
pub fn main() !void {
    const fd = std.c.open("test.bin", std.c.O.WRONLY | std.c.O.CREAT | std.c.O.TRUNC, 0o644);
    _ = std.c.write(fd, "hello", 5);
    _ = std.c.close(fd);
}
