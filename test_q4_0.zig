const std = @import("std");

const BlockQ4_0 = extern struct {
    d: f32,
    qs: [16]u8,
};

pub fn main() !void {
    const fd = try std.posix.openatZ(std.posix.AT.FDCWD, "test.bin", .{ .ACCMODE = .RDONLY }, 0);
    var block: BlockQ4_0 = undefined;
    
    var bytes_read: usize = 0;
    while (bytes_read < @sizeOf(BlockQ4_0)) {
        const n = try std.posix.read(fd, std.mem.asBytes(&block)[bytes_read..]);
        if (n == 0) break;
        bytes_read += n;
    }
    
    std.debug.print("Scale d: {d}\n", .{block.d});
    for (0..16) |c| {
        const packed_byte = block.qs[c];
        const even = (packed_byte >> 4) & 0x0F;
        const odd = packed_byte & 0x0F;
        
        const w_even = (@as(f32, @floatFromInt(even)) - 8.0) * block.d;
        const w_odd = (@as(f32, @floatFromInt(odd)) - 8.0) * block.d;
        
        std.debug.print("[{d}] = {d}\n", .{c*2, w_even});
        std.debug.print("[{d}] = {d}\n", .{c*2+1, w_odd});
    }
}
