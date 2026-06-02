const std = @import("std");

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    char_to_id: std.AutoHashMap(u8, u8),
    id_to_char: []u8,
    vocab_size: usize,

    pub fn init(allocator: std.mem.Allocator, text: []const u8) !Tokenizer {
        var char_to_id = std.AutoHashMap(u8, u8).init(allocator);
        
        // Find unique characters
        for (text) |c| {
            if (!char_to_id.contains(c)) {
                try char_to_id.put(c, 0); // Temporary value
            }
        }
        
        const vocab_size = char_to_id.count();
        var id_to_char = try allocator.alloc(u8, vocab_size);
        
        // Sort characters to make assignment deterministic
        var keys_arr = try allocator.alloc(u8, vocab_size);
        defer allocator.free(keys_arr);
        
        var it = char_to_id.keyIterator();
        var idx: usize = 0;
        while (it.next()) |k_ptr| {
            keys_arr[idx] = k_ptr.*;
            idx += 1;
        }
        
        std.mem.sort(u8, keys_arr, {}, std.sort.asc(u8));
        
        // Assign IDs
        for (keys_arr, 0..) |c, i| {
            try char_to_id.put(c, @as(u8, @intCast(i)));
            id_to_char[i] = c;
        }
        
        return .{
            .allocator = allocator,
            .char_to_id = char_to_id,
            .id_to_char = id_to_char,
            .vocab_size = vocab_size,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.char_to_id.deinit();
        self.allocator.free(self.id_to_char);
    }

    pub fn encode(self: *Tokenizer, text: []const u8, out: []u8) void {
        std.debug.assert(text.len == out.len);
        for (text, 0..) |c, i| {
            out[i] = self.char_to_id.get(c) orelse 0; // Default to 0 if unknown
        }
    }

    pub fn decode(self: *Tokenizer, ids: []const u8, out: []u8) void {
        std.debug.assert(ids.len == out.len);
        for (ids, 0..) |id, i| {
            if (id < self.vocab_size) {
                out[i] = self.id_to_char[id];
            } else {
                out[i] = '?';
            }
        }
    }
};

test "Tokenizer basic" {
    const alloc = std.testing.allocator;
    const text = "hello world";
    var tokenizer = try Tokenizer.init(alloc, text);
    defer tokenizer.deinit();
    
    // Unique chars: ' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w'
    var encoded: [11]u8 = undefined;
    tokenizer.encode(text, &encoded);
    
    var decoded: [11]u8 = undefined;
    tokenizer.decode(&encoded, &decoded);
    
    try std.testing.expectEqualStrings("hello world", &decoded);
}
