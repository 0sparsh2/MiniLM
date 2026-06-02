const std = @import("std");

pub const BPETokenizer = struct {
    allocator: std.mem.Allocator,
    vocab_size: usize,
    merges: std.ArrayList(MergeRule),
    byte_to_id: [256]u16,
    id_to_byte: std.ArrayList(u8),
    
    // Merge rules: (id1, id2) -> new_id
    pub const MergeRule = struct {
        p1: u16,
        p2: u16,
        new_id: u16,
    };
    
    pub fn init(allocator: std.mem.Allocator) BPETokenizer {
        const byte_to_id = [_]u16{9999} ** 256;
        return .{
            .allocator = allocator,
            .vocab_size = 0,
            .merges = std.ArrayList(MergeRule).empty,
            .byte_to_id = byte_to_id,
            .id_to_byte = std.ArrayList(u8).empty,
        };
    }
    
    pub fn deinit(self: *BPETokenizer) void {
        self.merges.deinit(self.allocator);
        self.id_to_byte.deinit(self.allocator);
    }
    
    /// Trains the BPE tokenizer on the provided text, merging up to target_vocab_size.
    pub fn train(self: *BPETokenizer, text: []const u8, target_vocab_size: usize) !void {
        std.debug.print("--- Training BPE Tokenizer ---\n", .{});
        
        // 1. Establish base vocabulary (unique bytes in text)
        var used = [_]bool{false} ** 256;
        for (text) |b| used[b] = true;
        
        for (0..256) |i| {
            if (used[i]) {
                self.byte_to_id[i] = @as(u16, @intCast(self.vocab_size));
                try self.id_to_byte.append(self.allocator, @as(u8, @intCast(i)));
                self.vocab_size += 1;
            }
        }
        
        std.debug.print("Base Vocab Size: {d}\n", .{self.vocab_size});
        
        if (target_vocab_size <= self.vocab_size) {
            std.debug.print("Target vocab <= base vocab. No merges needed.\n", .{});
            return;
        }
        
        // Convert text to initial IDs
        var ids = std.ArrayList(u16).empty;
        try ids.ensureTotalCapacity(self.allocator, text.len);
        defer ids.deinit(self.allocator);
        
        for (text) |b| {
            ids.appendAssumeCapacity(self.byte_to_id[b]);
        }
        
        const num_merges = target_vocab_size - self.vocab_size;
        
        // Simple greedy BPE algorithm
        for (0..num_merges) |step| {
            if (ids.items.len < 2) break;
            
            // Find most frequent pair
            var pair_counts = std.AutoHashMap(u32, usize).init(self.allocator);
            defer pair_counts.deinit();
            
            var max_count: usize = 0;
            var best_pair: u32 = 0;
            
            for (0..ids.items.len - 1) |i| {
                const p1 = ids.items[i];
                const p2 = ids.items[i + 1];
                const pair_key = (@as(u32, p1) << 16) | @as(u32, p2);
                
                const gop = try pair_counts.getOrPut(pair_key);
                if (!gop.found_existing) {
                    gop.value_ptr.* = 0;
                }
                gop.value_ptr.* += 1;
                
                if (gop.value_ptr.* > max_count) {
                    max_count = gop.value_ptr.*;
                    best_pair = pair_key;
                }
            }
            
            if (max_count < 2) break; // No repeating pairs left
            
            const best_p1 = @as(u16, @intCast(best_pair >> 16));
            const best_p2 = @as(u16, @intCast(best_pair & 0xFFFF));
            const new_id = @as(u16, @intCast(self.vocab_size));
            
            try self.merges.append(self.allocator, .{ .p1 = best_p1, .p2 = best_p2, .new_id = new_id });
            self.vocab_size += 1;
            
            // Apply merge
            var new_ids = std.ArrayList(u16).empty;
            try new_ids.ensureTotalCapacity(self.allocator, ids.items.len);
            defer new_ids.deinit(self.allocator);
            
            var i: usize = 0;
            while (i < ids.items.len) {
                if (i < ids.items.len - 1 and ids.items[i] == best_p1 and ids.items[i + 1] == best_p2) {
                    new_ids.appendAssumeCapacity(new_id);
                    i += 2;
                } else {
                    new_ids.appendAssumeCapacity(ids.items[i]);
                    i += 1;
                }
            }
            
            // Swap buffers
            const old_items = try ids.toOwnedSlice(self.allocator);
            self.allocator.free(old_items);
            ids = try new_ids.clone(self.allocator);
            
            if (step % 50 == 0) {
                std.debug.print("Merge {d:0>3} | Created ID: {d} | Occurrences: {d} | Length: {d}\n", .{step, new_id, max_count, ids.items.len});
            }
        }
        
        std.debug.print("BPE Training Complete. Final Vocab Size: {d} | Final sequence length: {d}\n", .{self.vocab_size, ids.items.len});
    }

    /// Encodes a text string into a list of BPE token IDs.
    pub fn encode(self: *BPETokenizer, text: []const u8, out_ids: *std.ArrayList(u16)) !void {
        out_ids.clearRetainingCapacity();
        
        // 1. Initial mapping
        for (text) |b| {
            const id = self.byte_to_id[b];
            if (id == 9999) {
                out_ids.appendAssumeCapacity(0); // UNK token handling for PoC
            } else {
                try out_ids.append(self.allocator, id);
            }
        }
        
        // 2. Apply merges in order
        for (self.merges.items) |rule| {
            var i: usize = 0;
            var new_ids = std.ArrayList(u16).empty;
            defer new_ids.deinit(self.allocator);
            
            while (i < out_ids.items.len) {
                if (i < out_ids.items.len - 1 and out_ids.items[i] == rule.p1 and out_ids.items[i + 1] == rule.p2) {
                    try new_ids.append(self.allocator, rule.new_id);
                    i += 2;
                } else {
                    try new_ids.append(self.allocator, out_ids.items[i]);
                    i += 1;
                }
            }
            
            // Swap
            out_ids.clearRetainingCapacity();
            try out_ids.appendSlice(self.allocator, new_ids.items);
        }
    }
};

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("--- Loading Wikipedia (5MB) ---\n", .{});
    const text = @embedFile("wiki_5mb.txt");
    
    var bpe = BPETokenizer.init(allocator);
    defer bpe.deinit();
    
    // Train BPE on the 5MB text, targeting a vocab of 256.
    try bpe.train(text, 256);
    
    // Test Encoding
    var test_encoded = std.ArrayList(u16).empty;
    defer test_encoded.deinit(allocator);
    try bpe.encode("the quick brown fox", &test_encoded);
    
    std.debug.print("Encoded 'the quick brown fox':\n", .{});
    for (test_encoded.items) |id| {
        std.debug.print("{d} ", .{id});
    }
    std.debug.print("\n", .{});
}
