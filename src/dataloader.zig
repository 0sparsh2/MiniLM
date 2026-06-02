const std = @import("std");

pub const DataLoader = struct {
    allocator: std.mem.Allocator,
    full_text: []const u8,
    train_slice: []const u8,
    val_slice: []const u8,
    
    seq_len: usize,
    batch_size: usize,
    
    train_idx: usize,
    val_idx: usize,
    
    pub const vocab_size: usize = 256;

    /// Loads the text file and splits it 90/10 using raw bytes.
    pub fn init(allocator: std.mem.Allocator, text: []const u8, seq_len: usize, batch_size: usize) !DataLoader {
        // 90/10 Split
        const split_idx = (text.len * 9) / 10;
        const train_slice = text[0..split_idx];
        const val_slice = text[split_idx..];
        
        return .{
            .allocator = allocator,
            .full_text = text,
            .train_slice = train_slice,
            .val_slice = val_slice,
            .seq_len = seq_len,
            .batch_size = batch_size,
            .train_idx = 0,
            .val_idx = 0,
        };
    }

    pub fn deinit(self: *DataLoader) void {
        _ = self;
    }

    /// Fills the provided buffers with the next batch of training data using a sliding window.
    /// Expects x_buf and y_buf to be []usize or []u16, so we cast the bytes.
    pub fn nextTrainBatch(self: *DataLoader, x_buf: []usize, y_buf: []usize) void {
        std.debug.assert(x_buf.len == self.batch_size * self.seq_len);
        std.debug.assert(y_buf.len == self.batch_size);
        
        for (0..self.batch_size) |b| {
            if (self.train_idx + self.seq_len >= self.train_slice.len) {
                self.train_idx = 0; // Loop back
            }
            
            const start = self.train_idx;
            const end = start + self.seq_len;
            
            for (0..self.seq_len) |i| {
                x_buf[b * self.seq_len + i] = @as(usize, self.train_slice[start + i]);
            }
            y_buf[b] = @as(usize, self.train_slice[end]);
            
            self.train_idx += self.seq_len; // Non-overlapping chunk for speed
        }
    }

    /// Fills the provided buffers with the next batch of validation data.
    pub fn nextValBatch(self: *DataLoader, x_buf: []usize, y_buf: []usize) void {
        std.debug.assert(x_buf.len == self.batch_size * self.seq_len);
        std.debug.assert(y_buf.len == self.batch_size);
        
        for (0..self.batch_size) |b| {
            if (self.val_idx + self.seq_len >= self.val_slice.len) {
                self.val_idx = 0; // Loop back
            }
            
            const start = self.val_idx;
            const end = start + self.seq_len;
            
            for (0..self.seq_len) |i| {
                x_buf[b * self.seq_len + i] = @as(usize, self.val_slice[start + i]);
            }
            y_buf[b] = @as(usize, self.val_slice[end]);
            
            self.val_idx += self.seq_len;
        }
    }
};
