const std = @import("std");
const stochastic = @import("stochastic.zig");
const binary_kernel = @import("binary_kernel.zig");

/// A sub-megabyte LLM Architecture using Holographic Vector Symbolic logic.
/// It uses 0 parameters for embeddings and attention, mixing time and tokens via O(N) Bitwise logic.
/// The ONLY parameters are the final Readout weights (mapping the high-dimensional hash to tokens).
pub const HoloReservoir = struct {
    dim: usize,          // e.g. 8192 for high capacity
    vocab_size: usize,   // 256
    packed_dim: usize,
    
    // Readout Layer: 8192 dims -> 256 tokens = 2,097,152 bits = 262 Kilobytes!
    readout_weights: []u64,
    readout_shadow: []i16,

    pub fn init(allocator: std.mem.Allocator, dim: usize, vocab_size: usize) !HoloReservoir {
        std.debug.assert(dim % 64 == 0);
        const packed_dim = dim / 64;

        const readout_weights = try allocator.alloc(u64, packed_dim * vocab_size);
        const readout_shadow = try allocator.alloc(i16, dim * vocab_size);

        @memset(readout_weights, 0);
        @memset(readout_shadow, 0);

        return .{
            .dim = dim,
            .vocab_size = vocab_size,
            .packed_dim = packed_dim,
            .readout_weights = readout_weights,
            .readout_shadow = readout_shadow,
        };
    }

    pub fn deinit(self: *HoloReservoir, allocator: std.mem.Allocator) void {
        allocator.free(self.readout_weights);
        allocator.free(self.readout_shadow);
    }

    pub fn syncWeights(self: *HoloReservoir) void {
        const shadow_i8 = std.heap.page_allocator.alloc(i8, self.readout_shadow.len) catch unreachable;
        defer std.heap.page_allocator.free(shadow_i8);
        for (self.readout_shadow, 0..) |v, i| shadow_i8[i] = if (v >= 0) 1 else -1;
        binary_kernel.packBinary(shadow_i8, self.readout_weights);
    }

    pub fn randomize(self: *HoloReservoir, rng: *stochastic.VectorXorshift32) void {
        for (0..self.readout_shadow.len) |i| self.readout_shadow[i] = @as(i16, @intCast(rng.next()[0] % 50)) - 25;
        self.syncWeights();
    }

    /// O(N) Zero-Weight Token Embedding (Deterministic Random Projection)
    fn embedToken(self: *HoloReservoir, token_id: u8, out_packed: []u64) void {
        var rng = stochastic.VectorXorshift32.init(@as(u32, token_id) * 1234567 + 89123);
        for (0..self.packed_dim) |i| {
            const v1 = rng.next()[0];
            const v2 = rng.next()[0];
            out_packed[i] = (@as(u64, v1) << 32) | @as(u64, v2);
        }
    }

    /// O(N) Circular Bit Shift Left (Attention/Time Mixing)
    fn cyclicShiftLeft(vec: []u64) void {
        const len = vec.len;
        var carry_in: u64 = vec[len - 1] >> 63;
        for (0..len) |i| {
            const carry_out = vec[i] >> 63;
            vec[i] = (vec[i] << 1) | carry_in;
            carry_in = carry_out;
        }
    }

    /// Forward pass through the entire sequence.
    /// Operates completely sequentially, accumulating into state_packed.
    pub fn forwardTraining(
        self: *HoloReservoir, 
        allocator: std.mem.Allocator,
        sequence: []const u8, 
        target_char_id: usize,
        out_pred: *usize, 
        out_loss: *i64, 
        shift_scale: u4, 
        rng: *stochastic.VectorXorshift32
    ) !void {
        const state_packed = try allocator.alloc(u64, self.packed_dim);
        defer allocator.free(state_packed);
        @memset(state_packed, 0);

        const embed_packed = try allocator.alloc(u64, self.packed_dim);
        defer allocator.free(embed_packed);

        // Holographic Temporal Binding:
        // We mix context by continually shifting the state vector and XORing the new token embedding.
        for (0..sequence.len - 1) |t| {
            const token = sequence[t];
            self.embedToken(token, embed_packed);
            
            cyclicShiftLeft(state_packed); // Time shift
            for (0..self.packed_dim) |i| {
                state_packed[i] ^= embed_packed[i]; // Bind
                // Chaos non-linearity (Rule 30-like mixing) to act as a virtual deep layer
                state_packed[i] ^= (state_packed[i] << 3) ^ (state_packed[i] >> 7);
            }
        }

        // The final state_packed represents a unique mathematical hash of the entire sequence context!

        // --- READOUT LAYER (The only trainable part) ---
        const local_out = try allocator.alloc(i32, self.vocab_size);
        defer allocator.free(local_out);
        
        for (0..self.vocab_size) |v| {
            const w_row = self.readout_weights[v * self.packed_dim .. (v + 1) * self.packed_dim];
            local_out[v] = binary_kernel.dotProductBinary(w_row, state_packed);
        }

        const local_error = try allocator.alloc(i32, self.vocab_size);
        defer allocator.free(local_error);
        
        var max_val: i32 = -999999;
        var max_idx: u16 = 0;
        var loss: i64 = 0;
        
        const margin: i32 = @as(i32, @intCast(self.dim / 8)); // Adaptive Margin based on dims
        
        for (0..self.vocab_size) |v| {
            const current_val = local_out[v];
            var diff: i32 = 0;
            
            if (v == target_char_id) {
                if (current_val < margin) diff = current_val - margin;
            } else {
                if (current_val > -margin) diff = current_val + margin;
            }
            
            loss += @as(i64, diff) * @as(i64, diff);
            local_error[v] = diff;
            
            if (current_val > max_val) {
                max_val = current_val;
                max_idx = @as(u16, @intCast(v));
            }
        }
        
        out_pred.* = max_idx;
        out_loss.* = loss;

        // Local Stochastic Update
        var gradients = try allocator.alloc(i8, self.readout_shadow.len);
        defer allocator.free(gradients);
        @memset(gradients, 0);

        for (0..self.vocab_size) |out_idx| {
            const row_start = out_idx * self.dim;
            const err = local_error[out_idx];
            
            var grad = -err;
            if (grad > 127) grad = 127;
            if (grad < -128) grad = -128;
            
            const grad_pos = @as(i8, @intCast(grad));
            const grad_neg = -grad_pos;
            
            for (0..self.dim) |in_idx| {
                const block_idx = in_idx / 64;
                const bit_idx = @as(u6, @intCast(in_idx % 64));
                const bit = (state_packed[block_idx] >> bit_idx) & 1;
                
                gradients[row_start + in_idx] = if (bit == 1) grad_pos else grad_neg;
            }
        }
        
        stochastic.stochasticUpdate(self.readout_shadow, gradients, shift_scale, rng);
        self.syncWeights();
    }

    pub fn save(self: *HoloReservoir, writer: anytype) !void {
        try writer.writeAll(std.mem.sliceAsBytes(self.readout_weights));
    }
    
    pub fn load(self: *HoloReservoir, reader: anytype) !void {
        _ = try reader.readAll(std.mem.sliceAsBytes(self.readout_weights));
    }
}
