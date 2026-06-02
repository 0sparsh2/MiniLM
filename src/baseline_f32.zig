const std = @import("std");
const tokenizer = @import("tokenizer.zig");

// A standard f32 baseline implementing a tiny transformer block (Attention + Linear Head)
// using standard floating-point math and backpropagation.
pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const text = @embedFile("tiny_shakespeare.txt");
    
    var tok = try tokenizer.Tokenizer.init(allocator, text);
    defer tok.deinit();
    
    const encoded_text = try allocator.alloc(u8, text.len);
    defer allocator.free(encoded_text);
    tok.encode(text, encoded_text);
    
    const vocab_size = tok.vocab_size;
    const seq_len = 16;
    const embed_dim = vocab_size;
    
    // --- Allocate Weights (f32) ---
    // Attention: Wq, Wk, Wv, Wo
    var wq = try allocator.alloc(f32, embed_dim * embed_dim);
    var wk = try allocator.alloc(f32, embed_dim * embed_dim);
    var wv = try allocator.alloc(f32, embed_dim * embed_dim);
    var wo = try allocator.alloc(f32, embed_dim * embed_dim);
    
    // Head: W_head
    var w_head = try allocator.alloc(f32, embed_dim * vocab_size);
    
    // Deterministic pseudo-random initialization
    const scale = 1.0 / @as(f32, @floatFromInt(embed_dim));
    for (0..wq.len) |i| wq[i] = (@as(f32, @floatFromInt((i * 13) % 100)) / 50.0 - 1.0) * scale;
    for (0..wk.len) |i| wk[i] = (@as(f32, @floatFromInt((i * 17) % 100)) / 50.0 - 1.0) * scale;
    for (0..wv.len) |i| wv[i] = (@as(f32, @floatFromInt((i * 19) % 100)) / 50.0 - 1.0) * scale;
    for (0..wo.len) |i| wo[i] = (@as(f32, @floatFromInt((i * 23) % 100)) / 50.0 - 1.0) * scale;
    for (0..w_head.len) |i| w_head[i] = (@as(f32, @floatFromInt((i * 29) % 100)) / 50.0 - 1.0) * scale;

    // --- Buffers ---
    var x = try allocator.alloc(f32, seq_len * embed_dim);
    var q = try allocator.alloc(f32, seq_len * embed_dim);
    var k = try allocator.alloc(f32, seq_len * embed_dim);
    var v = try allocator.alloc(f32, seq_len * embed_dim);
    var scores = try allocator.alloc(f32, seq_len * seq_len);
    var context = try allocator.alloc(f32, seq_len * embed_dim);
    var attn_out = try allocator.alloc(f32, seq_len * embed_dim);
    var head_out = try allocator.alloc(f32, vocab_size);
    
    // Gradients for Head only (Forward-Forward local training comparison)
    var grad_w_head = try allocator.alloc(f32, embed_dim * vocab_size);

    const num_epochs = 1000;
    const start_idx = 0;
    const lr: f32 = 0.01;
    
    std.debug.print("--- Starting f32 Baseline Training ---\n", .{});

    for (0..num_epochs) |epoch| {
        // --- PREPARE INPUT ---
        @memset(x, 0.0);
        for (0..seq_len) |t| {
            const char_id = encoded_text[start_idx + t];
            x[t * embed_dim + char_id] = 1.0;
        }
        
        const target_char_id = encoded_text[start_idx + seq_len];
        
        // --- FORWARD PASS ---
        // 1. Q, K, V Projections
        for (0..seq_len) |t| {
            for (0..embed_dim) |out_d| {
                var sum_q: f32 = 0;
                var sum_k: f32 = 0;
                var sum_v: f32 = 0;
                for (0..embed_dim) |in_d| {
                    const in_val = x[t * embed_dim + in_d];
                    sum_q += in_val * wq[out_d * embed_dim + in_d];
                    sum_k += in_val * wk[out_d * embed_dim + in_d];
                    sum_v += in_val * wv[out_d * embed_dim + in_d];
                }
                q[t * embed_dim + out_d] = sum_q;
                k[t * embed_dim + out_d] = sum_k;
                v[t * embed_dim + out_d] = sum_v;
            }
        }
        
        // 2. Attention Scores (Q * K^T) + Softmax
        @memset(scores, 0.0);
        for (0..seq_len) |t_q| {
            var max_score: f32 = -999999.0;
            for (0..seq_len) |t_k| {
                if (t_k > t_q) continue; // Causal mask
                
                var score: f32 = 0;
                for (0..embed_dim) |d| {
                    score += q[t_q * embed_dim + d] * k[t_k * embed_dim + d];
                }
                scores[t_q * seq_len + t_k] = score;
                if (score > max_score) max_score = score;
            }
            
            // Softmax
            var sum_exp: f32 = 0;
            for (0..seq_len) |t_k| {
                if (t_k > t_q) continue;
                const exp_val = @exp(scores[t_q * seq_len + t_k] - max_score);
                scores[t_q * seq_len + t_k] = exp_val;
                sum_exp += exp_val;
            }
            
            for (0..seq_len) |t_k| {
                if (t_k > t_q) continue;
                scores[t_q * seq_len + t_k] /= sum_exp;
            }
        }
        
        // 3. Multiply by V
        @memset(context, 0.0);
        for (0..seq_len) |t_q| {
            for (0..seq_len) |t_k| {
                if (t_k > t_q) continue;
                const s = scores[t_q * seq_len + t_k];
                for (0..embed_dim) |d| {
                    context[t_q * embed_dim + d] += s * v[t_k * embed_dim + d];
                }
            }
        }
        
        // 4. Output Projection Wo
        @memset(attn_out, 0.0);
        for (0..seq_len) |t| {
            for (0..embed_dim) |out_d| {
                var sum: f32 = 0;
                for (0..embed_dim) |in_d| {
                    sum += context[t * embed_dim + in_d] * wo[out_d * embed_dim + in_d];
                }
                attn_out[t * embed_dim + out_d] = sum;
            }
        }
        
        // 5. Head Classification (using last token)
        const last_token_ctx = attn_out[(seq_len - 1) * embed_dim .. seq_len * embed_dim];
        @memset(head_out, 0.0);
        for (0..vocab_size) |out_d| {
            var sum: f32 = 0;
            for (0..embed_dim) |in_d| {
                sum += last_token_ctx[in_d] * w_head[out_d * embed_dim + in_d];
            }
            head_out[out_d] = sum;
        }
        
        // --- LOSS AND BACKPROPAGATION (Local to Head) ---
        var loss: f32 = 0;
        @memset(grad_w_head, 0.0);
        
        for (0..vocab_size) |v_idx| {
            const target_val: f32 = if (v_idx == target_char_id) 1.0 else 0.0;
            const diff = head_out[v_idx] - target_val;
            loss += diff * diff;
            
            const grad = diff; // Derivative of MSE
            for (0..embed_dim) |in_d| {
                grad_w_head[v_idx * embed_dim + in_d] = grad * last_token_ctx[in_d];
            }
        }
        
        // Update Head Weights (SGD)
        for (0..w_head.len) |i| {
            w_head[i] -= lr * grad_w_head[i];
        }
        
        if (epoch % 100 == 0 or epoch == num_epochs - 1) {
            var max_val: f32 = -999999.0;
            var max_idx: u8 = 0;
            for (0..vocab_size) |v_idx| {
                if (head_out[v_idx] > max_val) {
                    max_val = head_out[v_idx];
                    max_idx = @as(u8, @intCast(v_idx));
                }
            }
            
            var pred_str: [1]u8 = undefined;
            tok.decode(&[_]u8{max_idx}, &pred_str);
            var targ_str: [1]u8 = undefined;
            tok.decode(&[_]u8{target_char_id}, &targ_str);
            
            std.debug.print("Epoch {d:0>4} | Loss: {d:.4} | Pred: '{s}' | Target: '{s}'\n", .{epoch, loss, pred_str, targ_str});
        }
    }
}
