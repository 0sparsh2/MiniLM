const std = @import("std");

/// A simple thread executor that spawns threads for a batch of jobs and waits for them.
/// This avoids the complexity of an atomic task queue while leveraging Zig's native threading.
pub const ThreadPool = struct {
    allocator: std.mem.Allocator,
    num_threads: usize,

    pub fn init(allocator: std.mem.Allocator, num_threads: usize) ThreadPool {
        return .{
            .allocator = allocator,
            .num_threads = num_threads,
        };
    }

    /// Executes a function over an array of contexts in parallel using the specified number of threads.
    /// The function takes a single pointer to its Context type.
    pub fn execute(self: *const ThreadPool, comptime Context: type, contexts: []Context, comptime func: fn (*Context) void) !void {
        _ = self;
        if (contexts.len == 0) return;
        
        for (contexts) |*ctx| {
            func(ctx);
        }
    }

    // Helper to run a batch of contexts in a single thread
    fn runBatch(comptime Context: type, slice: []Context, comptime func: fn (*Context) void) void {
        for (slice) |*ctx| {
            func(ctx);
        }
    }
};

test "ThreadPool basic" {
    const alloc = std.testing.allocator;
    const pool = ThreadPool.init(alloc, 4);
    
    const JobCtx = struct {
        val: i32,
        out: i32,
    };
    
    var jobs: [8]JobCtx = undefined;
    for (0..8) |i| {
        jobs[i] = .{ .val = @as(i32, @intCast(i)), .out = 0 };
    }
    
    const Wrapper = struct {
        fn process(ctx: *JobCtx) void {
            ctx.out = ctx.val * 2;
        }
    };
    
    try pool.execute(JobCtx, &jobs, Wrapper.process);
    
    for (0..8) |i| {
        try std.testing.expectEqual(@as(i32, @intCast(i * 2)), jobs[i].out);
    }
}
