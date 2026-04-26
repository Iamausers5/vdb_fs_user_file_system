/**
 * @file  standalone_sync.cu
 * @brief 独立的 CPU-GPU 亚微秒级双向同步无锁环形队列测试代码
 * * 编译命令: nvcc standalone_sync.cu -o standalone_sync -std=c++11 -lpthread
 * 运行命令: ./standalone_sync
 */

#include <cuda_runtime.h>
#include <iostream>
#include <atomic>
#include <thread>
#include <functional>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <unistd.h>

#if defined(__x86_64__) || defined(__i386__)
    #include <immintrin.h>   // _mm_pause
    #define VDB_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__)
    #define VDB_CPU_RELAX() asm volatile("yield" ::: "memory")
#else
    #define VDB_CPU_RELAX() ((void)0)
#endif

// ============================================================================
// 1. 模拟缺失的项目依赖类型与宏定义 (Mocked Dependencies)
// ============================================================================
#define VDB_HOST_DEVICE __host__ __device__
#define VDB_LIKELY(x) __builtin_expect(!!(x), 1)

#define VDB_INFO(fmt, ...) printf("[INFO] " fmt "\n", ##__VA_ARGS__)
#define VDB_WARN(fmt, ...) printf("[WARN] " fmt "\n", ##__VA_ARGS__)
#define VDB_ERR(fmt, ...)  printf("[ERR]  " fmt "\n", ##__VA_ARGS__)

using ino_t = uint64_t;
using fd_t = int;
constexpr uint32_t IFLAG_NONE = 0;

enum class VdbStatus {
    OK = 0,
    ERR_INVAL = 1,
    ERR_BUSY = 2,
    ERR_NOMEM = 3,
    ERR_GDRCOPY = 4
};

// ============================================================================
// 2. 环形队列布局参数与数据结构
// ============================================================================
static constexpr uint64_t VDB_SYNC_BUFFER_SIZE = 16ULL * 1024 * 1024; // 16 MiB
static constexpr uint32_t VDB_SYNC_RING_SIZE   = 65536;               // power of 2
static constexpr uint32_t VDB_SYNC_RING_MASK   = VDB_SYNC_RING_SIZE - 1;

static constexpr uint32_t GPU_CACHE_LINE       = 128;
static constexpr uint32_t CPU_CACHE_LINE       = 64;

enum SyncFlag : uint32_t {
    FLAG_EMPTY    = 0,
    FLAG_RESERVED = 1,
    FLAG_READY    = 2,
};

enum SyncOpCode : uint32_t {
    OP_NONE   = 0,
    OP_READ   = 1,
    OP_WRITE  = 2,
};

struct alignas(16) AddrEntry {
    void* gpu_addr;
    uint64_t length;
};

struct alignas(64) IoParams {
    uint32_t op_type;
    uint32_t flags;
    ino_t    ino;
    uint32_t _pad0;
    uint64_t offset;
    uint64_t size;
    fd_t     fd;
    uint32_t seq;
    uint64_t user_tag;
    uint64_t _pad1[2];
};

struct alignas(GPU_CACHE_LINE) FlagSlot {
    uint32_t ready;
    uint32_t _pad[(GPU_CACHE_LINE / 4) - 1];
};

struct alignas(4096) SyncHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t capacity;
    uint32_t ring_mask;
    alignas(CPU_CACHE_LINE) uint64_t prod_idx;
    alignas(CPU_CACHE_LINE) uint64_t cons_idx;
    alignas(CPU_CACHE_LINE) uint64_t last_produce_ns;
    uint64_t _reserved[64];
};

static constexpr uint32_t VDB_SYNC_MAGIC   = 0x56444253;

struct VdbSyncBuffer {
    SyncHeader* header;
    AddrEntry* addr_list;
    IoParams* params;
    FlagSlot* flags;
    uint8_t* reserved;
    uint8_t* base;
};

VDB_HOST_DEVICE inline void vdb_sync_buffer_init_view(VdbSyncBuffer* buf, void* base_ptr) {
    uint8_t* base = static_cast<uint8_t*>(base_ptr);
    buf->base      = base;
    buf->header    = reinterpret_cast<SyncHeader*>(base + 0);
    buf->addr_list = reinterpret_cast<AddrEntry*>(base + 4096);
    buf->params    = reinterpret_cast<IoParams*>(base + 4096 + (1ULL << 20));
    buf->flags     = reinterpret_cast<FlagSlot*>(base + 4096 + (5ULL << 20));
    buf->reserved  = base + 4096 + (13ULL << 20);
}

// ============================================================================
// 3. GPU 设备端代码 (提交任务)
// ============================================================================
__device__ __forceinline__
uint32_t vdb_sync_gpu_submit(VdbSyncBuffer* buf,
                             SyncOpCode        op,
                             ino_t             ino,
                             uint64_t          offset,
                             uint64_t          size,
                             void* gpu_addr,
                             fd_t              fd)
{
    // 1. 获取序号
    const uint64_t my_seq = atomicAdd_system(
        reinterpret_cast<unsigned long long*>(&buf->header->prod_idx), 1ULL);
    const uint32_t idx = static_cast<uint32_t>(my_seq & buf->header->ring_mask);

    // 2. 自旋等待空槽位并预占
    constexpr uint64_t HARD_WAIT_NS = 100'000'000ULL;
    uint32_t backoff = 32;
    uint64_t waited  = 0;

    while (atomicCAS_system(&buf->flags[idx].ready, FLAG_EMPTY, FLAG_RESERVED) != FLAG_EMPTY) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(backoff);
#else
        for (volatile int i = 0; i < 256; ++i) { /* spin */ }
#endif
        waited  += backoff;
        backoff  = backoff < 4096 ? backoff * 2 : backoff;
        if (waited > HARD_WAIT_NS) return 0xFFFFFFFFu;
    }

    // 3. 写入参数
    buf->addr_list[idx].gpu_addr = gpu_addr;
    buf->addr_list[idx].length   = size;

    IoParams& p = buf->params[idx];
    p.op_type  = static_cast<uint32_t>(op);
    p.flags    = IFLAG_NONE;
    p.ino      = ino;
    p.offset   = offset;
    p.size     = size;
    p.fd       = fd;
    p.seq      = static_cast<uint32_t>(my_seq);
    p.user_tag = 0;

    // 4. 内存屏障
    __threadfence_system();

    // 5. 提交状态
    atomicExch_system(&buf->flags[idx].ready, FLAG_READY);

    return idx;
}

// ============================================================================
// 4. CPU 主机端类实现 (轮询与管理)
// ============================================================================
using SyncSlotCallback = std::function<void(uint32_t slot_idx, const AddrEntry& addr, const IoParams& params)>;

class VdbSyncChannel {
public:
    VdbSyncChannel() = default;
    ~VdbSyncChannel() { shutdown(); }

    VdbStatus init() {
        const unsigned flags = cudaHostAllocMapped | cudaHostAllocPortable;
        cudaError_t err = cudaHostAlloc(&host_ptr_, VDB_SYNC_BUFFER_SIZE, flags);
        if (err != cudaSuccess) return VdbStatus::ERR_NOMEM;

        err = cudaHostGetDevicePointer(&device_ptr_, host_ptr_, 0);
        if (err != cudaSuccess) return VdbStatus::ERR_GDRCOPY;

        std::memset(host_ptr_, 0, VDB_SYNC_BUFFER_SIZE);
        vdb_sync_buffer_init_view(&host_view_, host_ptr_);
        vdb_sync_buffer_init_view(&device_view_, device_ptr_);

        SyncHeader* h = host_view_.header;
        h->magic      = VDB_SYNC_MAGIC;
        h->capacity   = VDB_SYNC_RING_SIZE;
        h->ring_mask  = VDB_SYNC_RING_MASK;
        
        return VdbStatus::OK;
    }

    void shutdown() {
        if (running_.exchange(false, std::memory_order_release)) {
            if (poller_thread_.joinable()) poller_thread_.join();
        }
        if (host_ptr_) {
            cudaFreeHost(host_ptr_);
            host_ptr_ = nullptr;
        }
    }

    VdbStatus start_poller(SyncSlotCallback cb) {
        callback_ = std::move(cb);
        running_.store(true, std::memory_order_release);
        poller_thread_ = std::thread(&VdbSyncChannel::poll_loop, this);
        return VdbStatus::OK;
    }

    const VdbSyncBuffer& device_view() const { return device_view_; }

private:
    void poll_loop() {
        uint32_t  cons_idx    = 0;
        int       idle_count  = 0;
        long      sleep_ns    = 1000;

        FlagSlot* const flags     = host_view_.flags;
        const AddrEntry* const addr_list = host_view_.addr_list;
        const IoParams* const params    = host_view_.params;

        while (running_.load(std::memory_order_relaxed)) {
            const uint32_t flag = __atomic_load_n(&flags[cons_idx].ready, __ATOMIC_ACQUIRE);

            if (VDB_LIKELY(flag == FLAG_READY)) {
                // 触发回调消费数据
                callback_(cons_idx, addr_list[cons_idx], params[cons_idx]);

                // 归还槽位给 GPU
                __atomic_store_n(&flags[cons_idx].ready, FLAG_EMPTY, __ATOMIC_RELEASE);

                cons_idx = (cons_idx + 1) & VDB_SYNC_RING_MASK;
                idle_count = 0;
                sleep_ns   = 1000;
                continue;
            }

            // 退避策略
            ++idle_count;
            if (idle_count < 1024) {
                VDB_CPU_RELAX();
            } else if (idle_count < 8192) {
                sched_yield();
            } else {
                struct timespec ts{0, sleep_ns};
                clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, nullptr);
                sleep_ns = std::min<long>(sleep_ns * 2, 1000000);
            }
        }
    }

    void* host_ptr_ = nullptr;
    void* device_ptr_ = nullptr;
    VdbSyncBuffer host_view_{};
    VdbSyncBuffer device_view_{};
    std::thread poller_thread_;
    std::atomic<bool> running_{false};
    SyncSlotCallback callback_;
};

// ============================================================================
// 5. 测试用例: GPU Kernel 发起高并发请求
// ============================================================================
__global__ void benchmark_submit_kernel(VdbSyncBuffer device_view, int tasks_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = 0; i < tasks_per_thread; ++i) {
        // 模拟每次请求读取不同的文件偏移
        uint64_t offset = (tid * tasks_per_thread + i) * 4096;
        
        uint32_t slot = vdb_sync_gpu_submit(
            &device_view, 
            OP_READ, 
            999,          // dummy inode
            offset, 
            4096, 
            nullptr,      // dummy address
            10            // dummy fd
        );
        
        if (slot == 0xFFFFFFFFu) {
            printf("[GPU] Thread %d ERROR: Queue full & timeout!\n", tid);
        }
    }
}

// ============================================================================
// 6. Main 入口
// ============================================================================
int main() {
    VDB_INFO("初始化 VdbSyncChannel (分配 16MB 锁页内存)...");
    VdbSyncChannel channel;
    
    if (channel.init() != VdbStatus::OK) {
        VDB_ERR("初始化失败!");
        return -1;
    }

    // 统计 CPU 收到的任务数
    std::atomic<int> tasks_received{0};

    // 消费者回调函数 (CPU 处理逻辑)
    auto cpu_consumer_callback = [&](uint32_t slot_idx, const AddrEntry& addr, const IoParams& params) {
        tasks_received.fetch_add(1, std::memory_order_relaxed);
        // 如果想看详细打印可以取消下面这行注释，但在压测时会导致终端卡顿
        // printf("[CPU] Received task -> Slot: %u | Offset: %llu\n", slot_idx, params.offset);
    };

    VDB_INFO("启动 CPU 轮询线程...");
    channel.start_poller(cpu_consumer_callback);

    // 压测参数：256 个线程，每个线程提交 1000 个任务，总计 256,000 个任务
    int num_blocks = 2;
    int threads_per_block = 128;
    int tasks_per_thread = 1000;
    int total_tasks = num_blocks * threads_per_block * tasks_per_thread;

    VDB_INFO("启动 GPU 测试 Kernel...");
    VDB_INFO("配置: %d 线程并发，总任务数: %d. 开始压测...", num_blocks * threads_per_block, total_tasks);

    // 记录开始时间
    auto start_time = std::chrono::high_resolution_clock::now();

    // 启动 GPU Kernel
    benchmark_submit_kernel<<<num_blocks, threads_per_block>>>(channel.device_view(), tasks_per_thread);
    
    // 等待 GPU 提交完成
    cudaDeviceSynchronize();

    // 等待 CPU 消费完所有任务
    while (tasks_received.load() < total_tasks) {
        usleep(1000);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

    VDB_INFO("===========================================");
    VDB_INFO("压测完成!");
    VDB_INFO("总共成功流转任务数 : %d", tasks_received.load());
    VDB_INFO("总耗时 (包括提交与消费): %.2f ms", elapsed.count());
    VDB_INFO("吞吐量 : %.2f Million IOPS (请求/秒)", (total_tasks / (elapsed.count() / 1000.0)) / 1000000.0);
    VDB_INFO("===========================================");

    VDB_INFO("正在关闭通道...");
    channel.shutdown();
    VDB_INFO("退出程序。");

    return 0;
}
