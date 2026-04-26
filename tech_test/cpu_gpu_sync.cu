// =============================================================================
// shm_sync_test_v2.cu — 修复 PCIe AtomicOp 在虚拟化环境不可用的问题
//
// v1 (shm_sync_test.cu) 在 GPU 端用 atomicAdd_system(addr, 0) 做 acquire load。
// 这条 intrinsic 编译为 atom.add.sys.u32,执行时发出 PCIe AtomicOp (FetchAdd) TLP。
// 在虚拟化 / 云 GPU 环境(阿里云 ECS、AWS、GCP 大部分实例类型)中:
//   * Hypervisor 的 IOMMU 不透传 AtomicOp routing
//   * SR-IOV 虚拟功能不暴露 AtomicOp capability
//   * 结果:GPU 的 FetchAdd 失效,读到陈旧的 cache 值
//
// 修复:用 PTX `ld.acquire.sys.b32` 指令,这是普通 PCIe Memory Read + acquire
// 屏障,不依赖 PCIe AtomicOp,任何环境都能用。
//
// 编译:nvcc -O2 -std=c++17 -arch=sm_70 shm_sync_test_v2.cu -o shm_sync_test
// =============================================================================

#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>

#define CHECK(call) do {                                                       \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                     __FILE__, __LINE__, cudaGetErrorString(e));               \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

using clk = std::chrono::steady_clock;
static double us_since(clk::time_point t0) {
    return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}

// -----------------------------------------------------------------------------
// ★ 核心 helper:system-scope acquire load via PTX
//
// `ld.acquire.sys.b32` 在 SM_70+ 是原生指令。它做两件事:
//   1) 发出一个普通的 PCIe Memory Read TLP 读取 host RAM
//      (不是 AtomicOp,任何 PCIe 平台都支持)
//   2) 在 GPU 端插入 acquire 屏障,保证后续的 load/store 不会被
//      reorder 到这个 load 之前
//
// 等价于 C++ memory_order_acquire 的 system scope 版本。
// 这是替代 atomicAdd_system(addr, 0) 的正确做法。
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t load_acq_sys_u32(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.b32 %0, [%1];"
                 : "=r"(v)
                 : "l"(p)
                 : "memory");
    return v;
}

// -----------------------------------------------------------------------------
// 共享数据结构(与 v1 完全相同)
// -----------------------------------------------------------------------------
struct alignas(128) SharedSlot {
    uint32_t gpu_to_cpu_flag;
    uint32_t pad0[31];

    uint64_t payload[16];
    uint32_t pad1[16];

    uint32_t cpu_to_gpu_flag;
    uint32_t pad2[31];

    uint32_t pingpong_request;
    uint32_t pingpong_response;
    uint32_t pad3[30];
};
static_assert(sizeof(SharedSlot) % 128 == 0, "must be cache-line multiple");

// -----------------------------------------------------------------------------
// Test 1 与 Test 2 的 GPU kernel 不动:
// GPU→CPU 方向用 atomicExch_system 写,这条在硬件不支持 PCIe AtomicOp 时
// 驱动会 fallback 到"普通 PCIe Write + 屏障",语义等价,所以工作正常。
// -----------------------------------------------------------------------------
__global__ void k_gpu_to_cpu_simple(SharedSlot* dev_slot) {
    __threadfence_system();
    atomicExch_system(&dev_slot->gpu_to_cpu_flag, 1u);
}

bool test_gpu_to_cpu_simple(SharedSlot* host_slot, SharedSlot* dev_slot) {
    std::printf("\n[Test 1] GPU→CPU 单向通知\n");
    host_slot->gpu_to_cpu_flag = 0;

    auto t0 = clk::now();
    k_gpu_to_cpu_simple<<<1, 1>>>(dev_slot);

    int spins = 0;
    while (__atomic_load_n(&host_slot->gpu_to_cpu_flag, __ATOMIC_ACQUIRE) != 1u) {
        if (++spins > 100'000'000) {
            std::printf("  FAIL: spin %d 次未见 flag=1\n", spins); return false;
        }
    }
    double us = us_since(t0);
    CHECK(cudaDeviceSynchronize());
    std::printf("  PASS: 见到 flag=1,自 launch 起 %.2f μs,spin=%d\n", us, spins);
    return true;
}

__global__ void k_producer(SharedSlot* dev_slot, uint64_t round_id) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        dev_slot->payload[i] = (round_id << 16) | static_cast<uint64_t>(i);
    }
    __threadfence_system();
    atomicExch_system(&dev_slot->gpu_to_cpu_flag, 1u);
}

bool test_producer_consumer(SharedSlot* host_slot, SharedSlot* dev_slot) {
    std::printf("\n[Test 2] GPU producer / CPU consumer payload 完整性\n");
    constexpr int kRounds = 10000;
    int torn = 0, late = 0;

    for (int r = 1; r <= kRounds; ++r) {
        host_slot->gpu_to_cpu_flag = 0;
        k_producer<<<1, 1>>>(dev_slot, static_cast<uint64_t>(r));
        while (__atomic_load_n(&host_slot->gpu_to_cpu_flag, __ATOMIC_ACQUIRE) != 1u) {
        #if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
        #endif
        }
        for (int i = 0; i < 16; ++i) {
            uint64_t expect = (static_cast<uint64_t>(r) << 16) | static_cast<uint64_t>(i);
            if (host_slot->payload[i] != expect) {
                if (((host_slot->payload[i] >> 16) & 0xFFFF) != static_cast<uint64_t>(r))
                    ++late; else ++torn;
                break;
            }
        }
        CHECK(cudaDeviceSynchronize());
    }
    if (torn == 0 && late == 0) {
        std::printf("  PASS: %d 轮全部正确\n", kRounds); return true;
    }
    std::printf("  FAIL: torn=%d late=%d\n", torn, late);
    return false;
}

// -----------------------------------------------------------------------------
// ★ Test 3 修复:GPU spin loop 用 load_acq_sys_u32 而非 atomicAdd_system
// -----------------------------------------------------------------------------
__global__ void k_cpu_to_gpu_consumer(SharedSlot* dev_slot,
                                       uint32_t* d_observed,
                                       uint64_t  spin_limit)
{
    uint64_t spins = 0;
    uint32_t v;
    do {
        // ★ 改动:用 ld.acquire.sys 替代 atomicAdd_system(addr, 0)
        // 不需要 PCIe AtomicOp 支持
        v = load_acq_sys_u32(&dev_slot->cpu_to_gpu_flag);
        if (++spins >= spin_limit) break;
    } while (v != 1u);
    *d_observed = v;
}

bool test_cpu_to_gpu(SharedSlot* host_slot, SharedSlot* dev_slot) {
    std::printf("\n[Test 3] CPU→GPU 反向通知\n");

    uint32_t* d_observed = nullptr;
    CHECK(cudaMalloc(&d_observed, sizeof(uint32_t)));
    CHECK(cudaMemset(d_observed, 0, sizeof(uint32_t)));
    host_slot->cpu_to_gpu_flag = 0;

    k_cpu_to_gpu_consumer<<<1, 1>>>(dev_slot, d_observed, 100'000'000ull);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    auto t0 = clk::now();
    __atomic_store_n(&host_slot->cpu_to_gpu_flag, 1u, __ATOMIC_RELEASE);
    CHECK(cudaDeviceSynchronize());
    double us = us_since(t0);

    uint32_t observed = 0;
    CHECK(cudaMemcpy(&observed, d_observed, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_observed);

    if (observed == 1u) {
        std::printf("  PASS: GPU 见到 flag=1,自 store 起 %.2f μs\n", us);
        return true;
    }
    std::printf("  FAIL: GPU 仍未见,observed=%u\n", observed);
    return false;
}

// -----------------------------------------------------------------------------
// ★ Test 4 修复:同样把 GPU 端 spin 换成 load_acq_sys_u32
// -----------------------------------------------------------------------------
__global__ void k_pingpong(SharedSlot* dev_slot, int rounds) {
    for (int r = 1; r <= rounds; ++r) {
        // ★ 改动
        while (load_acq_sys_u32(&dev_slot->pingpong_request)
               != static_cast<uint32_t>(r)) {
        }
        __threadfence_system();
        atomicExch_system(&dev_slot->pingpong_response, static_cast<uint32_t>(r));
    }
}

bool test_round_trip(SharedSlot* host_slot, SharedSlot* dev_slot) {
    std::printf("\n[Test 4] Round-trip 延迟(CPU↔GPU ping-pong)\n");
    constexpr int kRounds = 1000;
    host_slot->pingpong_request  = 0;
    host_slot->pingpong_response = 0;

    k_pingpong<<<1, 1>>>(dev_slot, kRounds);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));

    auto t0 = clk::now();
    for (int r = 1; r <= kRounds; ++r) {
        __atomic_store_n(&host_slot->pingpong_request,
                         static_cast<uint32_t>(r), __ATOMIC_RELEASE);
        while (__atomic_load_n(&host_slot->pingpong_response, __ATOMIC_ACQUIRE)
               != static_cast<uint32_t>(r)) {
        #if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
        #endif
        }
    }
    double total_us = us_since(t0);
    CHECK(cudaDeviceSynchronize());
    double avg = total_us / kRounds;
    std::printf("  完成 %d 轮,平均 %.2f μs/round\n", kRounds, avg);
    if (avg < 50.0) {
        std::printf("  PASS\n"); return true;
    }
    std::printf("  WARN: 延迟偏高,可能虚拟化路径\n");
    return true;
}

int main() {
    int dev_count = 0;
    CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count == 0) { std::printf("no CUDA device\n"); return 1; }
    CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device 0: %s, CC %d.%d\n", prop.name, prop.major, prop.minor);
    if (prop.major < 7) {
        std::printf("WARNING: ld.acquire.sys.b32 需要 SM_70+,当前 %d.%d\n",
                    prop.major, prop.minor);
    }

    SharedSlot* host_slot = nullptr;
    CHECK(cudaHostAlloc(&host_slot, sizeof(SharedSlot),
                        cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(host_slot, 0, sizeof(SharedSlot));

    SharedSlot* dev_slot = nullptr;
    CHECK(cudaHostGetDevicePointer(&dev_slot, host_slot, 0));
    std::printf("host VA = %p, dev VA = %p\n", (void*)host_slot, (void*)dev_slot);

    bool all_ok = true;
    all_ok &= test_gpu_to_cpu_simple (host_slot, dev_slot);
    all_ok &= test_producer_consumer (host_slot, dev_slot);
    all_ok &= test_cpu_to_gpu        (host_slot, dev_slot);
    all_ok &= test_round_trip        (host_slot, dev_slot);

    cudaFreeHost(host_slot);
    std::printf("\n========== %s ==========\n", all_ok ? "ALL PASS" : "SOME FAIL");
    return all_ok ? 0 : 1;
}
