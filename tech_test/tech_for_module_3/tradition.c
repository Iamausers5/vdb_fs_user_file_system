/**
 * traditional_libaio_test.c
 *
 * 传统方式：使用 Linux libaio 异步 I/O 接口对 NVMe SSD 进行随机读写性能测试
 *
 * 编译:
 *   gcc -O2 -o traditional_libaio_test traditional_libaio_test.c -laio -lpthread
 *
 * 运行示例（需要 root 或对裸设备有读写权限）:
 *   sudo ./traditional_libaio_test --dev /dev/nvme0n1 --rw randread  --bs 4096 --iodepth 32 --time 10
 *   sudo ./traditional_libaio_test --dev /dev/nvme0n1 --rw randwrite --bs 4096 --iodepth 32 --time 10
 *
 * 观察指标:
 *   - IOPS        : 每秒完成的 I/O 次数（受内核路径开销限制，通常远低于 SPDK）
 *   - CPU sys%    : 通过 /proc/self/stat 采样，体现内核态 CPU 占用（通常较高）
 *   - Latency avg : 平均 I/O 延迟（us）
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <pthread.h>
#include <signal.h>
#include <libaio.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <linux/fs.h>

/* ───────────────────────────── 默认参数 ───────────────────────────── */
#define DEFAULT_BS       4096          /* 块大小 (bytes) */
#define DEFAULT_IODEPTH  32            /* 队列深度       */
#define DEFAULT_TIME_SEC 10            /* 测试时长 (s)   */
#define MAX_IODEPTH      256

/* ───────────────────────────── 全局控制 ───────────────────────────── */
static volatile int g_running = 1;

/* ───────────────────────────── 统计结构 ───────────────────────────── */
typedef struct {
    uint64_t ios_completed;
    uint64_t total_latency_us;   /* 所有 I/O 延迟之和 */
    uint64_t min_latency_us;
    uint64_t max_latency_us;
} stats_t;

/* ───────────────────────────── 工具函数 ───────────────────────────── */

/* 读取 /proc/self/stat 的 utime / stime（单位: clock ticks） */
static void get_proc_times(uint64_t *utime, uint64_t *stime)
{
    FILE *f = fopen("/proc/self/stat", "r");
    if (!f) { *utime = *stime = 0; return; }
    /* 跳过前 13 个字段 */
    unsigned long ut = 0, st = 0;
    fscanf(f,
        "%*d %*s %*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %lu %lu",
        &ut, &st);
    fclose(f);
    *utime = ut;
    *stime = st;
}

/* 获取单调时钟（微秒） */
static inline uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}

/* 随机 64 位偏移，按块大小对齐，限制在设备范围内 */
static inline uint64_t rand_offset(uint64_t dev_size, uint32_t bs)
{
    uint64_t max_blocks = dev_size / bs;
    return (((uint64_t)rand() << 32) | rand()) % max_blocks * bs;
}

static void signal_handler(int sig)
{
    (void)sig;
    g_running = 0;
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "用法: %s [选项]\n"
        "  --dev     <设备路径>   裸块设备，如 /dev/nvme0n1   (必填)\n"
        "  --rw      randread|randwrite  I/O 类型  (默认: randread)\n"
        "  --bs      <字节数>     块大小            (默认: 4096)\n"
        "  --iodepth <深度>       异步队列深度      (默认: 32, 最大: %d)\n"
        "  --time    <秒>         测试时长          (默认: 10)\n",
        prog, MAX_IODEPTH);
}

/* ─────────────────────────────── main ─────────────────────────────── */
int main(int argc, char *argv[])
{
    const char *dev_path = NULL;
    int         is_write  = 0;
    uint32_t    bs        = DEFAULT_BS;
    int         iodepth   = DEFAULT_IODEPTH;
    int         test_sec  = DEFAULT_TIME_SEC;

    /* ── 解析命令行 ── */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dev")     && i+1 < argc) dev_path  = argv[++i];
        else if (!strcmp(argv[i], "--rw") && i+1 < argc) {
            i++;
            if (!strcmp(argv[i], "randwrite")) is_write = 1;
        }
        else if (!strcmp(argv[i], "--bs")      && i+1 < argc) bs       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iodepth") && i+1 < argc) iodepth  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--time")    && i+1 < argc) test_sec = atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (!dev_path) { usage(argv[0]); return 1; }
    if (iodepth > MAX_IODEPTH) iodepth = MAX_IODEPTH;

    /* ── 打开设备 ── */
    int flags = O_RDWR | O_DIRECT;   /* O_DIRECT：绕过 page cache */
    if (!is_write) flags = O_RDONLY | O_DIRECT;
    int fd = open(dev_path, flags);
    if (fd < 0) { perror("open"); return 1; }

    /* ── 获取设备大小 ── */
    uint64_t dev_size = 0;
    if (ioctl(fd, BLKGETSIZE64, &dev_size) < 0) {
        perror("ioctl BLKGETSIZE64"); close(fd); return 1;
    }
    printf("设备: %s  大小: %.2f GiB\n",
           dev_path, (double)dev_size / (1024.0*1024.0*1024.0));
    printf("I/O 类型: %s  块大小: %u B  队列深度: %d  时长: %d s\n\n",
           is_write ? "randwrite" : "randread", bs, iodepth, test_sec);

    /* ── 分配 I/O 缓冲区（需 512 字节对齐） ── */
    uint8_t **bufs = malloc(iodepth * sizeof(uint8_t *));
    for (int i = 0; i < iodepth; i++) {
        if (posix_memalign((void **)&bufs[i], 512, bs) != 0) {
            perror("posix_memalign"); return 1;
        }
        memset(bufs[i], 0xAB, bs);
    }

    /* ── 初始化 libaio 上下文 ── */
    io_context_t ctx = 0;
    if (io_setup(iodepth, &ctx) < 0) {
        perror("io_setup"); return 1;
    }

    struct iocb  **iocbs      = malloc(iodepth * sizeof(struct iocb *));
    struct iocb   *iocb_pool  = calloc(iodepth, sizeof(struct iocb));
    struct io_event *events   = malloc(iodepth * sizeof(struct io_event));
    uint64_t      *submit_ts  = malloc(iodepth * sizeof(uint64_t)); /* 提交时间戳 */
    int           *in_flight  = calloc(iodepth, sizeof(int));

    srand((unsigned)time(NULL));
    signal(SIGINT,  signal_handler);
    signal(SIGALRM, signal_handler);
    alarm(test_sec);

    /* ── 统计初始 CPU 时间 ── */
    uint64_t t_utime0, t_stime0;
    get_proc_times(&t_utime0, &t_stime0);
    uint64_t wall_start = now_us();

    stats_t st = { 0, 0, UINT64_MAX, 0 };

    /* ── 预填充：提交首批 I/O ── */
    int submitted = 0;
    for (int i = 0; i < iodepth; i++) {
        struct iocb *cb = &iocb_pool[i];
        uint64_t off = rand_offset(dev_size, bs);
        if (is_write)
            io_prep_pwrite(cb, fd, bufs[i], bs, (long long)off);
        else
            io_prep_pread (cb, fd, bufs[i], bs, (long long)off);
        iocbs[0]     = cb;
        submit_ts[i] = now_us();
        in_flight[i] = 1;
        if (io_submit(ctx, 1, iocbs) != 1) {
            perror("io_submit"); break;
        }
        submitted++;
    }

    /* ── 主循环：收割完成事件，立即重新提交 ── */
    struct timespec timeout = { .tv_sec = 0, .tv_nsec = 100000 }; /* 100us */
    while (g_running) {
        int nr = io_getevents(ctx, 1, iodepth, events, &timeout);
        if (nr < 0) {
            if (errno == EINTR) break;
            perror("io_getevents"); break;
        }
        for (int i = 0; i < nr; i++) {
            struct iocb *cb  = (struct iocb *)events[i].obj;
            int          idx = (int)(cb - iocb_pool);
            uint64_t     lat = now_us() - submit_ts[idx];

            st.ios_completed++;
            st.total_latency_us += lat;
            if (lat < st.min_latency_us) st.min_latency_us = lat;
            if (lat > st.max_latency_us) st.max_latency_us = lat;

            if (!g_running) { in_flight[idx] = 0; continue; }

            /* 立即重新提交同一个 slot */
            uint64_t off = rand_offset(dev_size, bs);
            if (is_write)
                io_prep_pwrite(cb, fd, bufs[idx], bs, (long long)off);
            else
                io_prep_pread (cb, fd, bufs[idx], bs, (long long)off);
            iocbs[0]        = cb;
            submit_ts[idx]  = now_us();
            if (io_submit(ctx, 1, iocbs) != 1) {
                in_flight[idx] = 0;   /* 失败则不再重提 */
            }
        }
    }

    /* ── 等待剩余 in-flight I/O 排空 ── */
    {
        struct timespec drain = { .tv_sec = 5, .tv_nsec = 0 };
        io_getevents(ctx, 0, iodepth, events, &drain);
    }

    /* ── 采集最终 CPU 时间 ── */
    uint64_t t_utime1, t_stime1;
    get_proc_times(&t_utime1, &t_stime1);
    uint64_t wall_end   = now_us();
    double   wall_sec   = (wall_end - wall_start) / 1e6;
    long     clk_tck    = sysconf(_SC_CLK_TCK);

    double user_sec = (double)(t_utime1 - t_utime0) / clk_tck;
    double sys_sec  = (double)(t_stime1 - t_stime0) / clk_tck;
    double cpu_pct  = (user_sec + sys_sec) / wall_sec * 100.0;
    double sys_pct  = sys_sec / wall_sec * 100.0;

    /* ── 打印结果 ── */
    double iops    = st.ios_completed / wall_sec;
    double bw_mbps = iops * bs / (1024.0 * 1024.0);
    double avg_lat = st.ios_completed
                     ? (double)st.total_latency_us / st.ios_completed
                     : 0;

    printf("════════════ 测试结果 (传统 libaio) ════════════\n");
    printf("  测试时长      : %.2f s\n",    wall_sec);
    printf("  完成 I/O 数   : %lu\n",       (unsigned long)st.ios_completed);
    printf("  IOPS          : %.0f\n",      iops);
    printf("  吞吐量        : %.2f MiB/s\n", bw_mbps);
    printf("  延迟 avg/min/max: %.1f / %.1f / %.1f us\n",
           avg_lat,
           st.min_latency_us == UINT64_MAX ? 0.0 : (double)st.min_latency_us,
           (double)st.max_latency_us);
    printf("  CPU user%%     : %.1f%%\n",   user_sec / wall_sec * 100.0);
    printf("  CPU sys%%      : %.1f%%  ← 内核态开销（通常较高）\n", sys_pct);
    printf("  CPU total%%    : %.1f%%\n",   cpu_pct);
    printf("════════════════════════════════════════════════\n");

    /* ── 清理 ── */
    io_destroy(ctx);
    close(fd);
    for (int i = 0; i < iodepth; i++) free(bufs[i]);
    free(bufs); free(iocbs); free(iocb_pool);
    free(events); free(submit_ts); free(in_flight);
    return 0;
}
