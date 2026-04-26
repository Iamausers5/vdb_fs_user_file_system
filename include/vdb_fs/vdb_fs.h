/**
 * @file  vdb_fs.h
 * @brief VDB-FS public umbrella header.
 *
 * Include this single header to access the full VDB-FS API.
 * Internal modules should include specific sub-headers directly.
 */

#ifndef VDB_FS_H
#define VDB_FS_H

#include "vdb_fs/vdb_fs_types.h"

/* ── Forward declarations for upcoming modules ─────────────────────────────── */
namespace vdb {
namespace fs {

// Module 2: GPU memory manager
class GpuMetaRegion;

// Module 3: POSIX-like API
VdbStatus vdb_mount(const char* nvme_addr);
void      vdb_unmount();

fd_t      vdb_open(const char* path, uint32_t flags);
int64_t   vdb_read(fd_t fd, void* gpu_buf, uint64_t size, uint64_t offset);
int64_t   vdb_write(fd_t fd, const void* gpu_buf, uint64_t size, uint64_t offset);
VdbStatus vdb_close(fd_t fd);
VdbStatus vdb_sync(fd_t fd);
VdbStatus vdb_stat(const char* path, /* struct VdbStat* */ void* out);

} // namespace fs
} // namespace vdb

#endif /* VDB_FS_H */
