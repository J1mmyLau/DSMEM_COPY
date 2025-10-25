#include "cuda_runtime.h"                
#include "cooperative_groups.h"
#include <cuda.h>

namespace cg = cooperative_groups;

template<int cluster_size = 2>
__device__ __forceinline__ void __cluster_dims__(cluster_size, 1, 1) DSM_COPY_NOT_ASYNC(
    const uint32_t size, const uint32_t tid, const uint32_t cluster_block_id,
    uint32_t barrier, const uint32_t src_addr, const uint32_t dst_addr
) {
    cg::cluster_group cluster = cg::this_cluster();
    uint32_t dst_cta = 1-cluster_block_id;
    uint32_t neighbor_dst_addr;
    uint32_t neighbor_dst_bar;
    if(tid == 0) {
        asm volatile (
            "mbarrier.init.shared::cta.b64 [%0], %1;"
            :
            : "r"(barrier), "r"(1)
        );
        asm volatile (
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :
                : "r"(barrier), "r"(size) //expect size of "size" byte for cp.async.bulk
        );
    }
    cluster.sync();
    if (tid == 0) {
        asm volatile (
            "mapa.shared::cluster.u32 %0, %1, %2;\n"
            : "=r"(neighbor_dst_addr) //get shared address of neighbor CTA
            : "r"(dst_addr), "r"(dst_cta) 
        );
        asm volatile (
            "mapa.shared::cluster.u32 %0, %1, %2;\n"
            : "=r"(neighbor_dst_bar)
            : "r"(barrier), "r"(dst_cta)
        );
        asm volatile (
            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
            :
            :"r"(neighbor_dst_addr), "r"(src_addr), "r"(size), "r"(neighbor_dst_bar)
            : "memory"
        );
    }
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(barrier),
        "r"(0)
    );
}