#include "cuda_runtime.h"                
#include "cooperative_groups.h"
#include "dsm.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <stdio.h>
#include <time.h>
#include <cuda/barrier>

namespace cg = cooperative_groups;
using barrier = cuda::barrier<cuda::thread_scope_block>;


struct random_functor {
    unsigned int seed;
    random_functor(unsigned int s) : seed(s) {}

    __device__
    float operator()(unsigned int i) {
        thrust::minstd_rand rng(seed + i);
        thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
        rng.discard(1); 
        return dist(rng);
    }
};

template<int cluster_size = 2, size_t NUM_FLOATS_PER_CTA, size_t chunk_size>
__global__ void __cluster_dims__(cluster_size, 1, 1) dsm_cluster_kernel(
    float* data, 
    float* output, 
    long long* timing_output
) {
    cg::cluster_group cluster = cg::this_cluster();
    uint32_t cluster_block_id = cluster.block_rank();
    uint32_t tid = threadIdx.x;
    uint32_t dst_cta = 1 - cluster_block_id;

    constexpr size_t TOTAL_FLOATS = NUM_FLOATS_PER_CTA * cluster_size;
    
    __shared__ alignas(16) float shared_data[TOTAL_FLOATS];
    __shared__ alignas(8) uint64_t barrier;
    __shared__ long long start_time;

    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    
    // cluster.sync(); 

    static_assert(chunk_size % 2 == 0, "chunk_size 必须是 2 的倍数");
    constexpr size_t load_size = chunk_size / 2;
    static_assert(NUM_FLOATS_PER_CTA % load_size == 0, "NUM_FLOATS_PER_CTA 必须是 load_size 的倍数");
    
    constexpr uint32_t stages = NUM_FLOATS_PER_CTA / load_size; 
    
    if (tid == 0) {
        asm volatile (
            "mbarrier.inval.shared::cta.b64 [%0];" 
            : : "r"(bar_ptr)
        );
        // 现在我们可以安全地在阶段 0 初始化
        asm volatile (
            "mbarrier.init.shared::cta.b64 [%0], %1;"
            : : "r"(bar_ptr), "r"(1)
        );
        start_time = clock64();
    }
    __syncthreads();
    if (tid == 0) {
        start_time = clock64();
    }
    constexpr uint32_t load_bytes = load_size * sizeof(float);
    #pragma unroll
    for(size_t stage = 0; stage < stages; ++stage) {
        
        for(size_t i = threadIdx.x; i < load_size ; i += blockDim.x) {
            shared_data[i + stage * chunk_size + cluster_block_id * load_size] = 
                data[i + stage * chunk_size + cluster_block_id * load_size];
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t src_addr = (uint32_t)(shared_data + stage * chunk_size + cluster_block_id * load_size);
            uint32_t dst_addr_local = src_addr;
            
            uint32_t neighbor_dst_addr;
            uint32_t neighbor_dst_bar;

            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_addr)
                : "r"(dst_addr_local), "r"(dst_cta) 
            );
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_bar)
                : "r"(bar_ptr), "r"(dst_cta)
            );
            
            asm volatile (
                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                :
                :"r"(neighbor_dst_addr), "r"(src_addr), "r"(load_bytes), "r"(neighbor_dst_bar)
                : "memory"
            );
        }

    }

    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], 1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(bar_ptr)
    );
    if (tid == 0) {
        long long end_time = clock64(); 
        timing_output[cluster_block_id] = end_time - start_time;
    }
    // cluster.sync();
    float * output_local = output + cluster_block_id * TOTAL_FLOATS;
    for(size_t i = threadIdx.x; i < TOTAL_FLOATS; i += blockDim.x) {
        output_local[i] = shared_data[i];
    }
}

template<size_t NUM_FLOATS_PER_CTA>
__global__ void global_load_kernel(
    float* data, 
    float* output, 
    long long* timing_output
) {
    uint32_t cluster_block_id = blockIdx.x;
    uint32_t tid = threadIdx.x;

    const size_t TOTAL_FLOATS = NUM_FLOATS_PER_CTA * 2;
    
    __shared__ alignas(16) float shared_data[TOTAL_FLOATS];
    __shared__ long long start_time;


    if (tid == 0) {
        start_time = clock64();
    }
    for(size_t i = threadIdx.x; i < NUM_FLOATS_PER_CTA * 2  ; i += blockDim.x) {
        shared_data[i] = 
            data[i];
    }
    if (tid == 0) {
        long long end_time = clock64(); 
        timing_output[cluster_block_id] = end_time - start_time;
    }

    float * output_local = output + cluster_block_id * TOTAL_FLOATS;
    
    for(size_t i = threadIdx.x; i < TOTAL_FLOATS; i += blockDim.x) {
        output_local[i] = shared_data[i];
    }
}


int main() {
    constexpr size_t NUM_FLOATS_PER_CTA = 4096; 
    constexpr size_t CHUNK_SIZE = 4096;
    constexpr size_t TOTAL_FLOATS = NUM_FLOATS_PER_CTA * 2; 
    constexpr size_t N = TOTAL_FLOATS;

    dim3 numBlocks(2, 1, 1); 
    dim3 threadsPerBlock(128, 1, 1);
    
    int clockRateKHz = 0;
    cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, 0);
    double clockRateHz = clockRateKHz * 1000.0;
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    printf("Host: 设备名称: %s\n", props.name);
    printf("Host: 每个SM最大时钟频率: %.2f GHz\n", clockRateHz / 1e9);
    printf("Host: 每个块的最大共享内存: %zu 字节\n", props.sharedMemPerBlock);
    
    size_t required_shmem = TOTAL_FLOATS * sizeof(float) + sizeof(uint64_t) + sizeof(long long);
    printf("Host: 内核请求的共享内存: %zu 字节\n", required_shmem);
    if (required_shmem > props.sharedMemPerBlock) {
        printf("错误：请求的共享内存超过了设备限制！\n");
        return -1;
    }

    printf("Host: 准备生成 %zu 个随机浮点数 (%.2f KB)\n", N, (N * sizeof(float)) / 1024.0);

    // --- 2. 分配缓冲区 ---
    thrust::device_vector<float> d_vec(N);
    thrust::device_vector<float> output_vec_dsm(N * 2); 
    thrust::device_vector<long long> d_timing_dsm(numBlocks.x); 
    thrust::device_vector<float> output_vec_global(N * 2); 
    thrust::device_vector<long long> d_timing_global(numBlocks.x); 

    thrust::transform(
        thrust::device,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator(N),
        d_vec.begin(),
        random_functor(static_cast<unsigned int>(time(NULL)))
    );

    // --- 3. 配置内核启动 ---
    cudaLaunchConfig_t config = {0};
    config.gridDim = numBlocks;
    config.blockDim = threadsPerBlock;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 2;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;

    float* d_data = thrust::raw_pointer_cast(d_vec.data());
    float* d_output_dsm = thrust::raw_pointer_cast(output_vec_dsm.data());
    long long* d_timing_dsm_data = thrust::raw_pointer_cast(d_timing_dsm.data());
    float* d_output_global = thrust::raw_pointer_cast(output_vec_global.data());
    long long* d_timing_global_data = thrust::raw_pointer_cast(d_timing_global.data());

    printf("Host: 启动内核 (每个块获取 %zu 字节)...\n", TOTAL_FLOATS * sizeof(float));
    
    // --- 4. 运行实验 1: DSM 复制 ---
    printf("\nHost: 启动内核 1 (DSM: 分块加载 + DSM 交换)...\n");
    cudaError_t status_dsm = cudaLaunchKernelEx(
        &config, 
        dsm_cluster_kernel<2, NUM_FLOATS_PER_CTA, CHUNK_SIZE>,
        d_data, 
        d_output_dsm, 
        d_timing_dsm_data 
    );

    // --- 5. 运行实验 2: Global Load (全局加载) ---
    printf("Host: 启动内核 2 (Global: 加载本地 + 加载远程)...\n");
    global_load_kernel<NUM_FLOATS_PER_CTA><<<numBlocks, threadsPerBlock>>>( 
        d_data, 
        d_output_global, 
        d_timing_global_data 
    );

    if (status_dsm != cudaSuccess) {
        printf("内核启动失败! DSM: %s\n", 
            cudaGetErrorString(status_dsm));
        return -1;
    }

    cudaDeviceSynchronize();
    printf("Host: 所有内核执行完毕。\n");

    // --- 6. 报告时间 ---
    thrust::host_vector<long long> h_timing_dsm = d_timing_dsm;
    thrust::host_vector<long long> h_timing_global = d_timing_global;
    
    double time_sec_dsm = (double)h_timing_dsm[0] / clockRateHz;
    double time_sec_global = (double)h_timing_global[0] / clockRateHz;

    double data_GB = (double)(TOTAL_FLOATS * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    double bw_dsm = data_GB / time_sec_dsm;
    double bw_global = data_GB / time_sec_global;

    printf("----------------------------------------\n");
    printf("Host: 测量结果 (基于 CTA 0):\n");
    
    printf("\n  [DSM (分块加载 + 交换)]\n"); // 描述已更新
    printf("    总周期: %lld，%lld\n", h_timing_dsm[0], h_timing_dsm[1]);
    printf("    总时间: %f 毫秒\n", time_sec_dsm * 1000.0);
    printf("    有效带宽: %.2f GB/s\n", bw_dsm);

    printf("\n  [Global (加载本地 + 远程)]\n"); // 描述已更新
    printf("    总周期: %lld，%lld\n", h_timing_global[0], h_timing_global[1]);
    printf("    总时间: %f 毫秒\n", time_sec_global * 1000.0);
    printf("    有效带宽: %.2f GB/s\n", bw_global);
    
    printf("\n  [对比]\n");
    printf("    DSM 方案 (分块加载+交换) 速度是 Global 方案 (加载*2) 的 %.2f 倍\n", 
           time_sec_global / time_sec_dsm);
    printf("----------------------------------------\n");

    // --- 7. 验证数据 ---
    thrust::host_vector<float> h_input = d_vec;
    thrust::host_vector<float> h_output_dsm = output_vec_dsm;
    thrust::host_vector<float> h_output_global = output_vec_global;
    printf("Host: 验证数据...\n");

    bool verified_dsm = true;
    bool verified_global = true;

    for(size_t i = 0; i < N; ++i) { // N == TOTAL_FLOATS
        // 检查 CTA 0 的输出
        if(h_output_dsm[i] != h_input[i]) 
        {
            verified_dsm = false;
            printf("id: %ld,  expect: %f actuall: %f\n",i , h_input[i], h_output_dsm[i]);
        }
            
        if(h_output_global[i] != h_input[i]) verified_global = false;

        // 检查 CTA 1 的输出
        if(h_output_dsm[i + N] != h_input[i]) 
        {
            verified_dsm = false;
            printf("id: %ld,  expect: %f actuall: %f\n",i + N , h_input[i], h_output_dsm[i + N]);
            return -1;
        }
        if(h_output_global[i + N] != h_input[i]) verified_global = false;
    }
    
    if (!verified_dsm) 
        printf("  [DSM] 数据验证失败!\n");
    if (!verified_global) 
        printf("  [Global] 数据验证失败!\n");
    
    if (verified_dsm && verified_global) {
        printf("  数据验证成功！两个内核都产生了正确的结果。\n");
    }

    printf("Host: 完成。\n");
    return 0;
}