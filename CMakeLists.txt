cmake_minimum_required(VERSION 3.18...3.20 FATAL_ERROR)

project(DSM_COPY LANGUAGES CXX CUDA)

# 设置你希望的 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA 标准（可选）
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 120)  # 例如 Ampere（80）和 Hopper（90）

# 可选：如果需要 separable 编译（设备代码可以链接在库里）：
# set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# 源文件
set(SRC
    src/main.cu
    src/dsm.cuh
    # … 你其他 .cpp/.cu/.cuh 文件
)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SRC})

# 可选：如果你的 .cu 文件中需要可分离编译 device 函数等
set_target_properties(${PROJECT_NAME}
    PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
)

# 指定 include 路径
# target_include_directories(${PROJECT_NAME}
#     PRIVATE
#       ${CMAKE_CURRENT_SOURCE_DIR}/include
# )

# 链接库，如果需要 （例如 CUDA 库、cuBLAS、cuFFT 等）
# target_link_libraries(${PROJECT_NAME}
#    PRIVATE
#      cuda
# )

# 可选：给 CUDA 编译器额外的编译标志
# target_compile_options(${PROJECT_NAME}
#     PRIVATE
#       $<$<COMPILE_LANGUAGE:CUDA>:
#          -lineinfo   # 保存源代码行信息（有助于调试）
#          -G          # 在 Debug 模式启用设备调试（仅用于 Debug）
#       >
# )

# 可选：不同构建类型设置
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     message(STATUS "Configuring for Debug build")
#     target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG)
# else()
#     message(STATUS "Configuring for Release build")
#     target_compile_definitions(${PROJECT_NAME} PRIVATE NDEBUG)
#     target_compile_options(${PROJECT_NAME}
#         PRIVATE
#           $<$<COMPILE_LANGUAGE:CUDA>:-O3>
#     )
# endif()