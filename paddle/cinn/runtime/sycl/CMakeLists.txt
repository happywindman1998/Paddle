core_gather_headers()

gather_srcs(cinnapi_src SRCS
        sycl_module.cc
        sycl_intrinsics.cc
        sycl_intrinsics_reduce.cc
        sycl_backend_api.cc
        onednn_util.cc
)

cinn_sycl_library(sycl_runtime SRCS cinn_sycl_runtime_source.h)
