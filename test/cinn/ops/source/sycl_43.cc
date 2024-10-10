#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_reduce_sum_3_3_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  dout = (float* )(*(void **)(void_args[0]));
  float*  var_4 = (float* )(*(void **)(void_args[1]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space43_fn_reduce_sum_3_3_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      float* var_4__reduce_init = var_4;
      var_4__reduce_init[0] = 0.00000000f;
      for (int32_t reduce_k_0 = 0; reduce_k_0 < 128; reduce_k_0 += 1) {
        for (int32_t reduce_k_1 = 0; reduce_k_1 < 64; reduce_k_1 += 1) {
          for (int32_t reduce_k_2 = 0; reduce_k_2 < 32; reduce_k_2 += 1) {
            var_4[0] = (var_4[0] + dout[((2048 * reduce_k_0) + ((32 * reduce_k_1) + reduce_k_2))]);
          };
        };
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
