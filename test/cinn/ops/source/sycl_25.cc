#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_identity_1_identity_2_5_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const int32_t*  dout = (int32_t* )(*(void **)(void_args[0]));
  int32_t*  var_4 = (int32_t* )(*(void **)(void_args[1]));
  int32_t*  var_3 = (int32_t* )(*(void **)(void_args[2]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space25_fn_identity_1_identity_2_5_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      for (int32_t flat_i = 0; flat_i < 1024; flat_i += 1) {
        var_3[flat_i] = dout[flat_i];
        var_4[flat_i] = dout[flat_i];
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
