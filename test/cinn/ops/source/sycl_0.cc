#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_sort_0_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const int64_t*  x = (int64_t* )(*(void **)(void_args[0]));
  int64_t*  var_0 = (int64_t* )(*(void **)(void_args[1]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space0_fn_sort_0_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      int32_t _var_0_index_temp_buffer [ 128 ];
      int32_t _var_0_index_temp_temp_buffer [ 128 ];
      int32_t* var_0_index = _var_0_index_temp_buffer;
      int32_t* var_0_index_temp = _var_0_index_temp_temp_buffer;
      for (int32_t i = 0; i < 128; i += 1) {
        var_0_index_temp[i] = cinn_sycl_lt_num_int64(x, 128, x[i], 0, 1);
      };
      for (int32_t i = 0; i < 128; i += 1) {
        var_0_index[i] = cinn_sycl_next_smallest_int32(var_0_index_temp, 128, i, 0, 1);
      };
      for (int32_t i = 0; i < 128; i += 1) {
        var_0[i] = x[var_0_index[i]];
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
