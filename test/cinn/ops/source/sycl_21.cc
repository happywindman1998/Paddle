#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_argmin_0_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  x = (float* )(*(void **)(void_args[0]));
  int32_t*  var_0 = (int32_t* )(*(void **)(void_args[1]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space21_fn_argmin_0_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      int32_t _var_0_index_temp_buffer [ 1024 ];
      int32_t _var_0_index_temp_temp_buffer [ 1024 ];
      int32_t* var_0_index = _var_0_index_temp_buffer;
      int32_t* var_0_index_temp = _var_0_index_temp_temp_buffer;
      for (int32_t i = 0; i < 16; i += 1) {
        for (int32_t j = 0; j < 8; j += 1) {
          for (int32_t k = 0; k < 4; k += 1) {
            for (int32_t a = 0; a < 2; a += 1) {
              var_0_index_temp[((64 * i) + ((8 * j) + ((2 * k) + a)))] = cinn_sycl_lt_num_fp32(x, 4, x[((64 * i) + ((8 * j) + ((2 * k) + a)))], ((64 * i) + ((8 * j) + a)), 2);
            };
          };
        };
      };
      for (int32_t i = 0; i < 16; i += 1) {
        for (int32_t j = 0; j < 8; j += 1) {
          for (int32_t k = 0; k < 4; k += 1) {
            for (int32_t a = 0; a < 2; a += 1) {
              var_0_index[((64 * i) + ((8 * j) + ((2 * k) + a)))] = cinn_sycl_next_smallest_int32(var_0_index_temp, 4, k, ((64 * i) + ((8 * j) + a)), 2);
            };
          };
        };
      };
      for (int32_t i = 0; i < 16; i += 1) {
        for (int32_t j = 0; j < 8; j += 1) {
          for (int32_t a = 0; a < 2; a += 1) {
            var_0[((16 * i) + ((2 * j) + a))] = var_0_index[((64 * i) + ((8 * j) + a))];
          };
        };
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
