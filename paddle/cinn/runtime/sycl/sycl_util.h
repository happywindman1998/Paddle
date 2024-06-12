// Copyright (c) 2024 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <mutex>  // NOLINT
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#ifdef CINN_WITH_ONEDNN
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif
/*
#ifdef CINN_WITH_CNNL
#include <cnrt.h>
#include <cnnl.h>
#endif
*/

namespace cinn {
namespace runtime {
namespace sycl {

/**
 * Call a SYCL compiled kernel.
 *
 * @param kernel_fn the func pointer.
 * @param args an array of cinn_pod_value_ts(consists of scalars and buffers).
 */
void cinn_call_sycl_kernel(void* kernel_fn,
                           void* v_args,
                           int num_args,
                           int grid_x,
                           int grid_y,
                           int grid_z,
                           int block_x,
                           int block_y,
                           int block_z);

void cinn_call_sycl_memset(void* v_args,
                           int num_args,
                           int value,
                           size_t count);

void cinn_call_sycl_memcpy(void *v_args,
                           int num_args,
                           size_t count);

#ifdef CINN_WITH_ONEDNN
// 定义映射枚举到字符串的宏
#define DNNL_STATUS_TO_STRING(status) \
    ((status) == dnnl_success           ? "dnnl_success" : \
    (status) == dnnl_out_of_memory      ? "dnnl_out_of_memory" : \
    (status) == dnnl_invalid_arguments  ? "dnnl_invalid_arguments" : \
    (status) == dnnl_unimplemented      ? "dnnl_unimplemented" : \
    (status) == dnnl_last_impl_reached  ? "dnnl_last_impl_reached" : \
    (status) == dnnl_runtime_error      ? "dnnl_runtime_error" : \
    (status) == dnnl_not_required       ? "dnnl_not_required" : \
    (status) == dnnl_invalid_graph      ? "dnnl_invalid_graph" : \
    (status) == dnnl_invalid_graph_op   ? "dnnl_invalid_graph_op" : \
    (status) == dnnl_invalid_shape      ? "dnnl_invalid_shape" : \
    (status) == dnnl_invalid_data_type  ? "dnnl_invalid_data_type" : "Unknown status")

#define CHECK_DNNL_STATUS(status)\
  do { \
      if ((status) != dnnl_success) { \
          printf("DNNL Error: %s\n", DNNL_STATUS_TO_STRING(status)); \
      } \
  } while (0)

void cinn_call_onednn_gaussian_random(void* v_args,
                               int num_args,
                               float mean,
                               float std,
                               int seed,
                               void* stream);

void cinn_call_onednn_uniform_random(void* v_args,
                              int num_args,
                              float min,
                              float max,
                              int seed,
                              void* stream);

void cinn_call_onednn_randint(void* v_args,
                       int num_args,
                       int seed,
                       void* stream);

void cinn_call_onednn_cholesky(void* v_args,
                              int num_args,
                              int batch_size,
                              int m,
                              bool upper,
                              void* stream);

void cinn_call_onednn_triangular_solve(void* v_args,
                                      int num_args,
                                      int batch_size,
                                      int m,
                                      int k,
                                      bool left_side,
                                      bool upper,
                                      bool transpose_a,
                                      bool unit_diagonal,
                                      void* stream);

void cinn_call_onednn_matmul(void* v_args,
                      int num_args,
                      bool trans_a,
                      bool trans_b,
                      bool trans_o,
                      float alpha,
                      float beta,
                      int a1,
                      int a2,
                      int a3,
                      int a4,
                      int b1,
                      int b2,
                      int b3,
                      int b4,
                      void* stream);

void cinn_call_onednn_conv2d_forward(void* v_args,
                                    int num_args,
                                    int format,
                                    float alpha,
                                    float beta,
                                    int input_n,
                                    int input_c,
                                    int input_h,
                                    int input_w,
                                    int filter_n,
                                    int filter_c,
                                    int filter_h,
                                    int filter_w,
                                    int pad_h,
                                    int pad_w,
                                    int stride_h,
                                    int stride_w,
                                    int dilation_h,
                                    int dilation_w,
                                    int groups,
                                    int output_n,
                                    int output_c,
                                    int output_h,
                                    int output_w,
                                    void* stream);

void cinn_call_onednn_conv2d_backward_data(void* v_args,
                                          int num_args,
                                          int format,
                                          float alpha,
                                          float beta,
                                          int input_n,
                                          int input_c,
                                          int input_h,
                                          int input_w,
                                          int filter_n,
                                          int filter_c,
                                          int filter_h,
                                          int filter_w,
                                          int pad_h,
                                          int pad_w,
                                          int stride_h,
                                          int stride_w,
                                          int dilation_h,
                                          int dilation_w,
                                          int groups,
                                          int output_n,
                                          int output_c,
                                          int output_h,
                                          int output_w,
                                          void* stream);

void cinn_call_onednn_conv2d_backward_filter(void* v_args,
                                            int num_args,
                                            int format,
                                            float alpha,
                                            float beta,
                                            int input_n,
                                            int input_c,
                                            int input_h,
                                            int input_w,
                                            int filter_n,
                                            int filter_c,
                                            int filter_h,
                                            int filter_w,
                                            int pad_h,
                                            int pad_w,
                                            int stride_h,
                                            int stride_w,
                                            int dilation_h,
                                            int dilation_w,
                                            int groups,
                                            int output_n,
                                            int output_c,
                                            int output_h,
                                            int output_w,
                                            void* stream);

void cinn_call_onednn_pool2d_forward(void* v_args,
                                    int num_args,
                                    int mode,
                                    int format,
                                    float alpha,
                                    float beta,
                                    int input_n,
                                    int input_c,
                                    int input_h,
                                    int input_w,
                                    int kernel_h,
                                    int kernel_w,
                                    int pad_h,
                                    int pad_w,
                                    int stride_h,
                                    int stride_w,
                                    int output_n,
                                    int output_c,
                                    int output_h,
                                    int output_w,
                                    void* stream);

void cinn_call_onednn_pool2d_backward(void* v_args,
                                     int num_args,
                                     int mode,
                                     int format,
                                     float alpha,
                                     float beta,
                                     int input_n,
                                     int input_c,
                                     int input_h,
                                     int input_w,
                                     int kernel_h,
                                     int kernel_w,
                                     int pad_h,
                                     int pad_w,
                                     int stride_h,
                                     int stride_w,
                                     int output_n,
                                     int output_c,
                                     int output_h,
                                     int output_w,
                                     void* stream);
#endif // CINN_WITH_ONEDNN

}  // namespace sycl
}  // namespace runtime
}  // namespace cinn
