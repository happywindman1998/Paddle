// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <string>
#include <vector>

#include "paddle/cinn/common/type.h"
#include "paddle/cinn/runtime/cinn_runtime.h"

namespace cinn {
namespace runtime {
namespace Sycl {


void cinn_gpu_onednn_matmul(const std::vector<int> &attrs,
                          cinn_buffer_t *lhs,
                          cinn_buffer_t *rhs,
                          cinn_buffer_t *bias,
                          cinn_buffer_t *output,
                          void* queue);

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
                      void* queue);


#ifdef CINN_WITH_ONEDNN
void cinn_gpu_onednn_conv2d(const absl::flat_hash_map<std::string, int>& attr,
                           cinn_buffer_t* x,
                           cinn_buffer_t* w,
                           cinn_buffer_t* y,
                           void* stream = nullptr,
                           common::Layout target = common::Layout::kNCHW);

void cinn_gpu_onednn_conv2d_backward_data(
    const absl::flat_hash_map<std::string, int>& attr,
    cinn_buffer_t* w,
    cinn_buffer_t* dy,
    cinn_buffer_t* dx,
    void* stream = nullptr);

void cinn_gpu_onednn_conv2d_backward_filter(
    const absl::flat_hash_map<std::string, int>& attr,
    cinn_buffer_t* x,
    cinn_buffer_t* dy,
    cinn_buffer_t* dw,
    void* stream = nullptr);

void cinn_gpu_onednn_pool2d(const std::vector<int>& attrs,
                           const std::vector<std::string>& str_attrs,
                           cinn_buffer_t* input,
                           cinn_buffer_t* output,
                           void* stream = nullptr);

void cinn_gpu_onednn_softmax(const std::vector<int>& attrs,
                            cinn_buffer_t* input,
                            cinn_buffer_t* output,
                            void* stream = nullptr);

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

void cinn_call_onednn_softmax_forward(void* v_args,
                                     int num_args,
                                     int mode,
                                     int format,
                                     float alpha,
                                     float beta,
                                     int input_n,
                                     int input_c,
                                     int input_h,
                                     int input_w,
                                     int output_n,
                                     int output_c,
                                     int output_h,
                                     int output_w,
                                     void* stream);

void cinn_call_onednn_softmax_backward(void* v_args,
                                      int num_args,
                                      int mode,
                                      int format,
                                      float alpha,
                                      float beta,
                                      int input_n,
                                      int input_c,
                                      int input_h,
                                      int input_w,
                                      int output_n,
                                      int output_c,
                                      int output_h,
                                      int output_w,
                                      void* stream);

#endif 
// CINN_WITH_ONEDNN


}  // namespace onednn
}  // namespace runtime
}  // namespace cinn
