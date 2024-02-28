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

#include "oneapi/dnnl/dnnl.hpp"

#include "glog/logging.h"
#include "paddle/cinn/common/type.h"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

struct gemm_dims_t {
    memory::dim m, n, k;
};

namespace cinn {
namespace runtime {
namespace onednn {

void cinn_gpu_onednn_mul(const std::vector<int>& attrs,
                         cinn_buffer_t* input1,
                         cinn_buffer_t* input2,
                         cinn_buffer_t* output,
                         dnnl::stream dnnl_stream = nullptr);

void cinn_gpu_onednn_gemm(const std::vector<int>& attrs,
                          cinn_buffer_t* lhs,
                          cinn_buffer_t* rhs,
                          cinn_buffer_t* bias,
                          cinn_buffer_t* output,
                          dnnl::stream dnnl_stream = nullptr);

void cinn_call_onednn(void* v_args,
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

}  // namespace onednn
}  // namespace runtime
}  // namespace cinn
