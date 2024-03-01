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

#include <string>
#include <vector>

#include "paddle/cinn/common/type.h"
#include "paddle/cinn/runtime/cinn_runtime.h"

namespace cinn {
namespace runtime {
namespace onednn {


void cinn_gpu_onednn_mul(const std::vector<int>& attrs,
                         cinn_buffer_t* input1,
                         cinn_buffer_t* input2,
                         cinn_buffer_t* output);

}  // namespace onednn
}  // namespace runtime
}  // namespace cinn
