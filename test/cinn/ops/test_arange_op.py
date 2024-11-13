#!/usr/bin/env python3

# Copyright (c) 2023 CINN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import paddle
from paddle import nn
import unittest
import numpy as np

# 环境变量配置，启用Cinn优化
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_use_cinn'] = '1'
os.environ['FLAGS_cinn_bucket_compile'] = '1'

def apply_to_static(net, use_cinn, input_spec=None):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=input_spec,
        build_strategy=build_strategy,
        full_graph=True,
    )

# 创建一个用于生成arange的网络
class ArangeNet(nn.Layer):
    def __init__(self):
        super().__init__()
            def forward(self, start, end, step, dtype):
        out = paddle.arange(start, end, step, dtype=dtype)
        return out

class TestArange(unittest.TestCase):
    def setUp(self):
        self.prepare_data()

    def prepare_data(self):
        # 设置多组固定的start, end, step, 和dtype作为输入
        self.test_cases = [
            {"start": 0, "end": 10, "step": 1, "dtype": 'int32'},
            {"start": 10, "end": 0, "step": -1, "dtype": 'int32'},
            {"start": 0, "end": 10, "step": 0.5, "dtype": 'float32'},
            {"start": -5, "end": 5, "step": 1, "dtype": 'int64'},
        ]

    def eval(self, use_cinn, start, end, step, dtype):
        # 创建并转换网络为静态图
        net = ArangeNet()
        net = apply_to_static(net, use_cinn)
        net.eval()

        # 执行网络并返回结果
        out = net(start, end, step, dtype)
        return out

    def test_eval(self):
        for case in self.test_cases:
                        with self.subTest(case=case):
                dy_out = self.eval(use_cinn=False, **case)
                cinn_out = self.eval(use_cinn=True, **case)

                np.testing.assert_allclose(
                    cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
                )

if __name__ == '__main__':
    unittest.main()
