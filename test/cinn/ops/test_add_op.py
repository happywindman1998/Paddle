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
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_use_cinn'] = '1'
os.environ['FLAGS_cinn_bucket_compile'] = '1'
import paddle
from paddle import nn
import unittest
import numpy as np
def apply_to_static(net, use_cinn, input_spec=None):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=input_spec,
        build_strategy=build_strategy,
        full_graph=True,
    )
class AbsNet(nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        out = paddle.abs(x,y)
        return out
class TestAbs(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()
    def prepare_data(self):
        # 可以仿照之前脚本，设置多个shape case
        self.shape = [4, 4, 4096]
        self.x = paddle.randn(self.shape, dtype="float32")
        self.x.stop_gradient = False # or True
        self.y = paddle.randn(self.shape, dtype="float32")
        self.y.stop_gradient = False
    def eval(self, use_cinn):
        net = AbsNet()
        net = apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x,self.y)
        return out
    def test_eval(self):
        dy_out = self.eval(use_cinn=False)
        cinn_out = self.eval(use_cinn=True)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
        )
if __name__ == '__main__':
    unittest.main()
