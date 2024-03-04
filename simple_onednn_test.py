import numpy as np
from numpy import testing
from cinn import frontend,common
from cinn.common import SYCLTarget

def build_run(target:common.Target):
    print("Model running at ", target.arch)
    # Define the NetBuilder.
    builder = frontend.NetBuilder(name="matmul")

    # Define the input variables of the model
    inputs = {
            "x": np.random.random([10, 1, 128, 64]).astype("float32"),
            "y": np.random.random([10, 12, 64, 128]).astype("float32")
    }
    transpose_x = False
    transpose_y = False

    x = builder.create_input(common.Float(32), inputs["x"].shape, "x")
    y = builder.create_input(common.Float(32), inputs["y"].shape, "y")

    # Build the model using NetBuilder API
    out = builder.matmul(x, y, transpose_x, transpose_y)

    # Specify target and generate the computation
    prog = builder.build()
    passes=[]
    result = prog.build_and_get_output(target, [x, y], [inputs["x"], inputs["y"]], [out], passes, None)
    

SYCL_target = common.SYCLTarget()
#SYCL_target = common.SYCLTarget(arch=common.Target.Arch.AMDGPU)
build_run(SYCL_target)


