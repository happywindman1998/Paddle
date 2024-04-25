import numpy as np
from numpy import testing
from cinn import frontend
from cinn import common
from cinn.common import SYCLTarget




def build_run(target:common.Target):
    print("Model running at ", target.arch)
    # Define the NetBuilder.
    builder = frontend.NetBuilder(name="matmul")

    # Define the input variables of the model
    inputs = {
            "x": np.random.random([1024, 1024]).astype("float32"),
            "y": np.random.random([1024, 1024]).astype("float32")
    }
    transpose_x = False
    transpose_y = False

    x = builder.create_input(common.Float(32), inputs["x"].shape, "x")
    y = builder.create_input(common.Float(32), inputs["y"].shape, "y")
    
    A = inputs["x"]
    B = inputs["y"]
    np_data = A @ B

    # Build the model using NetBuilder API
    out = builder.matmul(x, y, transpose_x, transpose_y)

    # Specify target and generate the computation
    prog = builder.build()
    passes=[]
    result = prog.build_and_get_output(target, [x, y], [inputs["x"], inputs["y"]], [out], passes, None)
    #print(result[0].flatten()) 
    cinn_data = []
    for res in result:
        cinn_data.append(res.numpy(target))
    testing.assert_almost_equal(cinn_data[0], np_data, decimal=3)
    #for i in range(0, 100):
    #    print(outs_and_grads[0][i])


SYCL_target = common.SYCLTarget()
#SYCL_target = common.SYCLTarget(arch=common.Target.Arch.AMDGPU)
build_run(SYCL_target)


