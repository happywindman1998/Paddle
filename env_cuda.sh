export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export CPLUS_INCLUDE_PATH=$CUDA_PATH/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_PATH/lib:$LIBRARY_PATH

# export MLU_ERR_PRT_ON=1
# export CNRT_PRINT_INFO=1

# llvm clang & sycl env
export LLVM_PATH=/home/wzy/sycl_workspace/build-cuda-2022-06

export PATH=$LLVM_PATH/bin:$PATH
export CPLUS_INCLUDE_PATH=$LLVM_PATH/include:$CPLUS_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$LLVM_PATH/include/sycl:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$LLVM_PATH/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LLVM_PATH/lib:$LIBRARY_PATH

#oneTBB env
export TBBROOT=/home/wzy/sycl_workspace/build-oneTBB
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$TBBROOT/include
export LD_LIBRARY_PATH=$TBBROOT/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$TBBROOT/lib:$LIBRARY_PATH

# onednn env
export DNNLROOT=/home/wzy/sycl_workspace/oneDNN-cuda-v32
export PATH=$DNNLROOT/bin:$PATH
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$DNNLROOT/include
export LD_LIBRARY_PATH=$DNNLROOT/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$DNNLROOT/lib:$LIBRARY_PATH

export FLAGS_cinn_infer_model_version=1.0
