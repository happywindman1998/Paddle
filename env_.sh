export NEUWARE_HOME="/usr/local/neuware"
export PATH="${NEUWARE_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="${NEUWARE_HOME}/lib64:${LIBRARY_PATH}"

# export MLU_ERR_PRT_ON=1
# export CNRT_PRINT_INFO=1

# llvm clang & sycl env
# export LLVM_PATH=/home/pangyunfei/llvm_install/debug
export LLVM_PATH=/home/wzy/repos/zkjh-llvm-mlu/build

export PATH=$LLVM_PATH/bin:$PATH
export CPLUS_INCLUDE_PATH=$LLVM_PATH/include:$CPLUS_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$LLVM_PATH/include/sycl:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$LLVM_PATH/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LLVM_PATH/lib:$LIBRARY_PATH

#oneTBB env
export TBBROOT=/home/wzy/build-oneTBB
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$TBBROOT/include
export LD_LIBRARY_PATH=$TBBROOT/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$TBBROOT/lib:$LIBRARY_PATH

# onednn env
export DNNLROOT=/home/wzy/oneDNN-mlu
export PATH=$DNNLROOT/bin:$PATH
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$DNNLROOT/include
export LD_LIBRARY_PATH=$DNNLROOT/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$DNNLROOT/lib:$LIBRARY_PATH
