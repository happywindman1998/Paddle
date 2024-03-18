export CC=gcc
export CXX=g++

export DPCPP_ROOT=/CINN-SYCL/llvm-sycl-nightly-20220501
export ONEDNN_ROOT=/CINN-SYCL/oneDNN

mkdir build-docker
cd build-docker
cmake -DCINN_ONLY=ON -DWITH_CINN=ON -DWITH_GPU=OFF -DCINN_WITH_SYCL=ON  -DCINN_WITH_ONEDNN=ON -DWITH_DLNNE=ON -DWITH_TESTING=OFF \
      -DWITH_MKL=OFF -DPYTHON_EXECUTABLE=python3.8 -DPY_VERSION=3.8 -G Ninja ..
ninja -j 16
