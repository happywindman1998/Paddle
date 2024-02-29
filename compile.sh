export CC=gcc
export CXX=g++

mkdir build
cd build
cmake -DCINN_ONLY=ON -DWITH_CINN=ON -DWITH_GPU=ON -DCINN_WITH_CUDNN=ON -DWITH_DLNNE=OFF -DWITH_TESTING=OFF \
      -DWITH_MKL=OFF -DPYTHON_EXECUTABLE=/usr/local/bin/python3 -DPYTHON_LIBRARIES=/usr/local/python3/include/python3.8 -DPYTHON_INCLUDE_DIR=/usr/local/python3/lib/libpython3.8.a -DPY_VERSION=3.8 -G Ninja ..
ninja -j 16
