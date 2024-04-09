
<p align="center">
<img align="center" src="doc/imgs/logo.png", width=1600>
<p>

--------------------------------------------------------------------------------
## Paddle-CINN-oneDNN的搭建过程

### Step 0. docker镜像构建
- 构建docker的必要性：
1. Paddle目前只能用GCC8.2编译，其他高版本或者低版本的GCC都编译不通过。
2. 其次，不同加速器依赖的驱动与运行时库都不同。为不同的加速器构建不同的镜像，方便以后使用
- 构建Paddle的编译环境需要：
1. 操作系统最好是ubuntu18.04,与paddle官方给出的cuda构建环境保持一致，出错概论可能会小一些
2. GCC8.2
3. 加速器的驱动与运行时环境
4. python3.8 以上

- 将intel-llvm，oneTBB，oneDNN, paddle，四个项目源文件放到同一个目录下，并在此目录上启动docker。
`docker run --gpus=all --network=host --name cinn-sycl-v2 --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -it -v $PWD:/CINN-SYCL cinn:v2 /bin/bash`
- <font color=#FF000 > 注意，以下编译步骤都需要在Docker容器中进行。编译的次序为：intel-llvm -> oneTBB -> oneDNN -> paddle </font>

### Step 1. Build Intel SYCL LLVM
- ~~当前选择的intel-llvm版本为20220501，选择依据是为了兼容我们使用的oneDNN版本~~
- 选用llvm-sycl 20220501的版本是为了兼容DCU，更高版的llvm在dtk-23.04.1上执行asinh，和atanh会有精度问题

### Step 2. Build oneDNN
- 当前选择的oneDNN版本是v3.2，选择依据是这个版本具有MIOpen的实现
- oneDNN依赖于oneTBB，下载与编译oneTBB，编译命令：
    - mkdir build & cd build
    - cmake .. & make -j 8
    - make install
- oneDNN的编译命令为:

```
# 添加llvm-sycl的bin与lib环境变量
source /CINN-SYCL/llvm-sycl-nightly-20220501/env.sh

export CC=clang
export CXX=clang++
export TBBROOT=/CINN-SYCL/oneTBB_installed

mkdir build
cd build
cmake -DDNNL_CPU_RUNTIME=TBB -DDNNL_GPU_RUNTIME=DPCPP \
      -DDNNL_GPU_VENDOR=NVIDIA -DONEDNN_BUILD_GRAPH=OFF -G Ninja ..
ninja -j 16

```
- 其中，DNNL_CPU_RUNTIME或许可以设为DPCPP（未尝式），但是一定不能设为None，因为Paddle中有些算子需要调到用cpu

### Step 3. Build Paddle
当前选择的Paddle版本为2.6，是最新的发版。
下载源文件并编译，编译命令为：

```
export CC=gcc
export CXX=g++

export DPCPP_ROOT=/CINN-SYCL/llvm-sycl-nightly-20220501
export ONEDNN_ROOT=/CINN-SYCL/oneDNN

mkdir build-docker
cd build-docker
cmake -DCINN_ONLY=ON -DWITH_CINN=ON -DWITH_GPU=OFF -DCINN_WITH_SYCL=ON  -DCINN_WITH_ONEDNN=ON -DWITH_DLNNE=ON -DWITH_TESTING=OFF \
      -DWITH_MKL=OFF -DPYTHON_EXECUTABLE=python3.8 -DPY_VERSION=3.8 -G Ninja ..
ninja -j 16

```
-----------------------------------------------------------------------------------
## CINN-oneDNN的测试
1. 安装cinn的whl包：
`pip3 install -U build-docker/python/dist/cinn-0.0.0-py3-none-any.whl`

2. GEMM单算子测试
`python3 simple_onednn_test.py`

3. ResNet18模型测试
  - 下载ResNet18的paddle模型，`wget https://paddle-inference-dist.bj.bcebos.com/CINN/ResNet18.tar.gz`
  - 解压文件，`tar -zxvf ResNet18.tar.gz`
  - 导入环境变量，`export FLAGS_cinn_infer_model_version=1.0`
  - 测试，`cd test/cinn && python3 test_onednn_resnet18.py Resnet_model_dir ON`

-----------------------------------------------------------------------------------

English | [简体中文](./README_cn.md) | [日本語](./README_ja.md)

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)
[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle.svg)](https://github.com/PaddlePaddle/Paddle/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-1ca0f1.svg?logo=twitter&logoColor=white)](https://twitter.com/PaddlePaddle)

Welcome to the PaddlePaddle GitHub.

PaddlePaddle, as the first independent R&D deep learning platform in China, has been officially open-sourced to professional communities since 2016. It is an industrial platform with advanced technologies and rich features that cover core deep learning frameworks, basic model libraries, end-to-end development kits, tools & components as well as service platforms.
PaddlePaddle is originated from industrial practices with dedication and commitments to industrialization. It has been widely adopted by a wide range of sectors including manufacturing, agriculture, enterprise service, and so on while serving more than 8 million developers, 220,000 companies and generating 800,000 models. With such advantages, PaddlePaddle has helped an increasing number of partners commercialize AI.


## Installation

### Latest PaddlePaddle Release: [v2.5](https://github.com/PaddlePaddle/Paddle/tree/release/2.5)

Our vision is to enable deep learning for everyone via PaddlePaddle.
Please refer to our [release announcement](https://github.com/PaddlePaddle/Paddle/releases) to track the latest features of PaddlePaddle.
### Install Latest Stable Release:
```
# CPU
pip install paddlepaddle
# GPU
pip install paddlepaddle-gpu

```
For more information about installation, please view [Quick Install](https://www.paddlepaddle.org.cn/install/quick)

Now our developers can acquire Tesla V100 online computing resources for free. If you create a program by AI Studio, you will obtain 8 hours to train models online per day. [Click here to start](https://aistudio.baidu.com/aistudio/index).

## FOUR LEADING TECHNOLOGIES

- **Agile Framework for Industrial Development of Deep Neural Networks**

    The PaddlePaddle deep learning framework facilitates the development while lowering the technical burden, through leveraging a programmable scheme to architect the neural networks. It supports both declarative programming and imperative programming with both development flexibility and high runtime performance preserved.  The neural architectures could be automatically designed by algorithms with better performance than the ones designed by human experts.


-  **Support Ultra-Large-Scale Training of Deep Neural Networks**

    PaddlePaddle has made breakthroughs in ultra-large-scale deep neural networks training. It launched the world's first large-scale open-source training platform that supports the training of deep networks with 100 billion features and trillions of parameters using data sources distributed over hundreds of nodes. PaddlePaddle overcomes the online deep learning challenges for ultra-large-scale deep learning models, and further achieved real-time model updating with more than 1 trillion parameters.
     [Click here to learn more](https://github.com/PaddlePaddle/Fleet)


- **High-Performance Inference Engines for Comprehensive Deployment Environments**

   PaddlePaddle is not only compatible with models trained in 3rd party open-source frameworks , but also offers complete inference products for various production scenarios. Our inference product line includes [Paddle Inference](https://paddle-inference.readthedocs.io/en/master/guides/introduction/index_intro.html): Native inference library for high-performance server and cloud inference; [Paddle Serving](https://github.com/PaddlePaddle/Serving): A service-oriented framework suitable for distributed and pipeline productions; [Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite): Ultra-Lightweight inference engine for mobile and IoT environments; [Paddle.js](https://www.paddlepaddle.org.cn/paddle/paddlejs): A frontend inference engine for browser and mini-apps. Furthermore, by great amounts of optimization with leading hardware in each scenario, Paddle inference engines outperform most of the other mainstream frameworks.


- **Industry-Oriented Models and Libraries with Open Source Repositories**

     PaddlePaddle includes and maintains more than 100 mainstream models that have been practiced and polished for a long time in the industry. Some of these models have won major prizes from key international competitions. In the meanwhile, PaddlePaddle has further more than 200 pre-training models (some of them with source codes) to facilitate the rapid development of industrial applications.
     [Click here to learn more](https://github.com/PaddlePaddle/models)


## Documentation

We provide [English](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html) and
[Chinese](https://www.paddlepaddle.org.cn/documentation/docs/zh/guide/index_cn.html) documentation.

- [Guides](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)

  You might want to start from how to implement deep learning basics with PaddlePaddle.

- [Practice](https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/index_cn.html)

  So far you have already been familiar with Fluid. And the next step should be building a more efficient model or inventing your original Operator.

- [API Reference](https://www.paddlepaddle.org.cn/documentation/docs/en/api/index_en.html)

   Our new API enables much shorter programs.

- [How to Contribute](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/08_contribution/index_en.html)

   We appreciate your contributions!

## Communication

- [Github Issues](https://github.com/PaddlePaddle/Paddle/issues): bug reports, feature requests, install issues, usage issues, etc.
- QQ discussion group: 441226485 (PaddlePaddle).
- [Forums](https://aistudio.baidu.com/paddle/forum): discuss implementations, research, etc.

## Courses

- [Server Deployments](https://aistudio.baidu.com/aistudio/course/introduce/19084): Courses introducing high performance server deployments via local and remote services.
- [Edge Deployments](https://aistudio.baidu.com/aistudio/course/introduce/22690): Courses introducing edge deployments from mobile, IoT to web and applets.

## Copyright and License
PaddlePaddle is provided under the [Apache-2.0 license](LICENSE).
