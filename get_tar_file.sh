function _download_and_untar {
    local tar_file=$1
    if [[ ! -f $tar_file ]]; then
        wget https://paddle-inference-dist.bj.bcebos.com/CINN/$tar_file
        tar -zxvf $tar_file
    fi
}


function prepare_model {
    cd $build_dir/models
    _download_and_untar ResNet18.tar.gz
    _download_and_untar MobileNetV2.tar.gz
    _download_and_untar EfficientNet.tar.gz
    _download_and_untar MobilenetV1.tar.gz
    _download_and_untar ResNet50.tar.gz
    _download_and_untar SqueezeNet.tar.gz
    _download_and_untar FaceDet.tar.gz
    wget http://paddle-inference-dist.bj.bcebos.com/lite_naive_model.tar.gz
    tar zxf lite_naive_model.tar.gz
}


build_dir = "/home/wzy/Paddle-CINN-model"
prepare_model