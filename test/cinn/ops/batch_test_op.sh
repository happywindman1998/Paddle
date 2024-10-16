#!/bin/bash

# 定义 Python 解释器的路径
python_path="/home/wzy/build-python3-10-14/bin/python3"

# 定义要运行的 Python 脚本列表
scripts=(
        # "test_abs_op.py" "test_acos_op.py" "test_acosh_op.py" "test_add_op.py" \
        # "test_arange_op.py" \
        # "test_argmax_op.py" "test_argmin_op.py" "test_argsort_op.py" "test_asin_op.py" "test_asinh_op.py" \
        # "test_atan_op.py" "test_atanh_op.py" "test_batch_norm_op.py" "test_binary_elementwise_op.py" \
        # "test_bitcast_convert_op.py" "test_bitwise_op.py" \
        # "test_broadcast_to_op.py" "test_cast_op.py" "test_cbrt_op.py" "test_ceil_op.py" "test_comparison_op.py" \
        # "test_concat_op.py" "test_constant_op.py" "test_conv2d_op.py" "test_cos_op.py" "test_cosh_op.py" \
        # "test_depthwise_conv2d_op.py" "test_divide_op.py" "test_dropout_infer_op.py" \
        # "test_erf_op.py" "test_exp_op.py" "test_expand_dims.py" "test_floor_op.py" \
        # "test_gather_op.py" "test_gaussian_random_op.py" "test_gelu_op.py" "test_is_finite_op.py" \
        # "test_is_inf_op.py" "test_is_nan_op.py" "test_isclose_op.py" "test_left_shift_op.py" "test_log_op.py" \
        # "test_matmul_op.py" "test_max_op.py" "test_min_op.py" "test_mod_op.py" "test_mul_op.py" "test_multiply_op.py" \
        # "test_negative_op.py" "test_one_hot_op.py" "test_pool2d_op.py" "test_pow_op.py" "test_randint_op.py" "test_reduce_op.py" \
        # "test_relu_op.py" "test_repeat_op.py" "test_reshape_op.py" "test_reverse_op.py" "test_right_shift_op.py" "test_round_op.py" \
        # "test_select_op.py" "test_sigmoid_op.py" "test_sign_op.py" "test_sin_op.py" "test_sinh_op.py" \
        # "test_slice_op.py" "test_softmax_op.py" "test_sort_op.py" "test_split_op.py" "test_sqrt_op.py" \
        # "test_squeeze_op.py" "test_subtract_op.py" "test_sum_op.py" "test_tan_op.py" "test_tanh_op.py" \
        # "test_top_k_op.py" "test_transpose_op.py" "test_trunc_op.py" "test_uniform_random_op.py" "test_fill_constant_op.py"
        "test_argsort_op.py" "test_sort_op.py"
        )

# 日志文件
log_file="test_op.log"

# 清空日志文件（可选）
# > "$log_file"

# 循环遍历并执行每个脚本
for script in "${scripts[@]}"; do
    echo "Running $script..."
    echo -e "Running $script...\n" &>> "$log_file"
    # 执行 Python 脚本并将输出发送到控制台和日志文件
    $python_path "$script" &>> "$log_file"
    echo "Finished running $script."
    echo -e "\nFinished running $script.\n" &>> "$log_file"
    echo ""
done