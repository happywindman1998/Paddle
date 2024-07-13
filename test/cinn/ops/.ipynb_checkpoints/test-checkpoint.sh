#!/bin/bash

# 定义文件夹路径
FOLDER_PATH="."
OUTPUT_FOLDER_PATH="./test"

# 确保输出文件夹存在
mkdir -p "$OUTPUT_FOLDER_PATH"

# 初始化计数器
cnt=0

# 进入文件夹
cd "$FOLDER_PATH"

# 遍历文件夹中以"test"开头的所有Python文件
for file in test*.py; do
    # 检查文件是否存在
    if [ -f "$file" ]; then
        # 构造输出文件名，去掉.py后缀
        output_filename=$(basename "$file" .py)
        
        # 输出文件的完整路径
        output_file_path="$OUTPUT_FOLDER_PATH/${output_filename}.out"
        
        # 输出文件名和当前计数
        echo "正在运行文件: $file"
        echo "输出将保存到: $output_file_path"
        echo "cnt: $((cnt=cnt+1))"
        
        # 执行命令，将输出重定向到以脚本名命名的文件中
        # 等待命令执行完成
        python3 "$file" > "$output_file_path" 2>&1
        echo "文件 $file 运行完成。"
        
        # 检查上一个脚本是否成功执行
        if [ $? -ne 0 ]; then
            echo "文件 $file 执行失败。"
            exit 1
        fi
    fi
done

echo "所有脚本已依次运行完成，总共运行了 $cnt 个文件。"