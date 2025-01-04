#!/bin/bash

# 定义存储文件夹路径
folder_path="/home/gy237/project/Biomedical_datasets/total_pubmed/New_PMID_abstract/Row_data_daily"

# 循环遍历文件夹中的所有 .xml.gz 文件
for file in "$folder_path"/*.xml.gz; do
    # 获取文件的 MD5 校验文件路径
    md5_file="${file}.md5"
    
    # 如果 .md5 文件存在，继续校验
    if [ -f "$md5_file" ]; then
        # 读取 .md5 文件中的哈希值
        # 使用正则表达式提取哈希值
        expected_md5=$(grep -oP '(?<=\= )[a-f0-9]{32}' "$md5_file")
        
        # 计算 .xml.gz 文件的 MD5 值
        actual_md5=$(md5sum "$file" | awk '{print $1}')
        
        # 比较实际 MD5 和预期 MD5
        if [ "$expected_md5" != "$actual_md5" ]; then
            echo "验证失败: $file"
        fi
    else
        echo "MD5 文件缺失: $md5_file"
    fi
done
