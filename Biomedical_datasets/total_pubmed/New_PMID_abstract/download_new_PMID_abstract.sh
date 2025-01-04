#!/bin/bash

# 基础URL
base_url="ftp://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles"

# 指定下载目录
download_dir="/home/gy237/project/Biomedical_datasets/total_pubmed/New_PMID_abstract/Row_data_daily"


# 循环下载pubmed24n0001.xml.gz到pubmed24n1219.xml.gz
# pubmed24n1220.xml.gz - pubmed24n1575.xml.gz, 2024-10-21 14:04

# pubmed24n0585.xml.gz 
# pubmed24n1124.xml.gz
# pubmed24n0046.xml.gz
# wget -P /home/gy237/project/Biomedical_datasets/total_pubmed/New_PMID_abstract/abstracts_1-1219 ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/

for i in $(seq -w 1564 1575); do
    # 格式化文件名
    file="pubmed24n${i}.xml.gz"
    
    # 使用wget下载文件到指定目录
    wget -P "$download_dir" "${base_url}/${file}"

    # 格式化文件名
    file="pubmed24n${i}.xml.gz.md5"
    
    # 使用wget下载文件到指定目录
    wget -P "$download_dir" "${base_url}/${file}"
done
