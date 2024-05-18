#!/bin/bash

# 下载文件函数
download_file() {
    local url=$1
    local file_id=$(echo $url | grep -o 'd/[^/]*' | cut -d'/' -f2)
    local confirm=$(curl -sc /tmp/gcookie "https://drive.google.com/uc?export=download&id=${file_id}" | grep -o 'confirm=[^&]*' | cut -d'=' -f2)
    local file_name=$(curl -s -Lb /tmp/gcookie "https://drive.google.com/uc?export=download&id=${file_id}&confirm=${confirm}" | grep -o 'filename="[^"]*"' | cut -d'"' -f2)
    wget --load-cookies /tmp/gcookie "https://drive.google.com/uc?export=download&id=${file_id}&confirm=${confirm}" -O "${file_name}"
    rm -rf /tmp/gcookie
}

# 文件链接
url="https://drive.google.com/file/d/15_YBkQAnToRJ9xEOnK3FoqotHgoTvf6j/view?usp=sharing"

# 下载文件
download_file "$url"

echo "文件下载完成！"
