#!/bin/bash

download_file() {
    local url=$1
    local file_id=$(echo $url | grep -o 'd/[^/]*' | cut -d'/' -f2)
    local file_name=$(curl -s -L "https://drive.google.com/uc?export=download&id=${file_id}" | grep -o 'filename="[^"]*"' | cut -d'"' -f2)
    wget --no-check-certificate "https://drive.google.com/uc?export=download&id=${file_id}" -O "${file_name}"
}

urls=(
    "https://drive.google.com/file/d/1ca7J--RN_AdTdvquFGGW8qYVzaHXYp_5/view?usp=sharing"
    "https://drive.google.com/file/d/1d9cZDyBPSOUq3qM0MNMLZuJ_YdRtn5Kl/view?usp=sharing"
    "https://drive.google.com/file/d/1a3JJTKnKWNuw4ALChigETyDRc51Hlph0/view?usp=sharing"
    "https://drive.google.com/file/d/1rDQIKeRlpC1Eyd_V1rF5PpGxXVzXHklX/view?usp=sharing"
    "https://drive.google.com/file/d/1PRY_yD4z0ct6GZBVsxABHRYy6kkSPlhZ/view?usp=sharing"
    "https://drive.google.com/file/d/1HEH38IWdVj5UTsIe6u1z-BijpormlRxK/view?usp=sharing"
    "https://drive.google.com/file/d/1mWJlk-p0Ii3IzCfRwVvfWaWPLtTNHdZi/view?usp=sharing"
    "https://drive.google.com/file/d/1qH6d5Bf5XhzTvr_eooE08jrPJK2SKl1g/view?usp=sharing"
)

for url in "${urls[@]}"; do
    download_file "$url"
done

echo "finish！"
