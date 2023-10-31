#!/bin/bash

# Define file links
file_links=(
    "https://drive.google.com/uc?export=download&id=1vGWt-qF_DEKKhi-mtSpO7S_zkUCXUBvQ"
    "https://drive.google.com/uc?export=download&id=1F250henyzJ4CHo_lLmbNL1pi44MvLeYI"
    "https://drive.google.com/uc?export=download&id=1-IsjdzaQYWhcuP1ZfCiHX0Sw5uILIBf9"
    "https://drive.google.com/uc?export=download&id=1hQIzHrTZUXazDdw8uraAoaJfLyOYUpk8"
    "https://drive.google.com/uc?export=download&id=1JKwX9YvmSaZ68V0c-fzGn2NQ0ovRo-ld"
    "https://drive.google.com/uc?export=download&id=1Ks4DwXiU3vAZ6DE6VeIy_2BpIP_7Qqs9"
    "https://drive.google.com/uc?export=download&id=1QGBgZaCwVqUVcKp0R8vSlG62ir13bbIJ"
    "https://drive.google.com/uc?export=download&id=11wwxVzyMjCH2GgFg4LavoJbp3YiQO_dr"
)

# Loop through file links and download files with default names
for file_link in "${file_links[@]}"; do
    # Use wget to download the file (you can use curl as well)
    wget "$file_link"

    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "Download successful: $file_link"
    else
        echo "Download failed: $file_link"
    fi
done
