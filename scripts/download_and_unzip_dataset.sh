#!/usr/bin/env bash

readonly DOWNLOAD_URL="https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip"
readonly FILE_NAME="data.zip"
readonly CURRENT_DIR=$(dirname "$(readlink -f "$0")")
readonly DATA_DIR=$(realpath "$CURRENT_DIR/../data/")

if [ ! -f $DATA_DIR/$FILE_NAME ]; then
    wget -P $DATA_DIR $DOWNLOAD_URL
fi

if [ ! -f $DATA_DIR/$FILE_NAME ]; then
    echo "Cannot download data"
    exit
fi

cd $DATA_DIR
pwd
unzip $FILE_NAME
echo "Finish unzipping"
cd $CURRENT_DIR
