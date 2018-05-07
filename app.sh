#!/bin/bash

if [ ! -f ./app/edgeDetection ]; then
   echo "File edgeDetection does not exist, compiling..."
   nvcc -std=c++11 -o ./app/edgeDetection ./app/edgeDetection.cu
else
   echo "File edgeDetection exists."
fi

python3 ./app/getImage.py
./app/edgeDetection < tmp_input.txt
python3 ./app/saveImage.py
rm -rf ./tmp_img.txt ./edge_detect.txt ./tmp_input.txt
