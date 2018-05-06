#!/bin/bash

if [ ! -f ./app/edgeDetection ]; then
   echo "File edgeDetection does not exist, compiling..."
   nvcc -std=c++11 -o ./app/edgeDetection ./app/edgeDetection.cu
else
   echo "File edgeDetection exists."
fi

tmp="0"
for filename in ./data/512/*.txt
do
  echo '512 512' $filename > tmp_input.txt
  ./app/edgeDetection < tmp_input.txt
  tmp=$[ $tmp+1 ]
  echo ./figures/resulting_images/512/img$tmp.png > tmp_input.txt
  python3 ./app/saveImage2.py < tmp_input.txt
  rm tmp_input.txt
done

tmp="0"
for filename in ./data/1024/*.txt
do
  echo '1024 1024' $filename > tmp_input.txt
  ./app/edgeDetection < tmp_input.txt
  tmp=$[ $tmp+1 ]
  echo ./figures/resulting_images/1024/img$tmp.png > tmp_input.txt
  python3 ./app/saveImage2.py < tmp_input.txt
  rm tmp_input.txt
done

tmp="0"
for filename in ./data/2048/*.txt
do
  echo '2048 2048' $filename > tmp_input.txt
  ./app/edgeDetection < tmp_input.txt
  tmp=$[ $tmp+1 ]
  echo ./figures/resulting_images/2048/img$tmp.png > tmp_input.txt
  python3 ./app/saveImage2.py < tmp_input.txt
  rm tmp_input.txt
done
