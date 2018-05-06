#!/bin/bash

tmp="0"
for filename in ../data/512/*.txt
do
  echo '512 512' $filename > tmp_input.txt
  ./edgeDetection < tmp_input.txt
  tmp=$[ $tmp+1 ]
  echo ../figures/resulting_images/512/img$tmp.png > tmp_input.txt
  python3 saveImage2.py < tmp_input.txt
  rm tmp_input.txt
done

tmp="0"
for filename in ../data/1024/*.txt
do
  echo '1024 1024' $filename > tmp_input.txt
  ./edgeDetection < tmp_input.txt
  tmp=$[ $tmp+1 ]
  echo ../figures/resulting_images/1024/img$tmp.png > tmp_input.txt
  python3 saveImage2.py < tmp_input.txt
  rm tmp_input.txt
done

tmp="0"
for filename in ../data/2048/*.txt
do
  echo '2048 2048' $filename > tmp_input.txt
  ./edgeDetection < tmp_input.txt
  tmp=$[ $tmp+1 ]
  echo ../figures/resulting_images/2048/img$tmp.png > tmp_input.txt
  python3 saveImage2.py < tmp_input.txt
  rm tmp_input.txt
done
