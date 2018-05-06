#!/bin/bash

python3 getImage.py
./edgeDetection < tmp_input.txt
python3 saveImage.py
rm -r tmp_img.txt edge_detect.txt tmp_input.txt
