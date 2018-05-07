#!/bin/bash

# Compile the tests
if [ ! -f ./tests/checkSpeed ]; then
   echo "File checkSpeed does not exist, compiling..."
   nvcc -std=c++11 -o ./tests/checkSpeed ./tests/checkSpeed.cu
else
   echo "File checkSpeed exists."
fi

if [ ! -f ./tests/checkCorrectness ]; then
   echo "File checkCorrectness does not exist, compiling..."
   nvcc -std=c++11 -o ./tests/checkCorrectness ./tests/checkCorrectness.cu
else
   echo "File checkCorrectness exists."
fi

if [ ! -f ./tests/testSpeedConv ]; then
   echo "File testSpeedConv does not exist, compiling..."
   nvcc -std=c++11 -o ./tests/testSpeedConv ./tests/testSpeedConv.cu
else
   echo "File testSpeedConv exists."
fi

if [ ! -d ./figures ]; then
  echo "Directory figures does not exist, creating it..."
  mkdir ./figures
  mkdir ./figures/resulting_images
  mkdir ./figures/resulting_images/512
  mkdir ./figures/resulting_images/1024
  mkdir ./figures/resulting_images/2048
else
  echo "Directory figures exists."
fi

# Run the tests
echo "Checking speed of convolutions..."
./tests/checkSpeed
echo "Testing speed of convolutions..."
./tests/testSpeedConv
echo "Testing correctness of convolutions..."
./tests/checkCorrectness
python3 ./tests/plot_check_correctness.py
echo "Theoretical arithmetic intensity"
python3 ./tests/plot_theoretical_ai.py
python3 ./tests/plot_test_speed_conv.py

# Cleanup
rm -f ./tests/*.txt
rm -f ./*.txt
