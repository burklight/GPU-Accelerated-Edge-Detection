# GPU accelerated edge detection

For this project you need the following dependencies:
- [CUDA 9.1](https://developer.nvidia.com/cuda-downloads) or higher
- python3, numpy, matplotlib, mpl_toolkits which can be easily installed through pip3
- an NVIDIA GPU

To see the test we performed for the project you should type:
- bash run_tests.sh
- if you are interested in the raw data used for the plots in testSpeedConv.cu, please comment the lines 49 and 50 of run_tests.sh

To use an application of the project you should type. You can the 1.jpg image as an example:
- bash app.sh path_to_a_square_image

To get the results of the application with the best parallelization method in our 15 selected figures you should type (the result will be stored in figures/resulting_images):
- bash doAllImage.sh

The report of the project can be found in:
- [report](https://github.com/burklight/Parallels/blob/master/sf2568-project-gpu.pdf)
