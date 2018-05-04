#ifndef CONSTANTS_H
#define CONSTANTS_H

#define blockSizeX 32
#define blockSizeY 32
#define gaussian 1
#define laplacian 2
#define kerSizeGauss 5
#define kerRadGauss 2
#define kerSizeLaplacian 3
#define kerRadLaplacian 1
#define threshold 5
#define tilingFactor 10
#define keplerSharedMemSize 49152
#define maxwellSharedMemSize 65536
#define pascalSharedMemSize 98304

#define NAIVE 1
#define SHARED 2
#define CONSTANT 3
#define SEPARABLE 4
#define TILING 5

/*
const std::vector<float> gaussFilt =
{
  2 / 159.0,  4 / 159.0,  5 / 159.0,  4 / 159.0,  2 / 159.0,
  4 / 159.0,  9 / 159.0, 12 / 159.0,  9 / 159.0,  4 / 159.0,
  5 / 159.0, 12 / 159.0, 15 / 159.0, 12 / 159.0,  5 / 159.0,
  4 / 159.0,  9 / 159.0, 12 / 159.0,  9 / 159.0,  4 / 159.0,
  2 / 159.0,  4 / 159.0,  5 / 159.0,  4 / 159.0,  2 / 159.0
};*/

const std::vector<float> gaussFilt =
{
   4 / 289.0,  8 / 289.0, 10 / 289.0,  8 / 289.0,  4 / 289.0,
   8 / 289.0, 16 / 289.0, 20 / 289.0, 16 / 289.0,  8 / 289.0,
  10 / 289.0, 20 / 289.0, 25 / 289.0, 20 / 289.0, 10 / 289.0,
   8 / 289.0, 16 / 289.0, 20 / 289.0, 16 / 289.0,  8 / 289.0,
   4 / 289.0,  8 / 289.0, 10 / 289.0,  8 / 289.0,  4 / 289.0
};

const std::vector<float> gaussFiltLin =
{
  2 / 17.0,  4 / 17.0, 5 / 17.0,  4 / 17.0,  2 / 17.0,
};

const std::vector<float> LaplacianFilt =
{
  0,  1,  0,
  1, -4,  1,
  0,  1,  0
};

/****************************Constant kernels in GPU***************************/
__constant__ float gaussianKernel[kerSizeGauss*kerSizeGauss];
__constant__ float laplacianKernel[kerSizeLaplacian*kerSizeLaplacian];
__constant__ float gaussianKernelLin[kerSizeGauss];


#endif
