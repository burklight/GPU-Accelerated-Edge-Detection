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

#define NAIVE 1
#define SHARED 2
#define CONSTANT 3 


const std::vector<float> gaussFilt =
{
  2 / 159.0,  4 / 159.0,  5 / 159.0,  4 / 159.0,  2 / 159.0,
  4 / 159.0,  9 / 159.0, 12 / 159.0,  9 / 159.0,  4 / 159.0,
  5 / 159.0, 12 / 159.0, 15 / 159.0, 12 / 159.0,  5 / 159.0,
  4 / 159.0,  9 / 159.0, 12 / 159.0,  9 / 159.0,  4 / 159.0,
  2 / 159.0,  4 / 159.0,  5 / 159.0,  4 / 159.0,  2 / 159.0
};

const std::vector<float> LaplacianFilt =
{
  0,  1,  0,
  1, -4,  1,
  0,  1,  0
};


#endif
