#ifndef CPU_H
#define CPU_H

#include "constants.hpp"

/****************************Function definition*******************************/
static void CPU_convolution(std::vector<short> image, std::vector<short> &result,
  int Nx, int Ny, int filterChoice);

static void CPU_thresholding(std::vector<short> &image, int Nx, int Ny);

static void CPU_edgeDetection(std::vector<short> image, std::vector<short> &result,
  int Nx, int Ny);

/****************************Function implementation***************************/
static void CPU_convolution(std::vector<short> image, std::vector<short> &result,
  int Nx, int Ny, int filterChoice){
/* This function implements a convolution of 'image' and the filter of choice and
  returns 'result' (both input and output are Nx x Ny pixels size) */

  // First we obtain the kernel size and radius depending on the coice
  int kerRad, M;
  switch (filterChoice) {
    case gaussian:
      kerRad = kerRadGauss;
      M = kerSizeGauss;
      break;
    case laplacian:
      kerRad = kerRadLaplacian;
      M = kerSizeLaplacian;
      break;
  }

  // We create a float varialbe to handle the filters (it will be later
  // changed back to short)
  float tmp = 0.0;

  // For each pixel in the image
  for (int j = 0; j < Ny; ++j){
    for (int i = 0; i < Nx; ++i){
      // For all the pixels in the neigbourhood
      for (int y = -kerRad; y <= kerRad; ++y){
        for (int x = -kerRad; x <= kerRad; ++x){
          // If the neigbour pixel is outside of the image we do not use it
          if ((i+x < 0) || (j+y < 0) || (i+x >= Nx) || (j+y >= Ny)) continue;
          // Otherwise we compute the weighted sum
          switch (filterChoice) {
            case gaussian:
              tmp += (gaussFilt[x+kerRad + M*(y+kerRad)] * (float) image[(i+x) + Nx*(j+y)]);
              break;
            case laplacian:
              tmp += (LaplacianFilt[x+kerRad + M*(y+kerRad)] * image[(i+x) + Nx*(j+y)]);
              break;
          }
        }
      }
      result[i + Nx*j] = (short) tmp;
      tmp = 0.0;
    }
  }
}

static void CPU_thresholding(std::vector<short> &image, int Nx, int Ny){
/* This function computes the thresholded image after having been filtered */

  // For each pixel in the image
  for (int j = 0; j < Ny; ++j){
    for (int i = 0; i < Nx; ++i){
      // We apply the threshold
      image[i+Nx*j] = (image[i+Nx*j] > threshold) ? 1 : 0;
    }
  }
}


static void CPU_edgeDetection(std::vector<short> image, std::vector<short> &result,
  int Nx, int Ny){
/* This function implements the full edge detection computation by first computing
  the smoothing of the 'image' and then applying the Laplacian filter and threshold
  it. The result is then written in 'result' */

  std::vector<short> filtered (Nx*Ny);
  CPU_convolution(image, filtered, Nx, Ny, gaussian);
  CPU_convolution(filtered, result, Nx, Ny, laplacian);
  CPU_thresholding(result, Nx, Ny);
}

#endif
