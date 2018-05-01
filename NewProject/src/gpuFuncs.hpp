#ifndef GPU_H
#define GPU_H

#include "constants.hpp"
#include "gpuNaive.hpp"
#include "gpuShared.hpp"
#include "gpuConst.hpp"
#include "gpuSeparable.hpp"

/****************************Function definition*******************************/
__global__ void CUDA_thresholding(short *image, int Nx);

static void GPU_thresholding(std::vector<short> &image, int Nx, int Ny);

static void GPU_edgeDetection(std::vector<short> image, std::vector<short> &result,
  int Nx, int Ny, int method);

/****************************Function implementation***************************/
__global__ void CUDA_thresholding(short *image, int Nx){
/* implementation of the thresholding in the 'image' */

  // for each pixel in the result
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // we apply the threshold
  image[i + Nx*j] = (image[i + Nx*j] > threshold) ? 1 : 0;

}

static void GPU_thresholding(std::vector<short> &image, int Nx, int Ny){
/* implementation of the thresholding call to the GPU */

  // we create a device copy of the image
  int size_image = Nx*Ny * sizeof(short);
  short *d_image;
  cudaMalloc((void **)&d_image, size_image);
  cudaMemcpy(d_image, &image[0], size_image, cudaMemcpyHostToDevice);

  // we call the thresholding kernel
  dim3 threads(blockSizeX, blockSizeY);
  dim3 blocks(int(ceilf(Nx/(float)blockSizeX)), int(ceilf(Ny/(float)blockSizeY)));
  CUDA_thresholding<<<blocks,threads>>>(d_image, Nx);

  // Copy result back to host
  cudaMemcpy(&image[0], d_image, size_image, cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_image);
}

static void GPU_edgeDetection(std::vector<short> image, std::vector<short> &result,
  int Nx, int Ny, int method){
/* This function implements the full edge detection computation by first computing
  the smoothing of the 'image' and then applying the Laplacian filter and threshold
  it. The result is then written in 'result' */

    std::vector<short> filtered (Nx*Ny);
    switch (method) {
      case NAIVE:
        GPU_convolution_naive(image, filtered, Nx, Ny, gaussian);
        GPU_convolution_naive(filtered, result, Nx, Ny, laplacian);
        break;
      case SHARED:
        GPU_convolution_shared(image, filtered, Nx, Ny, gaussian);
        GPU_convolution_shared(filtered, result, Nx, Ny, laplacian);
        break;
      case CONSTANT:
        GPU_convolution_const(image, filtered, Nx, Ny, gaussian);
        GPU_convolution_const(filtered, result, Nx, Ny, laplacian);
        break;
      case SEPARABLE:
        GPU_convolution_sep(image, filtered, Nx, Ny); // This is only for gaussian
        GPU_convolution_const(filtered, result, Nx, Ny, laplacian);
        break;
    }
    GPU_thresholding(result, Nx, Ny);
}

#endif
