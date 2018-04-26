#ifndef GPU_NAIVE_H
#define GPU_NAIVE_H

#include "constants.hpp"

/****************************Function definition*******************************/
__global__ void CUDA_convolution_naive(short *image, short *result,
  float *filt, int Nx, int Ny, int kerRad);

static void GPU_convolution_naive(std::vector<short> image, std::vector<short> &result,
  int Nx, int Ny, int filterChoice);

/****************************Function implementation***************************/
__global__ void CUDA_convolution_naive(short *image, short *result,
  float *filt, int Nx, int Ny, int kerRad){
/* This function implements the naive convolution in CUDA */

  int M = 2*kerRad + 1;

  //for each pixel in the result
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  //thread-local register to hold local sum
  float tmp = 0.0;

  //for each pixel in the neigbourhood
  for (int y = -kerRad; y <= kerRad; ++y){
    for (int x = -kerRad; x <= kerRad; ++x){
      // If the neigbour pixel is outside of the image we do not use it
      if ((i+x < 0) || (j+y < 0) || (i+x >= Nx) || (j+y >= Ny)) continue;
      // Otherwise we compute the weighted sum
      tmp += (filt[x+kerRad + M*(y+kerRad)] * image[(i+x) + Nx*(j+y)]);
    }
  }

  //store result to global memory
  result[i + Nx*j] = (short) tmp;

}

static void GPU_convolution_naive(std::vector<short> image, std::vector<short> &result,
  int Nx, int Ny, int filterChoice){
/* This function implements the call to the CUDA naive convolution */

  // First we obtain the kernel size and radius depending on the coice
  int M, kerRad;
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

   // Create the device copies for the GPU and copy them to the device
   int sizeImg = Nx*Ny * sizeof(short);
   int sizeRes = Nx*Ny * sizeof(short);
   int sizeFilt = M*M * sizeof(float);;
   short *d_img, *d_res;
   float *d_filt;

   cudaMalloc((void **)&d_img, sizeImg);
   cudaMalloc((void **)&d_res, sizeRes);
   cudaMalloc((void **)&d_filt, sizeFilt);

   cudaMemcpy(d_img, &image[0], sizeImg, cudaMemcpyHostToDevice);

   switch (filterChoice) {
     case gaussian:
       cudaMemcpy(d_filt, &gaussFilt[0], sizeFilt, cudaMemcpyHostToDevice);
       break;
     case laplacian:
       cudaMemcpy(d_filt, &LaplacianFilt[0], sizeFilt, cudaMemcpyHostToDevice);
       break;
    }

   // Launch conv() kernel on GPU with N blocks
   dim3 threads(blockSizeX, blockSizeY);
   dim3 blocks(int(ceilf(Nx/(float)blockSizeX)), int(ceilf(Ny/(float)blockSizeY)));
   CUDA_convolution_naive<<<blocks,threads>>>(d_img, d_res, d_filt, Nx, Ny, kerRad);

   // Copy result back to host
   cudaMemcpy(&result[0], d_res, sizeRes, cudaMemcpyDeviceToHost);

   // Cleanup
   cudaFree(d_img); cudaFree(d_filt); cudaFree(d_res);
}

#endif
