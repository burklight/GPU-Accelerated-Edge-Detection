#ifndef GPU_SEP_H
#define GPU_SEP_H

#include "constants.hpp"

/****************************Function definition*******************************/
__global__ void CUDA_convolution_row(short *image, short *result,
  int Nx, int Ny);

__global__ void CUDA_convolution_col(short *image, short *result,
  int Nx, int Ny);

static void GPU_convolution_sep(std::vector<short> image, std::vector<short> &result,
  int Nx, int Ny);


/****************************Function implementation***************************/
__global__ void CUDA_convolution_row(short *image, short *result,
  int Nx, int Ny) {
/* This function implements the row convolution in CUDA for Gaussian kernel */

    int kerRad = kerRadGauss;

    //for each pixel in the result
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //some data is shared between the threads in a block
    __shared__ short data[blockSizeX+kerSizeGauss-1][blockSizeY];

    //position of the thread (X-axis) in the shared data
    int idx = threadIdx.x + kerRad;
    int idy = threadIdx.y;

    //position of the tread within the block
    int thx = threadIdx.x;

    //each thread loads its own position
    data[idx][idy] = image[i+Nx*j];

    //load left apron
    if (thx-kerRad < 0){
      if (i-kerRad < 0) data[idx-kerRad][idy] = 0;
      else data[idx-kerRad][idy] = image[(i-kerRad)+Nx*j];
    }

    //load right apron
    if (thx+kerRad >= blockDim.x){
      if (i+kerRad >= Nx) data[idx+kerRad][idy] = 0;
      else data[idx+kerRad][idy] = image[(i+kerRad)+Nx*j];
    }

    //make sure that all the data is available for all the threads
    __syncthreads();

    //thread-local register to hold local sum
    float tmp = 0.0;

    //for each filter weight
    for (int x = -kerRad; x <= kerRad; ++x){
      if ((i+x < 0) || (i+x >= Nx)) continue;
      tmp += (gaussianKernelLin[x+kerRad] * data[idx+x][idy]);
    }

    //store result to global memory
    result[i + Nx*j] = tmp;
}

__global__ void CUDA_convolution_col(short *image, short *result,
  int Nx, int Ny) {
/* This function implements the row convolution in CUDA for Gaussian kernel */

    int kerRad = kerRadGauss;

    //for each pixel in the result
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //some data is shared between the threads in a block
    __shared__ short data[blockSizeX][blockSizeY+kerSizeGauss-1];

    //position of the thread (Y-axis) in the shared data
    int idx = threadIdx.x;
    int idy = threadIdx.y + kerRad;

    //position of the tread within the block
    int thy = threadIdx.y;

    //each thread loads its own position
    data[idx][idy] = image[i+Nx*j];

    //load top apron
    if (thy-kerRad < 0){
      if (j-kerRad < 0) data[idx][idy-kerRad] = 0;
      else data[idx][idy-kerRad] = image[i+Nx*(j-kerRad)];
    }

    //load bottom apron
    if (thy+kerRad >= blockDim.y){
      if (j+kerRad >= Ny) data[idx][idy+kerRad] = 0;
      else data[idx][idy+kerRad] = image[i+Nx*(j+kerRad)];
    }

    //make sure that all the data is available for all the threads
    __syncthreads();

    //thread-local register to hold local sum
    float tmp = 0.0;

    //for each filter weight
    for (int y = -kerRad; y <= kerRad; ++y){
      if ((j+y < 0) || (j+y >= Ny)) continue;
      tmp += (gaussianKernelLin[y+kerRad] * data[idx][idy+y]);
    }

    //store result to global memory
    result[i + Nx*j] = tmp;
}

static void GPU_convolution_sep(std::vector<short> image, std::vector<short> &result,
  int Nx, int Ny){
/* This function implements the call to the CUDA separable filters convolution*/

  // First we obtain the kernel size and radius depending on the coice
  int M = kerSizeGauss;

  // Create the device copies for the GPU and copy them to the device
  int sizeImg = Nx*Ny * sizeof(short);
  int sizeTmp = Nx*Ny * sizeof(short);
  int sizeRes = Nx*Ny * sizeof(short);
  int sizeFilt = M * sizeof(float);
  short *d_img, *d_tmp, *d_res;

  cudaMalloc((void **)&d_img, sizeImg);
  cudaMalloc((void **)&d_tmp, sizeTmp);
  cudaMalloc((void **)&d_res, sizeRes);

  cudaMemcpy(d_img, &image[0], sizeImg, cudaMemcpyHostToDevice);
  dim3 threads(blockSizeX, blockSizeY);
  dim3 blocks(int(ceilf(Nx/(float)blockSizeX)), int(ceilf(Ny/(float)blockSizeY)));

  cudaMemcpyToSymbol(gaussianKernelLin, &gaussFiltLin[0], sizeFilt);
  CUDA_convolution_row<<<blocks,threads>>>(d_img, d_tmp, Nx, Ny);
  CUDA_convolution_col<<<blocks,threads>>>(d_tmp, d_res, Nx, Ny);

  // Copy result back to host
  cudaMemcpy(&result[0], d_res, sizeRes, cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_img); cudaFree(d_res);

}

#endif
