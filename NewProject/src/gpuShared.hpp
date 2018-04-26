#ifndef GPU_SHARED_H
#define GPU_SHARED_H

#include "constants.hpp"

/****************************Function definition*******************************/
__global__ void CUDA_convolution_shared_gaussian(short *image, short *result,
  float *filt, int Nx, int Ny);

__global__ void CUDA_convolution_shared_laplacian(short *image, short *result,
  float *filt, int Nx, int Ny);

static void GPU_convolution_shared(std::vector<short> image, std::vector<short> &result,
  int Nx, int Ny, int filterChoice);

/****************************Function implementation***************************/
__global__ void CUDA_convolution_shared_gaussian(short *image, short *result,
  float *filt, int Nx, int Ny) {
/* This function implements the shared memory convolution in CUDA for Laplacian */

    int kerRad = kerRadGauss;
    int M = kerSizeGauss;

    //for each pixel in the result
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //some data is shared between the threads in a block
    __shared__ short data[blockSizeX+kerSizeGauss-1][blockSizeY+kerSizeGauss-1];

    //position of the thread in the shared data
    int idx = threadIdx.x + kerRad;
    int idy = threadIdx.y + kerRad;

    //position of the tread within the block
    int thx = threadIdx.x;
    int thy = threadIdx.y;

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

    //load top apron
    if (thy-kerRad < 0){
      if (j-kerRad < 0) data[idx][idy-kerRad] = 0;
      else data[idx][idy-kerRad] = image[i+Nx*(j-kerRad)];
    }

    //load bottom apron
    if (thy+kerRad >= blockDim.y){
      if (i+kerRad >= Ny) data[idx][idy+kerRad] = 0;
      else data[idx][idy+kerRad] = image[i+Nx*(j+kerRad)];
    }

    //load top-lef apron
    if ((thx-kerRad < 0) && (thy-kerRad < 0)){
      if ((i-kerRad < 0) || (j-kerRad < 0)) data[idx-kerRad][idy-kerRad] = 0;
      else data[idx-kerRad][idy-kerRad] = image[(i-kerRad)+Nx*(j-kerRad)];
    }

    //load top-right apron
    if ((thx+kerRad >= blockDim.x) && (thy-kerRad < 0)){
      if ((i+kerRad >= Nx) || (j-kerRad < 0)) data[idx+kerRad][idy-kerRad] = 0;
      else data[idx+kerRad][idy-kerRad] = image[(i+kerRad)+Nx*(j-kerRad)];
    }

    //load bottom-lef apron
    if ((thx-kerRad < 0) && (thy+kerRad >= blockDim.y)){
      if ((i-kerRad < 0) || (j+kerRad >= Ny)) data[idx-kerRad][idy+kerRad] = 0;
      else data[idx-kerRad][idy+kerRad] = image[(i-kerRad)+Nx*(j+kerRad)];
    }

    //load bottom-right apron
    if ((thx+kerRad  >= blockDim.x) && (thy+kerRad >= blockDim.y)){
      if ((i+kerRad >= Nx) || (j+kerRad >= Ny)) data[idx+kerRad][idy+kerRad] = 0;
      else data[idx+kerRad][idy+kerRad] = image[(i+kerRad)+Nx*(j+kerRad)];
    }

    //make sure that all the data is available for all the threads
    __syncthreads();

    //thread-local register to hold local sum
    float tmp = 0.0;

    //for each filter weight
    for (int y = -kerRad; y <= kerRad; ++y){
      for (int x = -kerRad; x <= kerRad; ++x){
          if ((i+x < 0) || (j+y < 0) || (i+x >= Nx) || (j+y >= Ny)) continue;
          tmp +=
            (filt[x+kerRad + M*(y+kerRad)] * data[idx+x][idy+y]);
        }
    }

    //store result to global memory
    result[i + Nx*j] = tmp;
}

__global__ void CUDA_convolution_shared_laplacian(short *image, short *result,
  float *filt, int Nx, int Ny) {
/* This function implements the shared memory convolution in CUDA for Laplacian */

    int kerRad = kerRadLaplacian;
    int M = kerSizeLaplacian;

    //for each pixel in the result
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //some data is shared between the threads in a block
    __shared__ short data[blockSizeX+kerSizeLaplacian-1][blockSizeY+kerSizeLaplacian-1];

    //position of the thread in the shared data
    int idx = threadIdx.x + kerRad;
    int idy = threadIdx.y + kerRad;

    //position of the tread within the block
    int thx = threadIdx.x;
    int thy = threadIdx.y;

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

    //load top apron
    if (thy-kerRad < 0){
      if (j-kerRad < 0) data[idx][idy-kerRad] = 0;
      else data[idx][idy-kerRad] = image[i+Nx*(j-kerRad)];
    }

    //load bottom apron
    if (thy+kerRad >= blockDim.y){
      if (i+kerRad >= Ny) data[idx][idy+kerRad] = 0;
      else data[idx][idy+kerRad] = image[i+Nx*(j+kerRad)];
    }

    //load top-lef apron
    if ((thx-kerRad < 0) && (thy-kerRad < 0)){
      if ((i-kerRad < 0) || (j-kerRad < 0)) data[idx-kerRad][idy-kerRad] = 0;
      else data[idx-kerRad][idy-kerRad] = image[(i-kerRad)+Nx*(j-kerRad)];
    }

    //load top-right apron
    if ((thx+kerRad >= blockDim.x) && (thy-kerRad < 0)){
      if ((i+kerRad >= Nx) || (j-kerRad < 0)) data[idx+kerRad][idy-kerRad] = 0;
      else data[idx+kerRad][idy-kerRad] = image[(i+kerRad)+Nx*(j-kerRad)];
    }

    //load bottom-lef apron
    if ((thx-kerRad < 0) && (thy+kerRad >= blockDim.y)){
      if ((i-kerRad < 0) || (j+kerRad >= Ny)) data[idx-kerRad][idy+kerRad] = 0;
      else data[idx-kerRad][idy+kerRad] = image[(i-kerRad)+Nx*(j+kerRad)];
    }

    //load bottom-right apron
    if ((thx+kerRad  >= blockDim.x) && (thy+kerRad >= blockDim.y)){
      if ((i+kerRad >= Nx) || (j+kerRad >= Ny)) data[idx+kerRad][idy+kerRad] = 0;
      else data[idx+kerRad][idy+kerRad] = image[(i+kerRad)+Nx*(j+kerRad)];
    }

    //make sure that all the data is available for all the threads
    __syncthreads();

    //thread-local register to hold local sum
    float tmp = 0.0;

    //for each filter weight
    for (int y = -kerRad; y <= kerRad; ++y){
      for (int x = -kerRad; x <= kerRad; ++x){
          if ((i+x < 0) || (j+y < 0) || (i+x >= Nx) || (j+y >= Ny)) continue;
          tmp +=
            (filt[x+kerRad + M*(y+kerRad)] * data[idx+x][idy+y]);
        }
    }

    //store result to global memory
    result[i + Nx*j] = tmp;
}

static void GPU_convolution_shared(std::vector<short> image, std::vector<short> &result,
  int Nx, int Ny, int filterChoice){
/* This function implements the call to the CUDA shared memory convolution */

  // First we obtain the kernel size and radius depending on the coice
  int M;
  switch (filterChoice) {
    case gaussian:
      M = kerSizeGauss;
      break;
    case laplacian:
      M = kerSizeLaplacian;
      break;
  }

  // Create the device copies for the GPU and copy them to the device
  int sizeImg = Nx*Ny * sizeof(short);
  int sizeRes = Nx*Ny * sizeof(short);
  int sizeFilt = M*M * sizeof(float);
  short *d_img, *d_res;
  float *d_filt;

  cudaMalloc((void **)&d_img, sizeImg);
  cudaMalloc((void **)&d_res, sizeRes);
  cudaMalloc((void **)&d_filt, sizeFilt);

  cudaMemcpy(d_img, &image[0], sizeImg, cudaMemcpyHostToDevice);
  dim3 threads(blockSizeX, blockSizeY);
  dim3 blocks(int(ceilf(Nx/(float)blockSizeX)), int(ceilf(Ny/(float)blockSizeY)));

  switch (filterChoice) {
    case gaussian:
      cudaMemcpy(d_filt, &gaussFilt[0], sizeFilt, cudaMemcpyHostToDevice);
      CUDA_convolution_shared_gaussian<<<blocks,threads>>>(d_img, d_res, d_filt, Nx, Ny);
      break;
    case laplacian:
      cudaMemcpy(d_filt, &LaplacianFilt[0], sizeFilt, cudaMemcpyHostToDevice);
      CUDA_convolution_shared_laplacian<<<blocks,threads>>>(d_img, d_res, d_filt, Nx, Ny);
      break;
   }

  // Copy result back to host
  cudaMemcpy(&result[0], d_res, sizeRes, cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_img); cudaFree(d_filt); cudaFree(d_res);
}


#endif
