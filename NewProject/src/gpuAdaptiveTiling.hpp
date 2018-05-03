#ifndef GPU_TILING_H
#define GPU_TILING_H

#include "constants.hpp"

/****************************Function definition*******************************/
__global__ void CUDA_convolution_tiling_gaussian(short *image, short *result,
  int Nx, int Ny);

__global__ void CUDA_convolution_tiling_laplacian(short *image, short *result,
  int Nx, int Ny);

__global__ void CUDA_convolution_row_tile(short *image, short *result,
  int Nx, int Ny);

__global__ void CUDA_convolution_col_tile(short *image, short *result,
  int Nx, int Ny);

static void GPU_convolution_tiling(std::vector<short> image, std::vector<short> &result,
  int Nx, int Ny, int filterChoice);


/****************************Function implementation***************************/
__global__ void CUDA_convolution_row_tile(short *image, short *result,
  int Nx, int Ny) {
/* This function implements the row convolution in CUDA for Gaussian kernel */
  /*
    int kerRad = kerRadGauss;

    //for each pixel in the result
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //some data is shared between the threads in a block
    __shared__ short data[tilingFactor*blockSizeX+kerSizeGauss-1][blockSizeY];
    
    //position of the thread (X-axis) in the shared data
    int idx = threadIdx.x + kerRad;
    int idy = threadIdx.y;

    //position of the tread within the block
    int thx = threadIdx.x;

    //each thread loads its own position
    for (int tileIdx = 0; tileIdx < tilingFactor; ++tileIdx) {
        if (col+tileIdx*blockDim.x >= Nx) data[idx+tileIdx*blockDim.x][idy] = 0;
        else data[idx+tileIdx*blockDim.x][idy] = image[col+tileIdx*blockDim.x+Nx*row];
    }

    //load left apron
    if (thx-kerRad < 0){
      if (col-kerRad < 0) data[idx-kerRad][idy] = 0;
      else data[idx-kerRad][idy] = image[(col-kerRad)+Nx*row];
    }

    //load right apron
    int tilingOffset = (tilingFactor-1)*blockDim.x;
    if (thx+tilingOffset+kerRad >= tilingFactor*blockDim.x){
      if (col+tilingOffset+kerRad >= Nx) data[idx+tilingOffset+kerRad][idy] = 0;
      else data[idx+tilingOffset+kerRad][idy] = image[(col+tilingOffset+kerRad)+Nx*row];
    }

    //make sure that all the data is available for all the threads
    __syncthreads();

    //thread-local register to hold local sum
    float tmp[tilingFactor] = {};

    //for each filter weight
    for (int x = -kerRad; x <= kerRad; ++x){
        for (int tileIdx = 0; tileIdx < tilingFactor; ++tileIdx) {
            if ((col+x < 0) || (col+x+tileIdx*blockDim.x >= Nx)) continue;
            tmp[tileIdx] += (gaussianKernelLin[x+kerRad] * data[idx+tileIdx*blockDim.x+x][idy]);
        }
    }

    //store result to global memory
    for (int tileIdx = 0; tileIdx < tilingFactor; ++tileIdx) {
        result[col+tileIdx*blockDim.x + Nx*row] = int(round(tmp[tileIdx]));
    }
    */
}

__global__ void CUDA_convolution_col_tile(short *image, short *result,
  int Nx, int Ny) {
/* This function implements the row convolution in CUDA for Gaussian kernel */

    int kerRad = kerRadGauss;

    //for each pixel in the result
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //some data is shared between the threads in a block
    __shared__ short data[tilingFactor*blockSizeX][blockSizeY+kerSizeGauss-1];
    
    //position of the thread (Y-axis) in the shared data
    int idx = threadIdx.x;
    int idy = threadIdx.y + kerRad;

    //position of the tread within the block
    int thy = threadIdx.y;
    
    //each thread loads its own position
    for (int tileIdx = 0; tileIdx < tilingFactor; ++tileIdx) {
        if (col+tileIdx*blockDim.x >= Nx) data[idx+tileIdx*blockDim.x][idy] = 0;
        else data[idx+tileIdx*blockDim.x][idy] = image[col+tileIdx*blockDim.x+Nx*row];
    }
    
    //load top apron
    if (thy-kerRad < 0){
        for (int tileIdx = 0; tileIdx < tilingFactor; ++tileIdx) {
            if (row-kerRad < 0 || col+tileIdx*blockDim.x >= Nx) data[idx+tileIdx*blockDim.x][idy-kerRad] = 0;
            else data[idx+tileIdx*blockDim.x][idy-kerRad] = image[col+tileIdx*blockDim.x+Nx*(row-kerRad)];
        }
    }
    
    //load bottom apron
    if (thy+kerRad >= blockDim.y){
        for (int tileIdx = 0; tileIdx < tilingFactor; ++tileIdx) {
            if (row+kerRad >= Ny || col+tileIdx*blockDim.x >= Nx) data[idx+tileIdx*blockDim.x][idy+kerRad] = 0;
            else data[idx+tileIdx*blockDim.x][idy+kerRad] = image[col+tileIdx*blockDim.x+Nx*(row+kerRad)];
        }
    }

    //make sure that all the data is available for all the threads
    __syncthreads();
    
    //thread-local register to hold local sum
    float tmp[tilingFactor] = {};

    //for each filter weight
    for (int y = -kerRad; y <= kerRad; ++y){
        for (int tileIdx = 0; tileIdx < tilingFactor; ++tileIdx) {
            if ((row+y < 0) || (row+y >= Ny)) continue;
            tmp[tileIdx] += (gaussianKernelLin[y+kerRad] * data[idx+tileIdx*blockDim.x][idy+y]);
        }
    }

    //store result to global memory
    for (int tileIdx = 0; tileIdx < tilingFactor; ++tileIdx) {
        result[col+tileIdx*blockDim.x + Nx*row] = int(round(tmp[tileIdx]));
    }
    
}

__global__ void CUDA_convolution_tiling_laplacian(short *image, short *result, 
  int Nx, int Ny) {
/* This function implements the shared memory and constant kernels convolution in CUDA for Laplacian */
    int kerRad = kerRadLaplacian;
    int M = kerSizeLaplacian;

    //for each pixel in the result
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = tilingFactor * blockIdx.x * blockDim.x + threadIdx.x;

    //some data is shared between the threads in a block
    __shared__ short data[blockSizeX*tilingFactor+kerSizeLaplacian-1][blockSizeY+kerSizeLaplacian-1];

    //position of the thread in the shared data
    int idx = threadIdx.x + kerRad;
    int idy = threadIdx.y + kerRad;

    //position of the tread within the block
    int thx = threadIdx.x;
    int thy = threadIdx.y;

    //each thread loads its own position
    for (int tileIdx = 0; tileIdx < tilingFactor; ++tileIdx) {
      data[idx+blockDim.x*tileIdx][idy] = image[(col+blockDim.x*tileIdx)+Nx*row];  
    }

    //load left apron
    if (thx-kerRad < 0){
      if (col-kerRad < 0) 
        data[idx-kerRad][idy] = 0;
      else data[idx-kerRad][idy] = image[(col-kerRad)+Nx*row];
    }

    //load right apron
    int tilingOffset = (tilingFactor-1)*blockDim.x;
    if (thx+tilingOffset+kerRad >= tilingFactor*blockDim.x){
      if (col+tilingOffset+kerRad >= Nx)
        data[idx+tilingOffset+kerRad][idy] = 0;
      else data[idx+tilingOffset+kerRad][idy] = image[(col+tilingOffset+kerRad)+Nx*row];
    }

    //load top apron
    if (thy-kerRad < 0){
      for (int tileIdx = 0; tileIdx < tilingFactor; ++tileIdx) {
        int offset = tileIdx*blockDim.x;
        if (row-kerRad < 0 || col+offset >= Nx) 
          data[idx+offset][idy-kerRad] = 0;
        else data[idx+offset][idy-kerRad] = image[col+offset+Nx*(row-kerRad)];
      }
    }

    //load bottom apron
    if (thy+kerRad >= blockDim.y){
      for (int tileIdx = 0; tileIdx < tilingFactor; ++tileIdx) {
        int offset = tileIdx*blockDim.x;
        if (col+kerRad >= Ny || col+offset >= Nx) 
          data[idx+offset][idy+kerRad] = 0;
        else data[idx+offset][idy+kerRad] = image[col+offset+Nx*(row+kerRad)];
      }
    }

    //load top-left apron
    if ((thx-kerRad < 0) && (thy-kerRad < 0)){
      if ((col-kerRad < 0) || (row-kerRad < 0)) data[idx-kerRad][idy-kerRad] = 0;
      else data[idx-kerRad][idy-kerRad] = image[(col-kerRad)+Nx*(row-kerRad)];
    }

    //load top-right apron
    if ((thx+tilingOffset+kerRad >= tilingFactor*blockDim.x) && (thy-kerRad < 0)){
      if ((col+tilingOffset+kerRad >= Nx) || (row-kerRad < 0)) data[idx+tilingOffset+kerRad][idy-kerRad] = 0;
      else data[idx+tilingOffset+kerRad][idy-kerRad] = image[(col+tilingOffset+kerRad)+Nx*(row-kerRad)];
    }

    //load bottom-left apron
    if ((thx-kerRad < 0) && (thy+kerRad >= blockDim.y)){
      if ((col-kerRad < 0) || (row+kerRad >= Ny)) data[idx-kerRad][idy+kerRad] = 0;
      else data[idx-kerRad][idy+kerRad] = image[(col-kerRad)+Nx*(row+kerRad)];
    }

    //load bottom-right apron
    if ((thx+tilingOffset+kerRad  >= tilingFactor*blockDim.x) && (thy+kerRad >= blockDim.y)){
      if ((col+tilingOffset+kerRad >= Nx) || (row+kerRad >= Ny)) data[idx+tilingOffset+kerRad][idy+kerRad] = 0;
      else data[idx+tilingOffset+kerRad][idy+kerRad] = image[(col+tilingOffset+kerRad)+Nx*(row+kerRad)];
    }

    //make sure that all the data is available for all the threads
    __syncthreads();
    //printf("4\n");
    //thread-local register to hold local sum
    float tmp[tilingFactor];


    //for each filter weight
    for (int y = -kerRad; y <= kerRad; ++y){
      for (int x = -kerRad; x <= kerRad; ++x){
        for (int tileIdx = 0; tileIdx < tilingFactor; ++tileIdx) {
          int tileLoc = tileIdx*blockSizeX;
          if ((col+x+tileLoc < 0) || (row+y < 0) || (col+x+tileLoc >= Nx) || (row+y >= Ny)) continue;
          tmp[tileIdx] +=
            (laplacianKernel[x+kerRad + M*(y+kerRad)] * data[idx+x+tileLoc][idy+y]);
        }
         
      }
    }

    //store result in global memory
    for (int tileIdx = 0; tileIdx < tilingFactor; ++tileIdx) {
      int tileLoc = tileIdx*blockSizeX;
      if (col+tileLoc >= Nx) break;
      result[(col+tileLoc) + Nx*row] = tmp[tileIdx];
    }
}

static void GPU_convolution_tiling(std::vector<short> image, std::vector<short> &result,
  int Nx, int Ny, int filterChoice){
/* This function implements the call to the CUDA shared memory and constant kernels convolution */

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
  int sizeTmp = Nx*Ny * sizeof(short);
  int sizeRes = Nx*Ny * sizeof(short);
  int sizeFilt = M*M * sizeof(float);
  short *d_img, *d_tmp, *d_res;

  cudaMalloc((void **)&d_img, sizeImg);
  cudaMalloc((void **)&d_tmp, sizeTmp);
  cudaMalloc((void **)&d_res, sizeRes);

  cudaMemcpy(d_img, &image[0], sizeImg, cudaMemcpyHostToDevice);
  dim3 threads(blockSizeX, blockSizeY);
  dim3 blocks(int(ceilf(int(ceilf(Nx/(float)blockSizeX)/(float)tilingFactor))), int(ceilf(Ny/(float)blockSizeY)));

  switch (filterChoice) {
    case gaussian:
      cudaMemcpyToSymbol(gaussianKernelLin, &gaussFiltLin[0], sizeFilt);
      //CUDA_convolution_row_tile<<<blocks,threads>>>(d_img, d_tmp, Nx, Ny);
      //CUDA_convolution_col_tile<<<blocks,threads>>>(d_tmp, d_res, Nx, Ny);
      break;
    case laplacian:
      cudaMemcpyToSymbol(laplacianKernel, &LaplacianFilt[0], sizeFilt);
      CUDA_convolution_tiling_laplacian<<<blocks,threads>>>(d_img, d_res, Nx, Ny);
      break;
   }

  // Copy result back to host
  cudaMemcpy(&result[0], d_res, sizeRes, cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_img); cudaFree(d_res);
}


#endif
