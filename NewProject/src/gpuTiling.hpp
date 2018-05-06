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
    int kerRad = kerRadGauss;

    //for each pixel in the result
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = tilingFactor * blockIdx.x * blockSizeX + threadIdx.x;

    if (threadIdx.x == 0 && threadIdx.y ==0) printf("Tiling factor for block (%d,%d) at pixel (%d,%d): %d\n",blockIdx.x,blockIdx.y,col,row,min(int(ceil((Nx-col)/((float)blockSizeX))),tilingFactor));
    int myTilingFactor = min(int(ceil((Nx-col)/((float)blockSizeX))),tilingFactor);

    //some data is shared between the threads in a block
    __shared__ short data[tilingFactor*blockSizeX+kerSizeGauss-1][blockSizeY];

    //position of the thread (X-axis) in the shared data
    int idx = threadIdx.x + kerRad;
    int idy = threadIdx.y;

    //position of the tread within the block
    int thx = threadIdx.x;

    //each thread loads its own position
    for (int tileIdx = 0; tileIdx < myTilingFactor; ++tileIdx) {
        //if (tileIdx == tilingFactor-1 && blockIdx.y == 0) printf("Truthvalue1: %d\n",col+tileIdx*blockSizeX >= Nx || row >= Ny);
        if (col+tileIdx*blockSizeX >= Nx || row >= Ny) {
            data[idx+tileIdx*blockSizeX][idy] = 0;
            //printf("Location1: %d\n",col+tileIdx*blockSizeX);
        }
        else data[idx+tileIdx*blockSizeX][idy] = image[col+tileIdx*blockSizeX+Nx*row];
    }

    //load left apron
    if (thx-kerRad < 0){
      if (col-kerRad < 0) data[idx-kerRad][idy] = 0;
      else data[idx-kerRad][idy] = image[(col-kerRad)+Nx*row];
    }

    //load right apron
    //TODO: FIX THIS SO IT INCORPORATES tileIdx
    int tilingOffset = (myTilingFactor-1)*blockSizeX;
    if (thx+tilingOffset+kerRad >= myTilingFactor*blockSizeX){
        //if (tileIdx == myTilingFactor-1 && blockIdx.y == 0) printf("Truthvalue1: %d",col+tilingOffset+kerRad >= Nx || row >= Ny);
      if (col+tilingOffset+kerRad >= Nx || row >= Ny) {
          data[idx+tilingOffset+kerRad][idy] = 0;
      }
      else data[idx+tilingOffset+kerRad][idy] = image[(col+tilingOffset+kerRad)+Nx*row];
    }

    //make sure that all the data is available for all the threads
    __syncthreads();

    //thread-local register to hold local sum
    float tmp[tilingFactor] = {0.0};

    //for each filter weight
    for (int x = -kerRad; x <= kerRad; ++x){
        for (int tileIdx = 0; tileIdx < myTilingFactor; ++tileIdx) {
            if ((idx+tileIdx*blockSizeX+x < 0) || (idx+tileIdx*blockSizeX+x >= myTilingFactor*blockSizeX+kerSizeGauss-1)/* || col+tileIdx*blockSizeX+x >=Nx*/ ) {
                //if (col+x+tileIdx*blockSizeX >= Nx) printf("Location3: %d > %d is %d\n",col+x+tileIdx*blockSizeX, Nx, col+x+tileIdx*blockSizeX > Nx);
                continue;
            }
            tmp[tileIdx] += (gaussianKernelLin[x+kerRad] * data[idx+tileIdx*blockSizeX+x][idy]);
            ///*if (col + tileIdx*blockSizeX > 700)*/ printf("tmp content at %d: %f",col + tileIdx*blockSizeX,tmp[tileIdx]);
        }
    }

    //store result to global memory
    for (int tileIdx = 0; tileIdx < myTilingFactor; ++tileIdx) {
        result[col+tileIdx*blockSizeX + Nx*row] = int(round(tmp[tileIdx]));
    }
}

__global__ void CUDA_convolution_col_tile(short *image, short *result,
  int Nx, int Ny) {
/* This function implements the row convolution in CUDA for Gaussian kernel */

    int kerRad = kerRadGauss;

    //for each pixel in the result
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = tilingFactor * blockIdx.x * blockSizeX + threadIdx.x;

    int myTilingFactor = min(int(ceil((Nx-col)/((float)blockSizeX))),tilingFactor);

    //some data is shared between the threads in a block
    __shared__ short data[tilingFactor*blockSizeX][blockSizeY+kerSizeGauss-1];

    //position of the thread (Y-axis) in the shared data
    int idx = threadIdx.x;
    int idy = threadIdx.y + kerRad;

    //position of the tread within the block
    int thy = threadIdx.y;

    //each thread loads its own position
    for (int tileIdx = 0; tileIdx < myTilingFactor; ++tileIdx) {
        //if (tileIdx == myTilingFactor-1 && blockIdx.y == 0) printf("Truthvalue1: %d\n",col+tileIdx*blockSizeX >= Nx);

        if (col+tileIdx*blockSizeX >= Nx) data[idx+tileIdx*blockSizeX][idy] = 0;
        else data[idx+tileIdx*blockSizeX][idy] = image[col+tileIdx*blockSizeX+Nx*row];
    }

    //load top apron
    if (thy-kerRad < 0){
        for (int tileIdx = 0; tileIdx < myTilingFactor; ++tileIdx) {
            if (row-kerRad < 0 || col+tileIdx*blockSizeX >= Nx) {
                data[idx+tileIdx*blockSizeX][idy-kerRad] = 0;
                //if (col+tileIdx*blockSizeX >= Nx) printf("Location4: %d\n",col+tileIdx*blockSizeX);
            }
            else data[idx+tileIdx*blockSizeX][idy-kerRad] = image[col+tileIdx*blockSizeX+Nx*(row-kerRad)];
        }
    }

    //load bottom apron
    if (thy+kerRad >= blockDim.y){
        for (int tileIdx = 0; tileIdx < myTilingFactor; ++tileIdx) {
            if (row+kerRad >= Ny || col+tileIdx*blockSizeX >= Nx) {
                //if (col+tileIdx*blockSizeX >= Nx) printf("Location5: %d\n",col+tileIdx*blockSizeX);
                data[idx+tileIdx*blockSizeX][idy+kerRad] = 0;
            }
            else data[idx+tileIdx*blockSizeX][idy+kerRad] = image[col+tileIdx*blockSizeX+Nx*(row+kerRad)];
        }
    }

    //make sure that all the data is available for all the threads
    __syncthreads();

    //thread-local register to hold local sum
    float tmp[tilingFactor] = {0.0};

    //for each filter weight
    for (int y = -kerRad; y <= kerRad; ++y){
        for (int tileIdx = 0; tileIdx < myTilingFactor; ++tileIdx) {
            if ((row+y < 0) || (row+y >= Ny)) continue;
            tmp[tileIdx] += (gaussianKernelLin[y+kerRad] * data[idx+tileIdx*blockSizeX][idy+y]);
        }
    }

    //store result to global memory
    for (int tileIdx = 0; tileIdx < myTilingFactor; ++tileIdx) {
        result[col+tileIdx*blockSizeX + Nx*row] = int(round(tmp[tileIdx]));
    }

}

__global__ void CUDA_convolution_tiling_laplacian(short *image, short *result,
  int Nx, int Ny) {
/* This function implements the shared memory and constant kernels convolution in CUDA for Laplacian */
    int kerRad = kerRadLaplacian;
    int M = kerSizeLaplacian;

    //for each pixel in the result
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = tilingFactor * blockIdx.x * blockSizeX + threadIdx.x;

    int myTilingFactor = min(int(ceil((Nx-col)/((float)blockSizeX))),tilingFactor);

    //some data is shared between the threads in a block
    __shared__ short data[blockSizeX*tilingFactor+kerSizeLaplacian-1][blockSizeY+kerSizeLaplacian-1];

    //position of the thread in the shared data
    int idx = threadIdx.x + kerRad;
    int idy = threadIdx.y + kerRad;

    //position of the tread within the block
    int thx = threadIdx.x;
    int thy = threadIdx.y;

    //each thread loads its own position
    for (int tileIdx = 0; tileIdx < myTilingFactor; ++tileIdx) {
      data[idx+blockSizeX*tileIdx][idy] = image[(col+blockSizeX*tileIdx)+Nx*row];
    }

    //load left apron
    if (thx-kerRad < 0){
      if (col-kerRad < 0)
        data[idx-kerRad][idy] = 0;
      else data[idx-kerRad][idy] = image[(col-kerRad)+Nx*row];
    }

    //load right apron
    int tilingOffset = (myTilingFactor-1)*blockSizeX;
    if (thx+tilingOffset+kerRad >= myTilingFactor*blockSizeX){
      if (col+tilingOffset+kerRad >= Nx) {
          data[idx+tilingOffset+kerRad][idy] = 0;
          //printf("Location6: %d\n",col+tilingOffset+kerRad);
      }
      else data[idx+tilingOffset+kerRad][idy] = image[(col+tilingOffset+kerRad)+Nx*row];
    }

    //load top apron
    if (thy-kerRad < 0){
      for (int tileIdx = 0; tileIdx < myTilingFactor; ++tileIdx) {
        int offset = tileIdx*blockSizeX;
        if (row-kerRad < 0 || col+offset >= Nx) {
          data[idx+offset][idy-kerRad] = 0;
          //if (col+offset >= Nx) printf("Location7: %d\n",col+offset);
        }
        else data[idx+offset][idy-kerRad] = image[col+offset+Nx*(row-kerRad)];
      }
    }

    //load bottom apron
    if (thy+kerRad >= blockDim.y){
      for (int tileIdx = 0; tileIdx < myTilingFactor; ++tileIdx) {
        int offset = tileIdx*blockSizeX;
        if (col+kerRad >= Ny || col+offset >= Nx) {
          data[idx+offset][idy+kerRad] = 0;
          //if (col+offset >= Nx) printf("Location8: %d\n",col+offset);
        }
        else data[idx+offset][idy+kerRad] = image[col+offset+Nx*(row+kerRad)];
      }
    }

    //load top-left apron
    if ((thx-kerRad < 0) && (thy-kerRad < 0)){
      if ((col-kerRad < 0) || (row-kerRad < 0)) data[idx-kerRad][idy-kerRad] = 0;
      else data[idx-kerRad][idy-kerRad] = image[(col-kerRad)+Nx*(row-kerRad)];
    }

    //load top-right apron
    if ((thx+tilingOffset+kerRad >= myTilingFactor*blockSizeX) && (thy-kerRad < 0)){
      if ((col+tilingOffset+kerRad >= Nx) || (row-kerRad < 0)) {
          data[idx+tilingOffset+kerRad][idy-kerRad] = 0;
          //if (col+tilingOffset+kerRad >= Nx) printf("Location9: %d\n",col+tilingOffset+kerRad);
      }
      else data[idx+tilingOffset+kerRad][idy-kerRad] = image[(col+tilingOffset+kerRad)+Nx*(row-kerRad)];
    }

    //load bottom-left apron
    if ((thx-kerRad < 0) && (thy+kerRad >= blockDim.y)){
      if ((col-kerRad < 0) || (row+kerRad >= Ny)) data[idx-kerRad][idy+kerRad] = 0;
      else data[idx-kerRad][idy+kerRad] = image[(col-kerRad)+Nx*(row+kerRad)];
    }

    //load bottom-right apron
    if ((thx+tilingOffset+kerRad  >= myTilingFactor*blockSizeX) && (thy+kerRad >= blockDim.y)){
      if ((col+tilingOffset+kerRad >= Nx) || (row+kerRad >= Ny)) {
          data[idx+tilingOffset+kerRad][idy+kerRad] = 0;
          //if (col+tilingOffset+kerRad >= Nx) printf("Location10: %d\n",col+tilingOffset+kerRad);
      }
      else data[idx+tilingOffset+kerRad][idy+kerRad] = image[(col+tilingOffset+kerRad)+Nx*(row+kerRad)];
    }

    //make sure that all the data is available for all the threads
    __syncthreads();
    //printf("4\n");
    //thread-local register to hold local sum
    float tmp[tilingFactor] = {0.0};


    //for each filter weight
    for (int y = -kerRad; y <= kerRad; ++y){
      for (int x = -kerRad; x <= kerRad; ++x){
        for (int tileIdx = 0; tileIdx < myTilingFactor; ++tileIdx) {
          int offset = tileIdx*blockSizeX;
          if ((col+x+offset < 0) || (row+y < 0) || (col+x+offset >= Nx) || (row+y >= Ny)) {
              //if (col+x+tileLoc >= Nx) printf("Location11: %d\n",col+x+tileLoc);
              continue;
          }
          tmp[tileIdx] +=
            (laplacianKernel[x+kerRad + M*(y+kerRad)] * data[idx+x+offset][idy+y]);
        }

      }
    }

    //store result in global memory
    for (int tileIdx = 0; tileIdx < myTilingFactor; ++tileIdx) {
      int offset = tileIdx*blockSizeX;
      if (col+offset >= Nx) {
          //printf("Location12: %d\n",col+tileLoc);
          break;
      }
      result[(col+offset) + Nx*row] = tmp[tileIdx];
    }
}

static void GPU_convolution_tiling(std::vector<short> image, std::vector<short> &result,
  int Nx, int Ny, int filterChoice){
/* This function implements the call to the CUDA shared memory and constant kernels convolution */

  // First we obtain the kernel size and radius depending on the coice
  int sizeFilt = sizeof(float);
  switch (filterChoice) {
    case gaussian:
      sizeFilt *= kerSizeGauss;
      break;
    case laplacian:
      sizeFilt *= kerSizeLaplacian*kerSizeLaplacian;
      break;
  }
  /*
  int blocksPerSM = 2;

  int numShorts = keplerSharedMemSize / (sizeof(short) * blocksPerSM);

  int gausApronSizes = 2*kerRadGauss*(blockSizeY+2*kerRadGauss);
  int lapApronSizes = 2*kerRadLaplacian*(blockSizeY+2*kerRadLaplacian);

  int numGausShorts = numShorts - gausApronSizes;
  int numLapShorts = numShorts - lapApronSizes;

  int gausTileSize = blockSizeX*(blockSizeY+2*kerRadGauss);
  int lapTileSize = blockSizeX*(blockSizeY+2*kerRadLaplacian);

  int gausTileFact = numGausShorts/gausTileSize;
  int lapTileFact = numLapShorts/lapTileSize;

  printf("Best laplacian tiling factor: %d\n",lapTileFact);
  printf("Best gaussian tiling factor: %d\n",gausTileFact);
  */

  // Create the device copies for the GPU and copy them to the device
  int sizeImg = Nx*Ny * sizeof(short);
  int sizeTmp = Nx*Ny * sizeof(short);
  int sizeRes = Nx*Ny * sizeof(short);
  short *d_img, *d_tmp, *d_res;

  cudaMalloc((void **)&d_img, sizeImg);
  cudaMalloc((void **)&d_tmp, sizeTmp);
  cudaMalloc((void **)&d_res, sizeRes);

  cudaMemcpy(d_img, &image[0], sizeImg, cudaMemcpyHostToDevice);
  dim3 threads(blockSizeX, blockSizeY);
  printf("Num blocks per row of tiles: %d\n", int(ceil(Nx/((float)(blockSizeX*tilingFactor)))));
  dim3 blocks(int(ceil(Nx/((float)(blockSizeX*tilingFactor)))), int(ceilf(Ny/(float)blockSizeY)));

  switch (filterChoice) {
    case gaussian:
      cudaMemcpyToSymbol(gaussianKernelLin, &gaussFiltLin[0], sizeFilt);
      CUDA_convolution_row_tile<<<blocks,threads>>>(d_img, d_tmp, Nx, Ny);
      CUDA_convolution_col_tile<<<blocks,threads>>>(d_tmp, d_res, Nx, Ny);
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
