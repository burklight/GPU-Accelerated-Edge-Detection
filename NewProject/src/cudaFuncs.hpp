#include <iostream>
#include <vector>

#define blockSizeX 16
#define blockSizeY 16
#define kerSizeGauss 5
#define kerSizeLaplacian 3
#define kerRadGauss 2
#define kerRadLaplacian 1

const std::vector<double> gaussFilt =
{
  2 / 159.0,  4 / 159.0,  5 / 159.0,  4 / 159.0,  2 / 159.0,
  4 / 159.0,  9 / 159.0, 12 / 159.0,  9 / 159.0,  4 / 159.0,
  5 / 159.0, 12 / 159.0, 15 / 159.0, 12 / 159.0,  5 / 159.0,
  4 / 159.0,  9 / 159.0, 12 / 159.0,  9 / 159.0,  4 / 159.0,
  2 / 159.0,  4 / 159.0,  5 / 159.0,  4 / 159.0,  2 / 159.0
};

const std::vector<double> LaplacianFilt =
{
  0,  1,  0,
  1, -4,  1,
  0,  1,  0
};

/* CPU convolution of the image and the filter selected: (1,2,3) -> (gauss, Laplacian, edge detect) */
void CPUconv(std::vector<short> image,
             std::vector<short> &result,
             unsigned int Nx, unsigned int Ny,
             unsigned int filt){

   int kerRad, M;
   switch (filt) {
     case 1: kerRad = kerRadGauss;
             M = kerSizeGauss;
             break;
     case 2: kerRad = kerRadLaplacian;
             M = kerSizeLaplacian;
             break;
    }

  double tmp = 0.0;
  for (int j = 0; j < Ny; ++j){
    for (int i = 0; i < Nx; ++i){
      result[i + Nx*j] = 0;
      for (int y = -kerRad; y <= kerRad; ++y){
        for (int x = -kerRad; x <= kerRad; ++x){
          if ((i+x < 0) || (j+y < 0) || (i+x >= Nx) || (j+y >= Ny)) continue;
          switch (filt) {
            case 1: tmp +=
              (gaussFilt[x+kerRad + M*(y+kerRad)] * (double) image[(i+x) + Nx*(j+y)]);
              break;
            case 2: tmp +=
              (LaplacianFilt[x+kerRad + M*(y+kerRad)] * image[(i+x) + Nx*(j+y)]);
              break;
            }
        }
      }
      result[i + Nx*j] = (short) tmp;
      tmp = 0.0;
    }
  }
}

/* Naive implementation of the convolution in CUDA */
__global__ void CUconv_naive(short *image, short *result,
  double *filt, int Nx, int Ny, int kerRad) {
    int M = 2*kerRad + 1;

    //for each pixel in the result
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //thread-local register to hold local sum
    double tmp = 0.0;

    //for each filter weight
    for (int y = -kerRad; y <= kerRad; ++y){
      for (int x = -kerRad; x <= kerRad; ++x){
          if ((i+x < 0) || (j+y < 0) || (i+x >= Nx) || (j+y >= Ny)) continue;
          tmp +=
            (filt[x+kerRad + M*(y+kerRad)] * image[(i+x) + Nx*(j+y)]);
        }
    }

    //store result to global memory
    result[i + Nx*j] = tmp;

}

/* GPU naive convolution of the image and the filter selected: (1,2,3) -> (gauss, Laplacian, edge detect) */
void GPUconv_naive(std::vector<short> image,
                   std::vector<short> &result,
                   unsigned int Nx, unsigned int Ny,
                   unsigned int filt){

   int kerRad, M;
   switch (filt) {
     case 1: kerRad = kerRadGauss;
             M = kerSizeGauss;
             break;
     case 2: kerRad = kerRadLaplacian;
             M = kerSizeLaplacian;
             break;
    }

    // Create the device copies for the GPU and copy them to the device
    int sizeImg = Nx*Ny * sizeof(short);
    int sizeRes = Nx*Ny * sizeof(short);
    int sizeFilt = M*M * sizeof(double);;
    short *d_img, *d_res;
    double *d_filt;

    cudaMalloc((void **)&d_img, sizeImg);
    cudaMalloc((void **)&d_res, sizeRes);
    cudaMalloc((void **)&d_filt, sizeFilt);

    cudaMemcpy(d_img, &image[0], sizeImg, cudaMemcpyHostToDevice);

    switch (filt) {
      case 1: cudaMemcpy(d_filt, &gaussFilt[0], sizeFilt, cudaMemcpyHostToDevice);
              break;
      case 2: cudaMemcpy(d_filt, &LaplacianFilt[0], sizeFilt, cudaMemcpyHostToDevice);
              break;
     }

    // Launch conv() kernel on GPU with N blocks
    dim3 threads(blockSizeX, blockSizeY);
    dim3 blocks(int(ceilf(Nx/(float)blockSizeX)), int(ceilf(Ny/(float)blockSizeY)));
    CUconv_naive<<<blocks,threads>>>(d_img, d_res, d_filt, Nx, Ny, kerRad);

    // Copy result back to host
    cudaMemcpy(&result[0], d_res, sizeRes, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_img); cudaFree(d_filt); cudaFree(d_res);
}

/* Shared Memory implementation of the convolution in CUDA */
__global__ void CUconv_shared_Laplacian(short *image, short *result,
  double *filt, int Nx, int Ny, int filtChoice) {

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
    double tmp = 0.0;

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



/* GPU shared memory convolution of the image and the filter selected: (1,2,3) -> (gauss, Laplacian, edge detect) */
void GPUconv_shared(std::vector<short> image,
                    std::vector<short> &result,
                    unsigned int Nx, unsigned int Ny,
                    unsigned int filt){

    int kerRad, M;
    switch (filt) {
      case 1: kerRad = kerRadGauss;
              M = kerSizeGauss;
              break;
      case 2: kerRad = kerRadLaplacian;
              M = kerSizeLaplacian;
              break;
     }

    // Create the device copies for the GPU and copy them to the device
    int sizeImg = Nx*Ny * sizeof(short);
    int sizeRes = Nx*Ny * sizeof(short);
    int sizeFilt = M*M * sizeof(double);;
    short *d_img, *d_res;
    double *d_filt;

    cudaMalloc((void **)&d_img, sizeImg);
    cudaMalloc((void **)&d_res, sizeRes);
    cudaMalloc((void **)&d_filt, sizeFilt);

    cudaMemcpy(d_img, &image[0], sizeImg, cudaMemcpyHostToDevice);
    dim3 threads(blockSizeX, blockSizeY);
    dim3 blocks(int(ceilf(Nx/(float)blockSizeX)), int(ceilf(Ny/(float)blockSizeY)));

    switch (filt) {
      case 1: cudaMemcpy(d_filt, &gaussFilt[0], sizeFilt, cudaMemcpyHostToDevice);
              // Launch conv() kernel on GPU with N blocks
              CUconv_naive<<<blocks,threads>>>(d_img, d_res, d_filt, Nx, Ny, filt);
              break;
      case 2: cudaMemcpy(d_filt, &LaplacianFilt[0], sizeFilt, cudaMemcpyHostToDevice);
              // Launch conv() kernel on GPU with N blocks
              CUconv_shared_Laplacian<<<blocks,threads>>>(d_img, d_res, d_filt, Nx, Ny, filt);
              break;
     }

    // Copy result back to host
    cudaMemcpy(&result[0], d_res, sizeRes, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_img); cudaFree(d_filt); cudaFree(d_res);
}
