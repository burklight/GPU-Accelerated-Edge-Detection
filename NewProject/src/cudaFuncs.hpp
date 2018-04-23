#include <iostream>
#include <vector>

#define blockSizeX 16
#define blockSizeY 16

const std::vector<double> gaussFilt =
{
  2 / 159.0,  4 / 159.0,  5 / 159.0,  4 / 159.0,  2 / 159.0,
  4 / 159.0,  9 / 159.0, 12 / 159.0,  9 / 159.0,  4 / 159.0,
  5 / 159.0, 12 / 159.0, 15 / 159.0, 12 / 159.0,  5 / 159.0,
  4 / 159.0,  9 / 159.0, 12 / 159.0,  9 / 159.0,  4 / 159.0,
  2 / 159.0,  4 / 159.0,  5 / 159.0,  4 / 159.0,  2 / 159.0
};

const std::vector<double> soebelFilt =
{
  0,  1,  0,
  1, -4,  1,
  0,  1,  0
};

const std::vector<double> edecFilt =
{
  -1, -1, -1,
  -1,  4, -1,
  -1, -1, -1
};

/* CPU convolution of the image and the filter selected: (1,2,3) -> (gauss, soebel, edge detect) */
void CPUconv(std::vector<short> image,
             std::vector<short> &result,
             unsigned int N,
             unsigned int filt){

  int imgW = N;
  int imgH = N;
  int kerRad;
  switch (filt) {
    case 1: kerRad = 2;
            break;
    case 2: kerRad = 1;
            break;
    case 3: kerRad = 1;
            break;
  }
  int M = 2*kerRad + 1;
  double tmp = 0.0;
  for (int j = 0; j < imgH; ++j){
    for (int i = 0; i < imgW; ++i){
      result[i + imgW*j] = 0;
      for (int y = -kerRad; y <= kerRad; ++y){
        for (int x = -kerRad; x <= kerRad; ++x){
          if ((i+x < 0) || (j+y < 0) || (i+x >= imgW) || (j+y >= imgH)) continue;
          switch (filt) {
            case 1: tmp +=
              (gaussFilt[x+kerRad + M*(y+kerRad)] * (double) image[(i+x) + imgW*(j+y)]);
              break;
            case 2: tmp +=
              (soebelFilt[x+kerRad + M*(y+kerRad)] * image[(i+x) + imgW*(j+y)]);
              break;
            case 3: tmp +=
              (edecFilt[x+kerRad + M*(y+kerRad)] * image[(i+x) + imgW*(j+y)]);
              break;
            }
        }
      }
      result[i + imgW*j] = (short) tmp;
      tmp = 0.0;
    }
  }
}

/* Naive implementation of the convolution in CUDA */
__global__ void CUconv_naive(short *image, short *result,
  double *filt, int N, int kerRad) {
    int M = 2*kerRad + 1;

    //for each pixel in the result
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //thread-local register to hold local sum
    double tmp = 0.0;

    //for each filter weight
    for (int y = -kerRad; y <= kerRad; ++y){
      for (int x = -kerRad; x <= kerRad; ++x){
          tmp +=
            (filt[x+kerRad + M*(y+kerRad)] * image[(i+x) + N*(j+y)]);
        }
    }

    //store result to global memory
    result[i + N*j] = tmp;

}

/* GPU naive convolution of the image and the filter selected: (1,2,3) -> (gauss, soebel, edge detect) */
void GPUconv_naive(std::vector<short> image,
                   std::vector<short> &result,
                   unsigned int N,
                   unsigned int filt){

   int kerRad;
   switch (filt) {
     case 1: kerRad = 2;
             break;
     case 2: kerRad = 1;
             break;
     case 3: kerRad = 1;
             break;
    }
    int M = kerRad*2 + 1;

    // Create the device copies for the GPU and copy them to the device
    int sizeImg = N*N * sizeof(short);
    int sizeRes = N*N * sizeof(short);
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
      case 2: cudaMemcpy(d_filt, &soebelFilt[0], sizeFilt, cudaMemcpyHostToDevice);
              break;
      case 3: cudaMemcpy(d_filt, &edecFilt[0], sizeFilt, cudaMemcpyHostToDevice);
              break;
     }

    // Launch conv() kernel on GPU with N blocks
    dim3 threads(blockSizeX, blockSizeY);
    dim3 blocks(int(ceilf(N/(float)blockSizeX)), int(ceilf(N/(float)blockSizeY)));
    std::cout << int(ceilf(N/(float)blockSizeX)) << std::endl;
    CUconv_naive<<<blocks,threads>>>(d_img, d_res, d_filt, N, kerRad);

    // Copy result back to host
    cudaMemcpy(&result[0], d_res, sizeRes, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_img); cudaFree(d_filt); cudaFree(d_res);

}
