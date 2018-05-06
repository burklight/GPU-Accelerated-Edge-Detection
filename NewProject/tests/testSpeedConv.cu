#include <iostream>
#include <vector>
#include <string>
#include "../src/loadImage.hpp"
#include "../src/cpuFuncs.hpp"
#include "../src/gpuFuncs.hpp"
#include "../src/constants.hpp"

int main(){

  // Declaration of necessary variable
  clock_t begin, end;
  double elapsed;
  int manyTimes = 100;

  // We check the speed of the convolution for different image sizes (for theoretical reasons)
  FILE *fp_img_size_CPU, *fp_img_size_GPU;
  fp_img_size_CPU = fopen("./tests/speed_conv_img_size_CPU.txt","wb");
  fp_img_size_GPU = fopen("./tests/speed_conv_img_size_GPU.txt","wb");
  for (int i = 256; i <= 4096; i*=2){
    // We generate a squared image with sizes ranging from 256 to 4096
    std::vector<short> tmpImage(i*i);
    std::vector<short> tmpResult(i*i);

    begin = clock();
    for (int j = 0; j < manyTimes; j++){
      CPU_convolution(tmpImage, tmpResult, i, i, gaussian);
    }
    end = clock();
    elapsed = double(end - begin) / CLOCKS_PER_SEC;
    fprintf(fp_img_size_CPU, "%.5f,", elapsed);

    // We apply the different algorithms and store the time elapsd
    begin = clock();
    for (int j = 0; j < manyTimes; j++){
      GPU_convolution_naive(tmpImage, tmpResult, i, i, gaussian);
    }
    end = clock();
    elapsed = double(end - begin) / CLOCKS_PER_SEC;
    fprintf(fp_img_size_GPU, "%.5f,", elapsed);


    begin = clock();
    for (int j = 0; j < manyTimes; j++){
      GPU_convolution_shared(tmpImage, tmpResult, i, i, gaussian);
    }
    end = clock();
    elapsed = double(end - begin) / CLOCKS_PER_SEC;
    fprintf(fp_img_size_GPU, "%.5f,", elapsed);

    begin = clock();
    for (int j = 0; j < manyTimes; j++){
      GPU_convolution_const(tmpImage, tmpResult, i, i, gaussian);
    }
    end = clock();
    elapsed = double(end - begin) / CLOCKS_PER_SEC;
    fprintf(fp_img_size_GPU, "%.5f,", elapsed);

    begin = clock();
    for (int j = 0; j < manyTimes; j++){
      GPU_convolution_sep(tmpImage, tmpResult, i, i);
    }
    end = clock();
    elapsed = double(end - begin) / CLOCKS_PER_SEC;
    fprintf(fp_img_size_GPU, "%.5f,", elapsed);

    begin = clock();
    for (int j = 0; j < manyTimes; j++){
      GPU_convolution_tiling(tmpImage, tmpResult, i, i, gaussian);
    }
    end = clock();
    elapsed = double(end - begin) / CLOCKS_PER_SEC;
    fprintf(fp_img_size_GPU, "%.5f,", elapsed);

    begin = clock();
    for (int j = 0; j < manyTimes; j++){
      CPU_convolution(tmpImage, tmpResult, i, i, laplacian);
    }
    end = clock();
    elapsed = double(end - begin) / CLOCKS_PER_SEC;
      fprintf(fp_img_size_CPU, "%.5f,", elapsed);

    begin = clock();
    for (int j = 0; j < manyTimes; j++){
      GPU_convolution_naive(tmpImage, tmpResult, i, i, laplacian);
    }
    end = clock();
    elapsed = double(end - begin) / CLOCKS_PER_SEC;
    fprintf(fp_img_size_GPU, "%.5f,", elapsed);

    begin = clock();
    for (int j = 0; j < manyTimes; j++){
      GPU_convolution_shared(tmpImage, tmpResult, i, i, laplacian);
    }
    end = clock();
    elapsed = double(end - begin) / CLOCKS_PER_SEC;
    fprintf(fp_img_size_GPU, "%.5f,", elapsed);

    begin = clock();
    for (int j = 0; j < manyTimes; j++){
      GPU_convolution_const(tmpImage, tmpResult, i, i, laplacian);
    }
    end = clock();
    elapsed = double(end - begin) / CLOCKS_PER_SEC;
    fprintf(fp_img_size_GPU, "%.5f,", elapsed);

    begin = clock();
    for (int j = 0; j < manyTimes; j++){
      GPU_convolution_tiling(tmpImage, tmpResult, i, i, laplacian);
    }
    end = clock();
    elapsed = double(end - begin) / CLOCKS_PER_SEC;
    fprintf(fp_img_size_GPU, "%.5f\n", elapsed);

    std::cout << "Size: " << i << std::endl;

  }

  fclose(fp_img_size_GPU);
  fclose(fp_img_size_CPU);

}
