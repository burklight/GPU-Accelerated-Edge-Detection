#include <iostream>
#include <vector>
#include <string>
#include "../src/loadImage.hpp"
#include "../src/cpuFuncs.hpp"
#include "../src/gpuFuncs.hpp"
#include "../src/constants.hpp"

int main(){

  // Declaration of necessary variables
  int Nx[3] = {512, 1024, 2048};
  int Ny[3] = {512, 1024, 2048};
  clock_t begin, end;
  double elapsed;
  int manyTimes = 1;
  std::string imgPath;

  // We make a speed test of the convolutons for each image size
  for (int s = 2; s < 3; s++){

    switch (s) {
      case 0:
        imgPath = "../data/512/img1.txt";
        break;
      case 1:
        imgPath = "../data/1024/img1.txt";
        break;
      case 2:
        imgPath = "../data/2048/img1.txt";
    }

    std::cout << "***********************************************" << std::endl;
    std::cout << Nx[s] << "x" << Ny[s] << std::endl;
    std::cout << "***********************************************" << std::endl;

    std::vector<short> image= loadImage(imgPath,Nx[s],Ny[s]);
    std::vector<short> result(Nx[s]*Ny[s]);


    begin = clock();
    for (int i = 0; i < manyTimes; i++){
      GPU_convolution_tiling(image, result, Nx[s], Ny[s], gaussian);
    }
    end = clock();
    elapsed = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "GPU 1x"<< tilingFactor << " tiling:  5x5 convolution on " << manyTimes << " " << Nx[s] << "x" << Ny[s]
      <<  " images: " << elapsed << std::endl;

/*
    begin = clock();
    for (int i = 0; i < manyTimes; i++){
      GPU_convolution_tiling(image, result, Nx[s], Ny[s], laplacian);
    }
    end = clock();
    elapsed = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "GPU 1x"<<tilingFactor<<" Tiling:  3x3 convolution on " << manyTimes << " " << Nx[s] << "x" << Ny[s]
      <<  " images: " << elapsed << std::endl;
*/
  }

  return 0;
}
