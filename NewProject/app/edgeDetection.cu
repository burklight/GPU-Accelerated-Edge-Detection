#include <iostream>
#include <vector>
#include "../src/loadImage.hpp"
#include "../src/gpuFuncs.hpp"
#include "../src/cpuFuncs.hpp"

int main(){

  /* We get one of the larger images */
  int Nx, Ny;
  std::cin >> Nx >> Ny;
  std::vector<short> ourImage = loadImage("tmp_img.txt",Nx,Ny);
  std::vector<short> result(Nx*Ny);

  /* We apply the edge detection algorithm */
  CPU_edgeDetection(ourImage, result, Nx, Ny);

  /* We write it in a file to later observe it is correct */
  FILE *fp;
  fp = fopen("edge_detect.txt","wb");

  for(int i = 0; i < Ny; ++i){
    for(int j = 0; j < Nx-1; ++j){
      fprintf(fp, "%d,", result[j + Nx*i]);
    }
    fprintf(fp,"%d\n", result[(Nx-1) + Nx*i]);
  }
  fclose(fp);

  return 0;
}
