#include <iostream>
#include <vector>
#include "../src/loadImage.hpp"
#include "../src/cudaFuncs.hpp"

int main(){

  int Nx = 2048;
  int Ny = 2048;
  std::vector<short> ourImage = loadImage("../data/2048/img5.txt",Nx,Ny);
  std::vector<short> filtered(Nx*Ny);
  std::vector<short> result(Nx*Ny);

  clock_t begin, end;
  double elapsed;
  begin = clock();
  //GPUconv_naive(ourImage, filtered, Nx, Ny, 1);
  CPUconv(filtered, result, Nx, Ny, 2);
  end = clock();
  elapsed = double(end - begin) / CLOCKS_PER_SEC;
  std::cout << "Elapsed time: " << elapsed << std::endl;

  /*
  for (int i = 5; i < 10; i++) std::cout << ourImage[i + 512*5] << std::endl;
  for (int i = 5; i < 10; i++) std::cout << filtered[i + 512*5] << std::endl;
  for (int i = 5; i < 10; i++) std::cout << result[i + 512*5] << std::endl;
  */

  FILE *fp;
  fp = fopen("edgeEx.txt","wb");
  for(int i = 0; i < Ny; ++i){
    for(int j = 0; j < Nx-1; ++j){
      fprintf(fp, "%d,", result[j + Nx*i]);
    }
    fprintf(fp,"%d\n", result[(Nx-1) + Nx*i]);
  }
  fclose(fp);

  return 0;
}
