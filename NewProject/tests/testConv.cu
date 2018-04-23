#include <iostream>
#include <vector>
#include "../src/loadImage.hpp"
#include "../src/cudaFuncs.hpp"

int main(){

  int N = 2048;
  std::vector<short> ourImage = loadImage("../data/2048/img5.txt",N);
  std::vector<short> filtered(N*N);
  std::vector<short> result(N*N);

  clock_t begin, end;
  double elapsed;
  begin = clock();
  GPUconv_naive(ourImage, filtered, N, 1);
  GPUconv_naive(filtered, result, N, 2);
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
  for(int i = 0; i < N; ++i){
    for(int j = 0; j < N-1; ++j){
      fprintf(fp, "%d,", result[j + N*i]);
    }
    fprintf(fp,"%d\n", result[(N-1) + N*i]);
  }
  fclose(fp);

  return 0;
}
