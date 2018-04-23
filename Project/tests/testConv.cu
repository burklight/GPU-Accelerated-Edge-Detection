#include <iostream>
#include <cstdlib>
#include <vector>
#include <ctime>


#include "../mnist/readMnist.hpp"
#include "../src/utils.hpp"

using namespace std;

int main(){
  //vector<vector<double> > testImages = ReadTestImages(false);
  vector<double> filter(5*5,0.0);
  // Laplacian approximation for edge detection
  filter[0+3*1] = 1;
  filter[1+3*0] = 1;
  filter[2+3*1] = 1;
  filter[1+3*2] = 1;
  filter[1+3*1] = -4;

  vector<vector<double> > testImages(500, vector<double>(28*28,0.4));

  int N = 28;
  int M = 5;
  vector<double> result ((N-M+1)*(N-M+1),0.0);
  conv(testImages[15], filter, result, N, M, true);

  clock_t begin, end;
  double elapsed;
  begin = clock();
  for(unsigned int i = 0; i < 250; i++)
    conv(testImages[i], filter, result, N, M, true);
  end = clock();
  elapsed = double(end - begin) / CLOCKS_PER_SEC;
  cout << "Time for 10000 conv with GPU: " << elapsed << endl;

  begin = clock();
  for(unsigned int i = 0; i < 250; i++)
    conv(testImages[i], filter, result, N, M, false);
  end = clock();
  elapsed = double(end - begin) / CLOCKS_PER_SEC;
  cout << "Time for 10000 conv with CPU: " << elapsed << endl;


  /*double* tmp = &testImages[15][0];
  for (unsigned int i = 0; i < N; ++i){
    for (unsigned int j = 0; j < N; ++j){
      (tmp[j+N*i] > 0.0) ? cout << " " : cout << "*";
      cout << " ";
    }
    cout << endl;
  }

  for (unsigned int j = 0; j < N-M+1; ++j){
    for (unsigned int i = 0; i < N-M+1; ++i){
      //cout << result[i + (N-M+1)*j];
      (result[i+(N-M+1)*j] > 0.0) ? cout << " " : cout << "*";
      cout << " ";
    }
    cout << endl;
  }*/
  /*FILE *fp;
  fp = fopen("color.txt","wb");
  for (unsigned int i = 0; i < 26; ++i){
    for (unsigned int j = 0; j < 26; ++j){
      fprintf(fp, "%f ",result[i+26*i]);
    }
    fprintf(fp,"\n");
  }*/

  return 0;
}
