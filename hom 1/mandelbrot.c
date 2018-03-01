#include <stdio.h>
#define M 2048
#define N 2048

#include <stdio.h>
#include <string.h>
//#include <mpi.h>

int cal_pixel(double dreal, double dimag, int b, int K)
{
  double tmp;
  int count = 0;
  double zreal = 0, zimag = 0;
  int b2 = b * b;
  while ((zreal * zreal + zimag * zimag <= b2) && (count < K)) {
    tmp = zreal * zreal - zimag * zimag + dreal;
    zimag = 2 * zreal * zimag + dimag;
    zreal = tmp;
    count++;
  }
  return count;
}

int main(int argc, char **argv)
{
  unsigned char color[M*N];
  FILE *fp;
  fp = fopen("color.txt","wb");
  /******** Mandelbrot ********/
  double b = 2.0;
  int K = 256;
  double dx = 2.0 * b / ((double) M);
  double dy = 2.0 * b / ((double) N);
  double dreal = 0, dimag = 0;
  for (unsigned int x = 0; x < M; x++){
    dreal = ((double) x) * dx - b;
    for (unsigned int y = 0; y < N; y++){
      dimag = ((double) y) * dy - b;
      color[x+y*M] = cal_pixel(dreal, dimag, b, K);
    }
  }
  /****************************/
  for (unsigned int i = 0; i < M; i++) {
    for (unsigned int j = 0; j < N; j++)
      fprintf(fp, "%d,", color[i+j*M]);
    fprintf(fp, "\n");
    }
  fclose(fp);
}
