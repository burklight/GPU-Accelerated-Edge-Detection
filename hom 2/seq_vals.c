#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1000 // number of inner grid points
#define M_PI 3.14159265358979323846 // PI
#define H 1.0 / (N + 1) // The first and last points x = 0, 1 must be 0
#define BLACK 0
#define RED 1

double r_func(double x)
{
  return cos(20*x) - 1.0;
}

double u_func(double x)
{
  return sin(5*M_PI*x);
}

double f_func(double x)
{
  return sin(5*M_PI*x)*(cos(20*x) - 1) - 25*M_PI*M_PI*sin(5*M_PI*x);
}

int main(int argc, char* argv[])
{

  /* Create the x values */
  double x[N];
  for (unsigned int i = 0; i < N; i++){
    x[i] = (i + 1)*H;
  }

  /* Compute the real values of u that we are going to use for the error */
  /* Also, compute the values of the functions so we don't need to do calculations
     at each iteration */
  double u_real[N], f[N], r[N];
  for (unsigned int i = 0; i < N; i++){
    u_real[i] = u_func(x[i]);
    f[i] = f_func(x[i]);
    r[i] = r_func(x[i]);
  }

  /* Print the functions */
  FILE *u_fp, *f_fp, *r_fp;
  u_fp = fopen("u_matr.txt","wb");
  f_fp = fopen("f_matr.txt","wb");
  r_fp = fopen("r_matr.txt","wb");

  for (unsigned int i = 0; i < N; i++){
    fprintf(u_fp, "%.3f\n", u_real[i]);
  }
  fclose(u_fp);
  for (unsigned int i = 0; i < N; i++){
    fprintf(f_fp, "%.3f\n", f[i]);
  }
  fclose(f_fp);
  for (unsigned int i = 0; i < N; i++){
    fprintf(r_fp, "%.3f\n", r[i]);
  }
  fclose(r_fp);
}
