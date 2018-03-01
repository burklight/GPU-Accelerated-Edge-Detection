#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N 1000 // number of inner grid points
#define SMX 100//0000 // number of iterations
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
  /* Local variables */
  int size, rank, rc, tag;
  MPI_Status status;
  tag = 5;

  /* Initialize MPI */
  rc = MPI_Init(&argc, &argv);
  rc = MPI_Comm_size(MPI_COMM_WORLD, &size);
  rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (N < size){
    fprintf(stderr, "%s\n", "Too few discretization points...");
    exit(1);
  }

  /* Linear data distribution */
  int Ip, inVal, M_tild, L, R, a;
  a = 2;
  M_tild = N + (size - 1)*a;
  L = M_tild / size;
  R = M_tild % size;
  Ip = L + (rank < R ? 1: 0);
  inVal = (L-a)*rank + (rank < R ? rank : R);

  /* Create the data */
  double u_real[Ip], u[Ip], u_new[Ip];
  double f[Ip], r[Ip];
  double mse;
  double total_mse[SMX];

  /* Create the x values */
  double x[Ip];
  for (unsigned int i = 0; i < Ip; i++){
    x[i] = (inVal + i + 1)*H;
  }

  /* Compute the real values of u that we are going to use for the error */
  /* Also, compute the values of the functions so we don't need to do calculations
     at each iteration */
  for (unsigned int i = 0; i < Ip; i++){
    u_real[i] = u_func(x[i]);
    f[i] = f_func(x[i]);
    r[i] = r_func(x[i]);
  }

  /* Actual computation */
  for (unsigned int step = 0; step < SMX; step++){
    // Communication
    if (rank % 2 == BLACK){
      if (rank < size - 1){ // Avoid sending to P process
        rc = MPI_Send(&u[Ip-2], 1, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD);
        rc = MPI_Recv(&u[Ip-1], 1, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD, &status);
      }
      if (rank > 0){ // Avoid sending to -1 process
        rc = MPI_Send(&u[1], 1, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD);
        rc = MPI_Recv(&u[0], 1, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD, &status);
      }
    } else{ // rank % 2 == RED
      if (rank > 0){ // Avoid sending to -1 process
        rc = MPI_Recv(&u[0], 1, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD, &status);
        rc = MPI_Send(&u[1], 1, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD);
      }
      if (rank < size - 1){ // Avoid sending to P process
        rc = MPI_Recv(&u[Ip-1], 1, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD, &status);
        rc = MPI_Send(&u[Ip-2], 1, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD);
      }
    }
    // Computation of the values
    for (unsigned int i = 1; i < Ip-1; i++){
      u_new[i] = (u[i-1] + u[i+1] - H*H*f[i]) / (2.0 - H*H*r[i]);
    }
    // We update our values and compute the error
    mse = 0;
    for (unsigned int i = 0; i < Ip-1; i++){
      if (i == 0 && rank != 0) continue; // Only the first process uses the first element
      else if(i == Ip-1 && rank != size-1) continue; // Only the last element computes the last element
      u[i] = u_new[i];
      mse += (u[i] - u_real[i])*(u[i] - u_real[i]);
    }
    if (rank == 0 || rank == size - 1) mse /= Ip - 1; // MSE with the number of elements
    else mse /= Ip - 2; // MSE with the number of elements
    // TO DO MPI_Allreduce()...
  }

  /* Finish the process */
  MPI_Finalize();
  exit(0);
}
