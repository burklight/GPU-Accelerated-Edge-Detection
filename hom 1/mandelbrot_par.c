#include <stdio.h>
#define M 128
#define N 128

#include <stdio.h>
#include <string.h>
#include <mpi.h>

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
  int rank, size, tag, rc;
  MPI_Status status;
  unsigned char color[M*N];
  FILE *fp;

  rc = MPI_Init(&argc, &argv);
  rc = MPI_Comm_size(MPI_COMM_WORLD, &size);
  rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  fp = fopen("color.txt","wb");
  double b = 2.0;
  int K = 256;
  int N_rows = M / (size - 1);
  double dx = 2.0 * b / ((double) M);
  double dy = 2.0 * b / ((double) N);
  double dreal, dimag;

  tag = 100;
  if (rank == 0) {
    /* Master code */
    int c[2];
    int row;
    for (int i = 1, row = 0; i < size; i++, row += N_rows){
      MPI_Send(&row, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
    }
    for (int n_pix = 0; n_pix < N * M; n_pix++){
      MPI_Recv(&c, 2, MPI_INT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
      color[c[0]] = c[1];
    }
    for (unsigned int i = 0; i < M; i++) {
      for (unsigned int j = 0; j < N; j++){
        fprintf(fp, "%d,", color[i+j*M]);
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  } else{
    /* Slave code */
    int row;
    int c[2];
    MPI_Request req;
    MPI_Recv(&row, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
    double dreal = 0, dimag = 0;
    for (int x = 0; x < M; x++){
      dreal = ((double) x) * dx - b;
      for (int y = row; y < row + N_rows; y++){
        dimag = ((double) y) * dy - b;
        c[0] = x + y*M;
        c[1] = cal_pixel(dreal, dimag, b, K);
      }
    }
    MPI_Isend(&c, 2, MPI_INT, 0, tag, MPI_COMM_WORLD, &req);
  }

  rc = MPI_Finalize();
}
