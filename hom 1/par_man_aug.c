#include <stdio.h>
#define M 2400 // Number of rows
#define N 2400 // Number of columns

#include <stdio.h>
#include <string.h>
#include <mpi.h>

int cal_pixel(double dreal, double dimag, int b, int K)
/* Function to calculate a specific pixel value */
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
  /* Definition of variables */
  unsigned char color[M][N];
  FILE *fp;
  fp = fopen("color.txt","wb");
  double b = 2.0;
  int K = 256;
  int zoom_x = 4*3*16;
  int zoom_y = 4*3*16;
  double prev_dx = 2.0 * b / ((double) M);
  double prev_dy = 2.0 * b / ((double) N);
  double dx = prev_dx / zoom_x;
  double dy = prev_dy / zoom_y;
  double dreal = 0, dimag = 0;

  /* Definition of special variables for parallelization */
  int rank, size, rc;
  int tag = 100;
  MPI_Status status;
  rc = MPI_Init(&argc, &argv);
  rc = MPI_Comm_size(MPI_COMM_WORLD, &size);
  rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int N_cols = N / (size - 1); // we assume it is divisible

  if (rank == 0){
    /* Master code */
    int c[M*N_cols+1];
    int col;
    /* First send which is the starting row for the slave processes */
    for (unsigned int p = 1, col = 0; p < size; p++, col += N_cols){
      MPI_Send(&col, 1, MPI_INT, p, tag, MPI_COMM_WORLD);
    }
    /* Then receive the other rows from the slaves and save them*/
    for (unsigned int p = 1; p < size; p++){
      MPI_Recv(&c, M*N_cols+1, MPI_INT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
      col = c[M*N_cols];
      for (unsigned int i = 0; i < M; i++){
        for (unsigned int j = col; j < col + N_cols; j++){
          color[i][j] = c[i*N_cols + j-col];
        }
      }
    }
    /* Write the obtained picture */
    for (unsigned int i = 0; i < M; i++){
      for (unsigned int j = 0; j < N; j++){
        fprintf(fp, "%d,",color[i][j]);
      }
      fprintf(fp,"\n");
    }
    fclose(fp);

  } else {
    /* Slave code */
    int c[M*N_cols+1];
    int col;
    /* First receive which is the starting row for each process */
    MPI_Recv(&col, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
    c[M*N_cols] = col;
    /* Then compute the pixels for the assigned N_rows rows */
    double offset_x = prev_dx * (3 * M / 8 + M / 12 + 7.5 *  M / 192) ;
    double offset_y = prev_dy * (1 * N / 8 + 0.5 * M / 192);
    for (unsigned int x = 0; x < M; x++){
      dimag = ((double) x) * dx - b + offset_x;
      for (unsigned int y = col; y < col + N_cols; y++){
        dreal = ((double) y) * dy - b + offset_y;
        c[x*N_cols+y-col] = cal_pixel(dreal, dimag, b, K);
      }
    }
    /* Then send back the obtained values to the master */
    MPI_Send(&c, M*N_cols+1, MPI_INT, 0, tag, MPI_COMM_WORLD);
  }

  rc = MPI_Finalize();
}
