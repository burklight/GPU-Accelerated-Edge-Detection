/* sample hello world program  *
 *  C Michael Hanke 2006-10-19 */

#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank, size, tag, rc, i;
    MPI_Status status;
    char message[20];

    rc = MPI_Init(&argc, &argv);
    rc = MPI_Comm_size(MPI_COMM_WORLD, &size);
    rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    tag = 100;
    if (rank == 0) {
      for(i = 1; i < size; i++){
        int rnk;
        rc = MPI_Recv(&rnk, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
        printf("node's rank : %d\n", rnk);
      }
    }
    else{
      /* With non-blocking send */
      MPI_Request req1;
      rc = MPI_Isend(&rank, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &req1);
      /* With blocking send */
      // rc = MPI_Send(&rank, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
    }

    rc = MPI_Finalize();
}
