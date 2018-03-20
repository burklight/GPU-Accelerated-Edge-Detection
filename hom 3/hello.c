/* sample hello world program  *
 *  C Michael Hanke 2006-10-19 */

#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <stdlib.h>


void merge(double* arr1, int len1, double* arr2, int len2, double* merged) {
    int idx1 = 0;
    int idx2 = 0;
    for (int i = 0; i < len1 + len2; ++i) {
        merged[i] = (arr1[idx1] < arr2[idx2] ? arr1[idx1++] : arr2[idx2++]);
    }
}

int getLocalArrayLen(int rank, int N, int P) {
    return ((int) N/P) + (rank < N % P ? 1 : 0);
}


int double_cmp(const void *a, const void *b)
{
  const double *da = (const double *) a;
  const double *db = (const double *) b;

  return (*da > *db) - (*da < *db);
}

int main(int argc, char **argv)
{
    int rank, size, tag, rc, i;
    MPI_Status status;
    char message[20];

    if (argc < 2){
      printf("The length of the list is required!\n");
      exit(0);
    }
    int N = atoi(argv[1]);

    rc = MPI_Init(&argc, &argv);
    rc = MPI_Comm_size(MPI_COMM_WORLD, &size);
    rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    tag = 100;
    srandom(rank+1);

    int evenprocess = (rank % 2 == 0);
    int evenphase = 1;
    int myArrayLen = getLocalArrayLen(rank, N, size);
    double A[myArrayLen];
    for (unsigned int i = 0; i < myArrayLen; ++i) {
        A[i] = (double)rand() / (double)RAND_MAX;
    }
    qsort(A, myArrayLen,sizeof(double),double_cmp);

    for (unsigned int i = 0; i < N; ++i) {
        if ((evenphase && evenprocess) || (!evenphase && !evenprocess)) {
            if (rank < size-1) {
                int recvLen = getLocalArrayLen(rank+1, N, size);
                double X[recvLen];
                rc = MPI_Send(&A, myArrayLen, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD);
                rc = MPI_Recv(&X, recvLen, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD, &status);

                double merged[recvLen + myArrayLen];
                merge(A, myArrayLen, X, recvLen, merged);
                for (unsigned int j = 0; j < myArrayLen; j++){
                  A[j] = merged[j];
                }
            }
        } else {
            if (rank > 0) {
                int recvLen = getLocalArrayLen(rank-1, N, size);
                double X[recvLen];
                rc = MPI_Recv(&X, recvLen, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD, &status);
                rc = MPI_Send(&A, myArrayLen, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD);

                double merged[recvLen + myArrayLen];
                merge(A, myArrayLen, X, recvLen, merged);
                for (unsigned int j = 0; j < myArrayLen; j++){
                  A[j] = merged[j+recvLen];
                }
            }
        }
        evenphase = !evenphase;
    }

    // Each process prints its chunk of ordered array
    FILE *fp;
    char pId[5];
    char nId[5];
    char nameFile[50] = "Files/ord_";
    sprintf(pId, "%d_", rank);
    strcat(nameFile,pId);
    sprintf(nId, "%d", N);
    strcat(nameFile,nId);
    strcat(nameFile, ".txt");
    fp = fopen(nameFile,"wb");
    for (unsigned int i = 0; i < myArrayLen-1; i++){
      fprintf(fp, "%.5f,", A[i]);
    }
    fprintf(fp, "%.5f\n", A[myArrayLen-1]);
    fclose(fp);

    rc = MPI_Finalize();
}
