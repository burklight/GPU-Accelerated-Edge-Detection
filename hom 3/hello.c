#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <stdlib.h>

/* Merges two sorted arrays into a sorted array O(len1 + len2)*/
void merge(double* arr1, int len1, double* arr2, int len2, double* merged) {
    int idx1 = 0;
    int idx2 = 0;
    for (int i = 0; i < len1 + len2; i++) {
      if (arr1[idx1] < arr2[idx2] && idx1 < len1){
        merged[i] = arr1[idx1];
        idx1++;
      } else if (idx2 < len2){
        merged[i] = arr2[idx2];
        idx2++;
      } else if (idx1 == len1){
        merged[i] = arr2[idx2];
        idx2++;
      } else {
        merged[i] = arr1[idx1];
        idx1++;
      }
    }
}

/* Gets the local array length depending on the process rank */
int getLocalArrayLen(int rank, int N, int P) {
    return ((int) N/P) + (rank < N % P ? 1 : 0);
}

/* Comparison function of the quicksort algorithm */
int double_cmp(const void *a, const void *b)
{
  const double *da = (const double *) a;
  const double *db = (const double *) b;

  return (*da > *db) - (*da < *db);
}


int main(int argc, char **argv)
{
    /* Definition of variables */
    int rank, size, tag, rc, i;
    MPI_Status status;
    tag = 100;

    /* If there length of the array is not given, exit! */
    if (argc < 2){
      printf("The length of the list is required!\n");
      exit(0);
    }
    int N = atoi(argv[1]);

    /* MPI Initialization */
    rc = MPI_Init(&argc, &argv);
    rc = MPI_Comm_size(MPI_COMM_WORLD, &size);
    rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Generation of the process chunk of random data */
    int myArrayLen = getLocalArrayLen(rank, N, size);
    double A[myArrayLen];
    srandom(rank+1);
    for (unsigned int i = 0; i < myArrayLen; ++i) {
        A[i] = (double)rand() / (double)RAND_MAX;
    }

    /* Each process prints its chunk of un-ordered array */
    FILE *fp;
    char pId[5];
    char nId[5];
    sprintf(pId, "%d_", rank);
    sprintf(nId, "%d", N);
    char nameFileU[50] = "Files/unord_";
    strcat(nameFileU,pId);
    strcat(nameFileU,nId);
    strcat(nameFileU, ".txt");
    fp = fopen(nameFileU,"wb");
    for (unsigned int i = 0; i < myArrayLen-1; i++){
      fprintf(fp, "%.5f,", A[i]);
    }
    fprintf(fp, "%.5f\n", A[myArrayLen-1]);
    fclose(fp);

    /* Initial sorting of the individual chunks of data */
    qsort(A, myArrayLen,sizeof(double),double_cmp);

    /* Initialization of odd-even sorting variables */
    int evenprocess = (rank % 2 == 0);
    int evenphase = 1;

    /* Odd-even sorting algorithm */
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

    /* Each process prints its chunk of ordered array */
    char nameFileO[50] = "Files/ord_";
    strcat(nameFileO,pId);
    strcat(nameFileO,nId);
    strcat(nameFileO, ".txt");
    fp = fopen(nameFileO,"wb");
    for (unsigned int i = 0; i < myArrayLen-1; i++){
      fprintf(fp, "%.5f,", A[i]);
    }
    fprintf(fp, "%.5f\n", A[myArrayLen-1]);
    fclose(fp);

    rc = MPI_Finalize();
}
