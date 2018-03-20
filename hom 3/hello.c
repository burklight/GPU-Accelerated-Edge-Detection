/* sample hello world program  *
 *  C Michael Hanke 2006-10-19 */

#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <stdlib.h>


void merge(int* arr1, int len1, int* arr2, int len2, int* merged) {
    int idx1 = 0;
    int idx2 = 0;
    for (int i = 0; i < len1 + len2; ++i) {
        merged[i] = (arr1[idx1] < arr2[idx2] ? arr1[idx1++] : arr2[idx2++]);
    }
}
int getLocalArrayLen(int rank, int N, int P) {
    return ((int) N/P) + (rank < N % P ? 1 : 0);
}


int int_cmp(const void *a, const void *b) 
{ 
    const int *ia = (const int *)a; 
    const int *ib = (const int *)b;

    return *ia  - *ib; 
}

int main(int argc, char **argv)
{
    int rank, size, tag, rc, i;
    MPI_Status status;
    char message[20];

    rc = MPI_Init(&argc, &argv);
    rc = MPI_Comm_size(MPI_COMM_WORLD, &size);
    rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = atoi(argv[1]);
    
    tag = 100;
    srand(rank+1);

    /*
    if (rank == 0) {
	    strcpy(message, "Hello, world");
	    for (i = 1; i < size; i++)
	      rc = MPI_Send(message, 13, MPI_CHAR, i, tag, MPI_COMM_WORLD);
    }
    else
	    rc = MPI_Recv(message, 50, MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);

    FILE *fp_1, *fp_2;
    char pId[5];
    char nId[5];
    char nameFile1[50] = "Files/hello_", nameFile2[50] = "Files/goodbye_";
    sprintf(pId, "%d_", rank);
    strcat(nameFile1,pId);
    strcat(nameFile2,pId);
    sprintf(nId, "%d", N);
    strcat(nameFile1,nId);
    strcat(nameFile2,nId);
    strcat(nameFile1, ".txt");
    strcat(nameFile2, ".txt");
    fp_1 = fopen(nameFile1,"wb");
    fprintf(fp_1, "Hello world!");
    fclose(fp_1);
    printf("node %d Has printed Hello world!\n", rank);
    fp_2 = fopen(nameFile2,"wb");
    fprintf(fp_2, "Good bye world!");
    fclose(fp_2);
    printf("node %d Has printed Good bye world!\n", rank);
    */
    int evenprocess = rank % 2 == 0;
    int evenphase = 1;
    int myArrayLen = getLocalArrayLen(rank, N, size);
    int *A;
    for (int i = 0; i < myArrayLen; ++i) {
        A[i] = rand(); 
    }
    qsort(A, myArrayLen, sizeof(int),int_cmp);

    for (int i = 0; i < N; ++i) {
        if ((evenphase && evenprocess) || (!evenphase && !evenprocess)) {
            if (rank < N-1) {
                int recvLen = getLocalArrayLen(rank+1, N, size);
                int *X;
                rc = MPI_Send(&A, myArrayLen, MPI_INTEGER, rank+1, tag, MPI_COMM_WORLD);
                rc = MPI_Recv(&X, recvLen, MPI_INTEGER, rank+1, tag, MPI_COMM_WORLD, &status);

                int *merged;
                merge(A, myArrayLen, X, recvLen, merged);
                A = merged;
            }
        } else {
            if (rank > 0) {
                int recvLen = getLocalArrayLen(rank-1, N, size);
                int *X;
                rc = MPI_Recv(&X, recvLen, MPI_INTEGER, rank-1, tag, MPI_COMM_WORLD, &status); 
                rc = MPI_Send(&A, myArrayLen, MPI_INTEGER, rank-1, tag, MPI_COMM_WORLD);
                
                int *merged;
                merge(A, myArrayLen, X, recvLen, merged);
                A = &merged[recvLen];
            }
        }
        evenphase = !evenphase;
    }

    for (int i = 0; i < myArrayLen; ++i)
    {
        printf("%d,",A[i]);
    }
    rc = MPI_Finalize();
}


