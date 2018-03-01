#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define P 4
#define Q 4

#define N 10
#define M 10


// Get node position in P x Q cluster from process rank
void getNodePosFromRank(int rank,int*pos) {
	pos[0] = (int) rank/Q;
	pos[1] = rank % Q;
}

// get process rank from P x Q cluster position
int getRankFromNodePos(int* pos) {
  int rank = pos[0]*Q;
  rank += pos[1];
  return rank;
}


//Returns the global column index of an element with local col index i on col process p
int muInvCol(int p, int i) {
	int n = (int) N / P;
	n *= p;
	// The leftover nbr of elements are distributed among the N % P first processes
	// so if p is less than N % P we add one extra element for every process with rank < p
	// Else if p >= N % P all leftover nbr of elements have been distributed.
	n += (p < N % P? p : N % P);
	n += i;
	return n;
}

//Returns the global row index of an element with local row index j on row process q
int muInvRow(int q, int j) {
	int m = (int) M / Q;
	m *= q;
	m += (q < M % Q? q : M % Q);
	m += j;
	return m;
}

void muCol(int n, int* localIndex) {
	int L = (int) N / P;
	int p, i;
	// True if p >= N % P for n
	if (n >= N % P * (L + 1)) {
		p = N % P;
		int leftover = n - N % P * (L + 1);
		p += (int) leftover / L;
		i = leftover % L;

	} else {
		p = (int) n / (L + 1);
		i = n % (L + 1);
	}
	localIndex[0] = p;
	localIndex[1] = i;
}

// get process q position and local index from global index m
void muRow(int m, int* localIndex) {
	int L = (int) M / Q;
	int q, j;
	// True if q >= M % Q for m
	if (m >= M % Q * (L + 1)) {
		q = M % Q;
		int leftover = m - M % Q * (L + 1);
		q += (int) leftover / L;
		j = leftover % L;
	} else {
		q = (int) m / (L + 1);
		j = m % (L + 1);
	}
	localIndex[0] = q;
	localIndex[1] = j;
}

int main(int argc, char* argv[])
{
  /* Local variables */
  int size, rank, rc, tag;
  MPI_Status status;
  MPI_Request request;
  tag = 5;


  /* Initialize MPI */	
  rc = MPI_Init(&argc, &argv);
  rc = MPI_Comm_size(MPI_COMM_WORLD, &size);
  rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int pos[2];
  getNodePosFromRank(rank,pos);

  int p = pos[0];
  int q = pos[1];

  int L = (int) N / P;
  int myDataSize = (p < N % P? L + 1 : L);
  double myData[myDataSize]; 

  // The value at each index of the y array is the same as its index
  // This for loop makes sure each process gets its portion of the y array
  int n = muInvCol(p,0);
  for (int i = 0; i < myDataSize; ++i) {
    myData[i] = n + i;
  }

  int sendPos[2] = {pos[1],pos[0]};
  int sendProcessRank = getRankFromNodePos(sendPos);

  if (sendProcessRank != rank) {
    int L = (int) M / Q;
    int recvDataSize = (q < M % Q? L + 1 : L);
    double receivedData[recvDataSize];
	
    rc = MPI_Isend(myData,myDataSize, MPI_DOUBLE,sendProcessRank,tag,MPI_COMM_WORLD, &request);
    rc = MPI_Recv(receivedData, recvDataSize, MPI_DOUBLE, sendProcessRank,tag, MPI_COMM_WORLD, &status);

    printf("Received data at process %d ", rank);
    printf("with cluster coordinates (%d,",p);
    printf("%d): ",q);
    for (int i = 0; i < recvDataSize; ++i)
    {
      printf("%d,",(int) receivedData[i]);
    }
    printf("\n");
  } else {
    printf("Data at process %d ", rank);
    printf("with cluster coordinates (%d,",p);
    printf("%d): ",q);
    for (int i = 0; i < myDataSize; ++i)
    {
      printf("%d,",(int) myData[i]);
    }
    printf("\n");
  }

  /* Finish the process */
  MPI_Finalize();
  exit(0);
}