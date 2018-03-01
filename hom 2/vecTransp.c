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
	return pos;
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
		p += (int) leftover / L;
		i = leftover % L;
	} else {
		q = (int) n / (L + 1);
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
  tag = 5;
  int myData[2]; 

  /* Initialize MPI */	
  rc = MPI_Init(&argc, &argv);
  rc = MPI_Comm_size(MPI_COMM_WORLD, &size);
  rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  

  int pos[2];
  getNodePosFromRank(rank,pos);

  int p = pos[0];
  int q = pos[1];

  int sIdx = muInvCol(p,0);
  int eIdx = muInvCol(p+1,0) - 1;

  // Find out how many and which nodes to send data to
  int sIdxLocal[2];
  int eIdxLocal[2];
  muRow(sIdx,sIdxLocal);
  muRow(eIdx,eIdxLocal);

  int firstQIdx = sIdxLocal[0]; // get q position of the node that we are sending the first matrix element to
  int lastQIdx = eIdxLocal[0]; // get q position of the node that we are sending the last matrix element to
  jStartOfFirstQIdx = sIdxLocal[1]; // get local index j in node q that the first matrix element to be sent has
  jEndOfLastQIdx = eIdxLocal[1];

  int numProcessesToSendTo = lastQIdx - firstQIdx + 1;

  int myArrayIdx = 0;
  int sendStartIdx;
  int sendEndIdx;
  // Send loop
  for (int qCur = firstQIdx; qCur <= lastQIdx; ++qCur)
  {
  	int L = (int) M / Q;
  	int numElemsInQ = L + (qCur < M % Q? 1 : 0);

  	if (qCur == firstQIdx) {
  		sendStartIdx = jStartOfFirstQIdx;
  	} else {
  		sendStartIdx = 0;
  	}

  	if (qCur == lastQIdx) {
  		sendEndIdx = jEndOfLastQIdx;
  	} else {
  		sendEndIdx = numElemsInQ;
  	}

  	int sendSize = sendEndIdx - sendStartIdx + 1;
  	myArrayIdx += sendSize;

  	if (P / Q >= 1) {

  	}
  	
  	MPI_Isend(myData[myArrayIdx],sendSize, MPI_DOUBLE,)
  }

  /*
  if (pos[0] != pos[1]) {
  	int sendRecvPos[2] = {pos[1],pos[0]};
  	int sendRecvRank = getRankFromNodePos(sendRecvPos);
  	RC = MPI_SendRecv	

  }
  */

  /*
  int rank1 = 10;
  int rank2 = 7;

  int pos1[2];
  int pos2[2];
  getNodePosFromRank(rank1,pos1);
  getNodePosFromRank(rank2,pos2);

  int finalRank1 = getRankFromNodePos(pos1);
  int finalRank2 = getRankFromNodePos(pos2);

  printf("rank1: %d\n",rank1);
  printf("pos1[0]: %d\n", pos1[0]);
  printf("pos1[1]: %d\n",pos1[1]);
  printf("finalRank1: %d\n",finalRank1);

  printf("rank2: %d\n",rank2);
  printf("pos2[0]: %d\n", pos2[0]);
  printf("pos2[1]: %d\n",pos2[1]);
  printf("finalRank2: %d\n",finalRank2);
  */

  exit(0);
}