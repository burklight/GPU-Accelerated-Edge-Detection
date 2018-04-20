#include <iostream>
#include <cstdlib>

#include "utils.hpp"

void random_ints(int* v, int N){
  for(unsigned int i = 0; i < N; ++i) v[i] = std::rand() % 100;
}

int main()
{
  // Create the input values
  int N;
  std::cin >> N;
  int size = N*sizeof(int);
  int* a = (int *) malloc(size); random_ints(a, N);
  int* b = (int *) malloc(size); random_ints(b, N);
  int* c = (int *) malloc(size);

  addV(a, b, c, N, true);

  std::cout << "a: ";
  for(unsigned int i = 0; i < N; ++i) std::cout << a[i] << "\t";
  std::cout << std::endl;
  std::cout << "b: ";
  for(unsigned int i = 0; i < N; ++i) std::cout << b[i] << "\t";
  std::cout << std::endl;
  std::cout << "c: ";
  for(unsigned int i = 0; i < N; ++i) std::cout << c[i] << "\t";
  std::cout << std::endl;

  // Cleanup
  free(a); free(b); free(c);
}
