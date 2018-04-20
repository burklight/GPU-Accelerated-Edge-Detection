#ifndef UTILS_HPP
#define UTILS_HPP

/* Cuda version for vector addition */
__global__ void addCuV(int *d_a, int* d_b, int* d_c){
  d_c[blockIdx.x] = d_a[blockIdx.x] + d_b[blockIdx.x];
}

/* Cuda version for vector dot product */
__global__ void dotCuV(int *d_a, int* d_b, int* d_c){
  d_c[blockIdx.x] = d_a[blockIdx.x] * d_b[blockIdx.x];
}

/* Vector addition */
void addV(int *a, int* b, int* c, int N, bool gpu = false){
  if (gpu){
    // Create the device copies for the GPU
    int size = N * sizeof(int);
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU with N blocks
    addCuV<<<N,1>>>(d_a, d_b, d_c);
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  } else{
    for (unsigned int i = 0; i < N; ++i)c[i] = a[i] + b[i];
  }
}

/* Vector multiplication */
void dotV(int *a, int* b, int* c, int N, bool gpu = false){
  if (gpu){
    // Create the device copies for the GPU
    int size = N * sizeof(int);
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU with N blocks
    dotCuV<<<N,1>>>(d_a, d_b, d_c);
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  } else{
    for (unsigned int i = 0; i < N; ++i)c[i] = a[i] * b[i];
  }
}

 #endif
