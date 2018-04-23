#ifndef UTILS_HPP
#define UTILS_HPP

#define KERNEL_RADIUS 1
#define WIDTH 28

/* Cuda version for vector addition */
__global__ void addCuV(double *d_a, double* d_b, double* d_c){
  d_c[blockIdx.x] = d_a[blockIdx.x] + d_b[blockIdx.x];
}

/* Cuda version for vector dot product */
__global__ void dotCuV(double *d_a, double* d_b, double* d_c){
  d_c[blockIdx.x] = d_a[blockIdx.x] * d_b[blockIdx.x];
}

/* Cuda version for matrix convolution */
__global__ void convCu(double *img, double *filt, double *res, int N, int M){
  int idx = threadIdx.x + blockDim.x*threadIdx.y;
  int idxImg, idxFilt;
  res[idx] = 0;
  for (unsigned int y = 0; y < M; ++y){
    for (unsigned int x = 0; x < M; ++x){
      idxImg = (threadIdx.x+x) + N*(threadIdx.y+y);
      idxFilt = x + M*y;
      res[idx] += img[idxImg] * filt[idxFilt];
    }
  }

  // Data in the cache
  /*__shared__ double data[WIDTH + KERNEL_RADIUS*2][WIDTH + KERNEL_RADIUS*2];

  // Global memory adress of the thread
  const int idxThr = threadIdx.x + blockIdx.x * blockDim.x + threadIdx.x + WIDTH
    + blockIdx.y * blockDim.y * WIDTH;

  int x, y;
  const int x0 = threadIdx.x + blockIdx.x*blockDim.x;
  const int y0 = threadIdx.x + blockIdx.y*blockDim.y;

  // Top left
  x = x0 - KERNEL_RADIUS;
  y = y0 - KERNEL_RADIUS;
  if (x < 0 || y < 0) data[threadIdx.x][threadIdx.y] = 0;
  else  data[threadIdx.x][threadIdx.y] = img[idxThr - KERNEL_RADIUS
    - KERNEL_RADIUS * WIDTH];

  // Top right
  x = x0 + KERNEL_RADIUS;
  y = y0 - KERNEL_RADIUS;
  if (x > WIDTH-1 || y < 0) data[threadIdx.x+blockDim.x][threadIdx.y] = 0;
  else  data[threadIdx.x+blockDim.x][threadIdx.y] = img[idxThr + KERNEL_RADIUS
      - KERNEL_RADIUS * WIDTH];

  // Bottom left
  x = x0 - KERNEL_RADIUS;
  y = y0 + KERNEL_RADIUS;
  if (x < 0 || y > WIDTH-1) data[threadIdx.x][threadIdx.y + blockDim.y] = 0;
  else  data[threadIdx.x][threadIdx.y + blockDim.y] = img[idxThr - KERNEL_RADIUS
    + KERNEL_RADIUS * WIDTH];

  // Bottom right
  x = x0 + KERNEL_RADIUS;
  y = y0 + KERNEL_RADIUS;
  if (x > WIDTH-1 || y > WIDTH-1) data[threadIdx.x+blockDim.x][threadIdx.y + blockDim.y] = 0;
  else  data[threadIdx.x + blockDim.x][threadIdx.y + blockDim.y] = img[idxThr + KERNEL_RADIUS
    + KERNEL_RADIUS * WIDTH];

  __syncthreads();

  // convolution
  double sum = 0;
  x = KERNEL_RADIUS + threadIdx.x;
  y = KERNEL_RADIUS + threadIdx.y;
  for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; ++i){
    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; ++j){
      sum += data[x+i][y+j] * filt[(KERNEL_RADIUS+j)+KERNEL_RADIUS*(KERNEL_RADIUS+i)];
    }
  }

  res[idxThr] = sum;*/
}


/* Vector addition */
void addV(double *a, double* b, double* c, int N, bool gpu = false){
  if (gpu){
    // Create the device copies for the GPU
    int size = N * sizeof(double);
    double *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch addV() kernel on GPU with N blocks
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
void dotV(double *a, double* b, double* c, int N, bool gpu = false){
  if (gpu){
    // Create the device copies for the GPU
    int size = N * sizeof(double);
    double *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch dotV() kernel on GPU with N blocks
    dotCuV<<<N,1>>>(d_a, d_b, d_c);
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  } else{
    for (unsigned int i = 0; i < N; ++i)c[i] = a[i] * b[i];
  }
}

/* Matrix convolution */
void conv(vector<double> img, vector<double> filt, vector<double> &res,
  int N, int M, bool gpu = false){
  if (gpu){
    // Create the device copies for the GPU
    int sizeImg = N*N * sizeof(double);
    int sizeFilt = M*M * sizeof(double);
    int sizeRes = (N-M+1)*(N-M+1) * sizeof(double);

    double *d_img, *d_filt, *d_res;
    cudaMalloc((void **)&d_img, sizeImg);
    cudaMalloc((void **)&d_filt, sizeFilt);
    cudaMalloc((void **)&d_res, sizeRes);

    // Copy inputs to device
    cudaMemcpy(d_img, &img[0], sizeImg, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filt, &filt[0], sizeFilt, cudaMemcpyHostToDevice);

    // Launch conv() kernel on GPU with N blocks
    //dim3 blocks(WIDTH, WIDTH);
    dim3 threads(N-M+1,N-M+1);
    convCu<<<1,threads>>>(d_img, d_filt, d_res, N, M);

    // Copy result back to host
    cudaMemcpy(&res[0], d_res, sizeRes, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_img); cudaFree(d_filt); cudaFree(d_res);

  } else {
    /* The resulting image will have (N-M+1)x(N-M+1) size */
    for (unsigned int resY = 0; resY < (N-M+1); ++resY){
      for (unsigned int resX = 0; resX < (N-M+1); ++resX){
        res[resX+(N-M+1)*resY] = 0;
        for (unsigned int filtY = 0; filtY < M; ++filtY){
          for (unsigned int filtX = 0; filtX < M; ++filtX){
            res[resX+(N-M+1)*resY] += img[(resX+filtX)+N*(resY+filtY)]
              * filt[filtX+M*filtY];
          }
        }
      }
    }
  }
}

 #endif
