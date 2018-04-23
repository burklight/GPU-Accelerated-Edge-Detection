#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

static unsigned char * ReadAllBytes(char const* filename)
{
  ifstream ifs(filename, ios::binary|ios::ate);
  ifstream::pos_type pos = ifs.tellg();

  unsigned int size = pos * sizeof(char);
  char* tmp = (char *) malloc(size);
  unsigned char* result = (unsigned char *) malloc(size);

  ifs.seekg(0, ios::beg);
  ifs.read(tmp, pos);

  // Very important since all Le Cun data is unsigned bytes
  for (unsigned int i = 0; i < pos; ++i) result[i] = (unsigned char) tmp[i];

  return result;
}

static double* ReadImages(char const* filename, bool asBinary = true){

  unsigned char* result = ReadAllBytes(filename);
  int curByte = 0; // This will account for the current bytes
  curByte += 2; // The first two bytes means nothing in Le Cun's dataset
  curByte++; // Third byte is the type but it is always unsigned bytes
  int dim = (int) result[curByte++]; // Forth byte is the number of dimensions (it will always be 3 in images)
  unsigned int N = (result[curByte++] << 24) | (result[curByte++] << 16) |
    (result[curByte++] << 8) | result[curByte++]; // Number of images
  int h = (result[curByte++] << 24) | (result[curByte++] << 16) |
    (result[curByte++] << 8) | result[curByte++]; // Height of the images
  int w = (result[curByte++] << 24) | (result[curByte++] << 16) |
    (result[curByte++] << 8) | result[curByte++]; // Width of the images
  double* images = (double *) malloc(N*h*w*sizeof(double)); // Images in a 1D array
  // Fill the matrix with the images
  for (unsigned int x = 0; x < N; x++){
    for (unsigned int y = 0; y < h; y++){
      for (unsigned int z = 0; z < w; z++){
        if (asBinary) images[z+y*w+x*w*h] = (result[curByte++] == 0) ? 0 : 1; // Make sure it is 0 or 1
        else images[z+y*w+x*w*h] = ((double) result[curByte++]) / 255.0;
      }
    }
  }

  return images;
}

static int* ReadLabels(char const* filename, bool asBinary = true){

  unsigned char* result = ReadAllBytes(filename);
  int curByte = 0; // This will account for the current bytes
  curByte += 2; // The first two bytes means nothing in Le Cun's dataset
  curByte++; // Third byte is the type but it is always unsigned bytes
  int dim = (int) result[curByte++]; // Forth byte is the number of dimensions (it will always be 1 in labels)
  unsigned int N = (result[curByte++] << 24) | (result[curByte++] << 16) |
    (result[curByte++] << 8) | result[curByte++]; // Number of labels
  int* labels = (int *) malloc(N*sizeof(int)); // Creating a vector of N zeros
  // Fill the vector with the labels
  for (unsigned int x = 0; x < N; x++){
    labels[x] = (int) result[curByte++];
  }

  return labels;
}

static double* ReadTestImages(bool asBinary = true){
  double* images = ReadImages("../mnist/t10k-images-idx3-ubyte", asBinary);
  cout << images[0] << endl;
  return images;
}

static double* ReadTrainImages(bool asBinary = true){
  return ReadImages("../mnist/train-images-idx3-ubyte", asBinary);
}

static int* readTestLabels(){
  return ReadLabels("../mnist/t10k-labels-idx1-ubyte");
}

static int* readTrainLabels(){
  return ReadLabels("../mnist/train-labels-idx1-ubyte");
}

int* tryd(){
  int a[3];
  a[0] = 8;
  return a;
}
