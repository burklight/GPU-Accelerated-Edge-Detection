#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

static vector<unsigned char> ReadAllBytes(char const* filename)
{
  ifstream ifs(filename, ios::binary|ios::ate);
  ifstream::pos_type pos = ifs.tellg();

  vector<char>  tmp(pos);

  ifs.seekg(0, ios::beg);
  ifs.read(&tmp[0], pos);

  // Very important since all Le Cun data is unsigned bytes
  vector<unsigned char> result(tmp.begin(), tmp.end());

  return result;
}

static vector<vector<double> > ReadImages(char const* filename, bool asBinary = true){

  vector<unsigned char> result = ReadAllBytes(filename);
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
  vector<vector<double> > images(N, vector<double>(h*w, 0.0)); // Creating a N x h x w matrix full of zeros
  // Fill the matrix with the images
  for (unsigned int x = 0; x < N; x++){
    for (unsigned int y = 0; y < h; y++){
      for (unsigned int z = 0; z < w; z++){
        if (asBinary) images[x][w*y+z] = (result[curByte++] == 0) ? 0 : 1; // Make sure it is 0 or 1
        else images[x][w*y+z] = ((double) result[curByte++]) / 255.0;
      }
    }
  }

  return images;
}

static vector<int> ReadLabels(char const* filename, bool asBinary = true){

  vector<unsigned char> result = ReadAllBytes(filename);
  int curByte = 0; // This will account for the current bytes
  curByte += 2; // The first two bytes means nothing in Le Cun's dataset
  curByte++; // Third byte is the type but it is always unsigned bytes
  int dim = (int) result[curByte++]; // Forth byte is the number of dimensions (it will always be 1 in labels)
  unsigned int N = (result[curByte++] << 24) | (result[curByte++] << 16) |
    (result[curByte++] << 8) | result[curByte++]; // Number of labels
  vector <int> labels(N); // Creating a vector of N zeros
  // Fill the vector with the labels
  for (unsigned int x = 0; x < N; x++){
    labels[x] = (int) result[curByte++];
  }

  return labels;
}

static vector<vector<double> > ReadTestImages(bool asBinary = true){
  return ReadImages("../mnist/t10k-images-idx3-ubyte", asBinary);
}

static vector<vector<double> > ReadTrainImages(bool asBinary = true){
  return ReadImages("../mnist/train-images-idx3-ubyte", asBinary);
}

static vector<int> readTestLabels(){
  return ReadLabels("../mnist/t10k-labels-idx1-ubyte");
}

static vector<int> readTrainLabels(){
  return ReadLabels("../mnist/train-labels-idx1-ubyte");
}
