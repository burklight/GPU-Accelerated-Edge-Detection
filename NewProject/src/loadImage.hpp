#include <iostream>
#include <string>
#include <fstream>
#include <vector>

std::vector<short> loadImage(std::string filename, int dim){
  std::ifstream file(filename);
  std::string value;
  std::vector<short> ourImage (dim*dim);
  unsigned int i = 0;
  while(file.good() && i < dim*dim){
    std::getline(file, value, ',');
    ourImage[i++] = (short) std::stoi(value);
  }
  return ourImage;
}
