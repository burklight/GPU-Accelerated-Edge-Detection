#include <iostream>
#include <string>
#include <fstream>
#include <vector>

std::vector<short> loadImage(std::string filename, int dimx, int dimy){
  std::ifstream file(filename);
  std::string value;
  std::vector<short> ourImage (dimx*dimy);
  unsigned int i = 0;
  while(file.good() && i < dimx*dimy){
    std::getline(file, value, ',');
    ourImage[i++] = (short) std::stoi(value);
  }
  return ourImage;
}
