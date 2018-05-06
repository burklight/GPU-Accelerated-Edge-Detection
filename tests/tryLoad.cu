#include <iostream>
#include <string>
#include <fstream>
#include <vector>

int main(){

  std::ifstream file("../data/512/img1.txt");
  std::string value;
  std::vector<unsigned char> ourImage (512*512);
  unsigned int i = 0;
  while(file.good() && i < 512*512){
    std::getline(file, value, ',');
    if(i==512*512) std::cout << "hello" << value << std::endl;
    ourImage[i++] = (unsigned char) std::stoi(value);
  }

  for(unsigned int k = 0; k < 512*512; ++k) std::cout << (int) ourImage[k] << std::endl;

  return 0;
}
