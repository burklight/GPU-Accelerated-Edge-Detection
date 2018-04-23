#include <iostream>
#include <vector>
#include "../mnist/readMnist.hpp"
using namespace std;

int main()
{
    vector<vector<vector<double> > > testImages = ReadTestImages(false);//  ReadImages("./mnist/t10k-images-idx3-ubyte");
    vector<vector<vector<double> > > trainImages = ReadTrainImages(false); //ReadImages("./mnist/train-images-idx3-ubyte");
    /* If you want the images not being in two values 0, 1:
    vector<vector<vector<int> > > testImages = ReadTestImages(false);//  ReadImages("./mnist/t10k-images-idx3-ubyte");
    vector<vector<vector<int> > > trainImages = ReadTrainImages(false); //ReadImages("./mnist/train-images-idx3-ubyte");
    */
    vector<int> testLabels = readTestLabels(); //ReadLabels("./mnist/t10k-labels-idx1-ubyte");
    vector<int> trainLabels = readTrainLabels(); //ReadLabels("./mnist/train-labels-idx1-ubyte");

    cout << "*******************************************************" << endl;
    cout << "Training example of number: " << trainLabels[0] << endl;
    cout << "*******************************************************" << endl;
    for (unsigned int y = 0; y < 28; y++){
      for (unsigned int z = 0; z < 28; z++){
        (trainImages[0][y][z] > 0.0) ? cout << " " : cout << "*";
        cout << " ";
      }
      cout << endl;
    }
    cout << "*******************************************************" << endl;
    cout << "Test example of number: " << testLabels[0] << endl;
    cout << "*******************************************************" << endl;
    for (unsigned int y = 0; y < 28; y++){
      for (unsigned int z = 0; z < 28; z++){
        (testImages[0][y][z] > 0.0) ? cout << " " : cout << "*";
        cout << " ";
      }
      cout << endl;
    }

  return 0;
}
