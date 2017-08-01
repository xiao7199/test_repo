#include "input.h"
#include <vector>
#include <fstream>
#include <cstdlib>

using namespace std;
using namespace cv;


void readMat(const string filename, Mat& dst) {

  ifstream file;
  string line;
  file.open(filename, ios::in);

  getline(file, line, ',');
  int nr = stoi(line);
  getline(file, line);
  int nc = stoi(line);  
  dst.create(nr, nc, CV_8UC1);

  for (int r = 0; r < nr; r++) {
    for (int c = 0; c < nc; c++) {
      if (c < nc-1)
        getline(file, line, ',');
      else
        getline(file, line);        
      dst.at<uint8_t>(r, c) = stoi(line);
    }
  }
}


void readFloatMat(const string filename, Mat& dst) {

  ifstream file;
  string line;
  file.open(filename, ios::in);

  getline(file, line, ',');
  int nr = stoi(line);
  getline(file, line);
  int nc = stoi(line);  
  dst.create(nr, nc, CV_32FC1);

  for (int r = 0; r < nr; r++) {
    for (int c = 0; c < nc; c++) {
      if (c < nc-1)
        getline(file, line, ',');
      else
        getline(file, line);
      float f = stof(line);
      dst.at<float>(r, c) = f;
    }
  }
}

void readMask(const string filename, Mat& dst) {
  dst = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
}