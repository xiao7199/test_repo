#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <string>
#include <array>

#include "types.h"
#include "models.h"
#include "channels.h"
#include "pretranslation_tracker_pyramid.h"
#include "input.h"
#include "utils.h"

using namespace cv;
using namespace std;



int main(int argc, char** argv)
{
  Mat I1, I2, M;
  readMat("M.csv", M);
  I1 = imread("I1.png", CV_LOAD_IMAGE_GRAYSCALE);
  I2 = imread("I2affine.png", CV_LOAD_IMAGE_GRAYSCALE);
  GaussianBlur(I1, I1, Size(5, 5), 1, 2);
  GaussianBlur(I2, I2, Size(5, 5), 1, 2);  

  PointVec pts;
  for (int r = 0; r < M.rows; r++) {
    for (int c = 0; c < M.cols; c++) {
      if (M.at<uint8_t>(r, c) == 1) {
        pts.push_back(Vector2f(c, r));
      }
    }
  }
  PretranslationTrackerPyramid<Homography, TwoBit> tr(2, 100, 0.5);
  Homography::ControlPoints cpts;
  cpts <<
    0, 0,
    0, 110,
    110, 0,
    110, 110;
  Homography::ParameterStruct eye(cpts);
  tr.set_template(I1, pts, eye);
  Homography::ParameterStruct init_params(cpts);
  Result<Homography> res = tr.step(I2, M, init_params);
  cout << res.iterations << " params: [ " << res.params.vec << "]" << endl;
  namedWindow("Original", CV_WINDOW_AUTOSIZE);
  imshow("Original", I2);
  namedWindow("Warped", CV_WINDOW_AUTOSIZE);
  Mat warped;
  warp_image<Homography>(I2, res.params, warped);
  imshow("Warped", warped);
  namedWindow("Template", CV_WINDOW_AUTOSIZE);
  imshow("Template", I1);
  waitKey(0);
  return 0;
}




