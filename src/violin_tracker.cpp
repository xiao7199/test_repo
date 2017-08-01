#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>

#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <string>
#include <array>
#include <chrono>
#include <sys/stat.h>

#include "types.h"
#include "models.h"
#include "channels.h"
#include "pretranslationpyr_tracker_pyramid.h"
#include "pretranslation_tracker_pyramid.h"
#include "simple_tracker_pyramid.h"
#include "tracker.h"
#include "input.h"
#include "utils.h"

using namespace cv;
using namespace std;
namespace po = boost::program_options;

void maskColorImage(const Mat I, const Mat M, Mat &dst)
{
  dst.create(I.size(), I.type());
  for (int r = 0; r < I.rows; r++)
  {
    for (int c = 0; c < I.cols; c++)
    {
      dst.at<Vec3b>(r, c) = M.at<uint8_t>(r, c) * I.at<Vec3b>(r, c);
    }
  }
}
int numDigits(int n)
{
  if (n == 0)
    return 1;
  int c = 0;
  while (n > 0)
  {
    n /= 10;
    c++;
  }
  return c;
}

Mat reset_tracker(po::variables_map vm, Mat I1_full_orig, float scale,
                  float sigma, PointVec *pts, int *x_min, int *y_min, int *x_max, int *y_max, int index)
{
  // imwrite("template.png", I1_full_orig);
  Mat I1_full, I1_small, I1, I1_int, I2_full, I2_gray, I2, I2_int, I2_small,
      M_full, M, wI, masked_wI, output_frame;
  string mask_file = "mask_xin_" + to_string(index) +".jpg";
  // if (index)
  // {
    // string mask_file = vm["mask-file"].as<string>();
    // end_frame = vm["end-frame"].as<int>(&end_frame);
    // end_frame = 10;
    // cout << "xxx" << end_frame << endl;
    if (mask_file.substr(mask_file.find_last_of('.') + 1) == "csv")
    {
      readMat(mask_file, M_full);
    }
    else
    {
      readMask(mask_file, M_full);
    }
  // }
  // else
  // {
  // }
  resize(I1_full_orig, I1_small, Size(), scale, scale, INTER_AREA);
  cvtColor(I1_full_orig, I1_full, CV_RGB2GRAY);
  resize(I1_full, I1_int, Size(), scale, scale, INTER_AREA);
  I1_int.convertTo(I1, CV_64FC3);
  GaussianBlur(I1, I1, Size(11, 11), sigma, sigma);
  // imshow("temp",I1);
  // waitKey();
  resize(M_full, M, Size(), scale, scale, INTER_AREA);

  cout << "Loaded mask and video file" << endl;

  Mat overlay;
  Mat overlay_solid;

  PointVec overlap_pts;
  //Find all the points inside the ROI
  int min_x = I1.cols, max_x = 0, min_y = I1.rows, max_y = 0;
  // PointVec pts;
  for (int r = 0; r < M.rows; r++)
  {
    for (int c = 0; c < M.cols; c++)
    {
      if (M.at<uint8_t>(r, c) >= 1)
      {
        pts->push_back(Vector2f(c, r));
        min_x = min(c, min_x);
        min_y = min(r, min_y);
        max_x = max(c, max_x);
        max_y = max(r, max_y);
      }
    }
  }
  *x_min = min_x;
  *y_min = min_y;
  *x_max = max_x;
  *y_max = max_y;
  return I1;
}

int main(int argc, char **argv)
{
  int start_frame, end_frame, test, levels;
  float scale, min_displacement;
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("video-file", po::value<string>(), "filename of the video to track")
    ("mask-file", po::value<string>(), "filename of the mask specifying the tracking region")
    ("overlay-file", po::value<string>(), "filename of an overlay to warp and apply to each image")
    ("start-frame", po::value<int>(&start_frame)->default_value(2),
     "The frame to start trackign at, defaults to 2 since the "
     "first frame is used as the template")
    ("end-frame", po::value<int>(&end_frame)->default_value(-1),
     "The frame to end trackign at, a value of -1 indicates that the tracking should"
     "continue for the entire video")
    ("caching",
     "Turn caching on. If turned on violin_tracker will output pngs of each frame "
     "read from the input video to make a cache.")
    ("read-cache",
     "Instead of reading from the video file read frames from the cache generated with the "
     "cache option. Note that the template is still read from the video file so it still "
     "needs to be provided")
    ("scale", po::value<float>(&scale)->default_value(1.0),
      "Scaling factor to apply to the video")
    ("levels", po::value<int>(&levels)->default_value(3),
      "Number of levels in the tracking pyramid")
    ("min_disp", po::value<float>(&min_displacement)->default_value(1.0),
     "The minimum displacement of the control points that indicates convergance")
    ("video-out", "Do not create an image for each frame, but write to video file");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    cout << desc << endl;
    return 1;
  }
  bool caching = vm.count("caching") > 0;
  bool read_cache = vm.count("read-cache") > 0;
  bool video_out = vm.count("video-out") > 0;
  VideoWriter out_vid;
  bool debug = true;
  string file_dir = "imgs", debug_dir = "debug", cache_dir = "cache";
  char file_name_buf[50];

  int frame_num = 1;
  int cnt = 1;
  int cur_debug_idx = 0;
  Mat I1_full, I2_full, I2_gray, I2, I2_int, I2_small,
      M_full, M, wI, masked_wI, output_frame, I1_full_orig;
  vector<int> debug_frames = {};
  float sigma = 1.5;

  VideoCapture cap;
  int success = cap.open(vm["video-file"].as<string>());
  cout << __LINE__ << " " << success << endl;

  //Get the first frame and save it as a template

  // imshow("temp", I1_full_orig);
  // waitKey();
  // imwrite("template.png", I1_full_orig);
  // video_out = 0;

  // string mask_file = vm["mask-file"].as<string>();
  // // end_frame = vm["end-frame"].as<int>(&end_frame);
  // // end_frame = 10;
  // cout << "xxx" << end_frame << endl;
  // if (mask_file.substr(mask_file.find_last_of('.') + 1) == "csv") {
  //   readMat(mask_file, M_full);
  // } else {
  //   readMask(mask_file, M_full);
  // }

  // resize(I1_full_orig, I1_small, Size(), scale, scale, INTER_AREA);
  // cvtColor(I1_full_orig, I1_full, CV_RGB2GRAY);
  // resize(I1_full, I1_int, Size(), scale, scale, INTER_AREA);
  // I1_int.convertTo(I1, CV_64FC3);
  // GaussianBlur(I1, I1, Size(11, 11), sigma, sigma);
  // // imshow("temp",I1);
  // // waitKey();
  // resize(M_full, M, Size(), scale, scale, INTER_AREA);

  // cout << "Loaded mask and video file" << endl;

  //Find all the points inside the ROI
  // int min_x = I1.cols, max_x = 0, min_y = I1.rows, max_y = 0;
  // PointVec pts;
  // for (int r = 0; r < M.rows; r++) {
  //   for (int c = 0; c < M.cols; c++) {
  //     if (M.at<uint8_t>(r, c) >= 1) {
  //       pts.push_back(Vector2f(c, r));
  //       min_x = min(c, min_x);
  //       min_y = min(r, min_y);
  //       max_x = max(c, max_x);
  //       max_y = max(r, max_y);
  //     }
  //   }
  // }
  // cout << min_x << " " << min_y << " " << max_x << " " << max_y << " " << endl;

  //Set up the inital parameters, possibly reading them from a file
  // Homography::ControlPoints cpts;
  // cpts <<
  //   min_x, min_y,
  //   min_x, max_y,
  //   max_x, min_y,
  //   max_x, max_y;
  // Homography::ParameterStruct cur_params(cpts);
  // if (start_frame > 2) {
  //   sprintf(file_name_buf, "%s/init-params-%06d.csv", file_dir.c_str(), start_frame-2);
  //   if (read_params<Homography>(file_name_buf, cur_params)) {
  //     cnt = start_frame-1;
  //     frame_num = start_frame;
  //   } else {
  //     return -1;
  //   }
  //   //Read through video until the start_frame is reached
  //   if (!read_cache) {
  //     cout << "Read frame ";
  //     int digits = 0;
  //     for (int i = 2; i < start_frame; i++) {
  //       Mat tmp;
  //       cout << string(digits, '\b');
  //       cap >> tmp;
  //       cout << i << std::flush;
  //       digits = numDigits(i);
  //     }
  //     cout << endl;
  //   }
  // }
  //Set up the tracker
  // cout << "Setting up tracker" << endl;
  // AlgorithmParameters ap;
  // ap.levels = levels;
  // ap.iterations = 100;
  // ap.min_disp = min_displacement;
  // ap.debug = &debug;
  // PretranslationTrackerPyramid<Homography, TwoBit> tr(ap);
  // Homography::ParameterStruct eye(cpts);
  // tr.set_template(I1, pts, eye);
  // vector<double> times;

  // while (cur_debug_idx < debug_frames.size() &&
  //        debug_frames[cur_debug_idx] <= start_frame) {
  //   cur_debug_idx++;
  // }
  // if (cur_debug_idx > 0 &&
  //     debug_frames[cur_debug_idx - 1] == -1 &&
  //     debug_frames[cur_debug_idx] > start_frame) cur_debug_idx--;

  // Mat masked_I1;
  // maskColorImage(I1_full_orig, M_full, masked_I1);
  // output_frame.create(I1_full_orig.rows, I1_full_orig.cols + masked_I1.cols, I1_full_orig.type());
  // Mat left(output_frame, Rect(0, 0, I1_full_orig.cols, I1_full_orig.rows));
  // Mat right(output_frame, Rect(I1_full_orig.cols, 0, masked_I1.cols, masked_I1.rows));
  // I1_full_orig.copyTo(left);
  // masked_I1.copyTo(right);
  // sprintf(file_name_buf, "%s/frame-%06d.png", file_dir.c_str(), 0);
  // imwrite(string(file_name_buf), output_frame);
  vector<double> times;
Homography::ControlPoints cpts;
      AlgorithmParameters ap;
        ap.levels = levels;
        ap.iterations = 100;
        ap.min_disp = min_displacement;
        ap.debug = &debug;
        PretranslationTrackerPyramid<Homography, TwoBit> tr(ap);
         Homography::ParameterStruct cur_params;
  while (end_frame == -1 || frame_num <= end_frame)
  {
    
    cout << "frame_num" << frame_num << endl;
    cap >> I2_full;
    if (I2_full.empty())
      break;  
    // if (frame_num < 1001){
    //   frame_num++;
    //   continue;
    // }  
    if ((frame_num - 1) % 100 == 0)
    { 
      int x_min;
      int y_min;
      int x_max;
      int y_max;
      // Homography::ControlPoints cpts;

      cout << "frame_num" << frame_num;
      PointVec pts;
    
      
    Mat I1 = reset_tracker(vm, I2_full, scale,
                    sigma, &pts, &x_min, &y_min, &x_max, &y_max, (frame_num - 1));
    
    Homography::ControlPoints tmp;
          tmp << x_min, y_min,
            x_min, y_max,
            x_max, y_min,
            x_max, y_max;
            cpts = tmp;
    
    imwrite("template.png", I2_full);
    
    cur_params.vec << 0, 0, 0, 0, 0, 0, 0, 0;
    cur_params.cpts= cpts;
    
       
        cout << "Setting up tracker" << endl;
        
        Homography::ParameterStruct eye(cpts);
        tr.set_template(I1, pts, eye);
            
      }

           
      //Write initial params to a file
      // sprintf(file_name_buf, "%s/init-params-%06d.csv", file_dir.c_str(), cnt);
      // write_params<Homography>(cur_params, file_name_buf);

      // if (caching)
      // {
      //   sprintf(file_name_buf, "%s/img-%06d.png", cache_dir.c_str(), frame_num);
      //   imwrite(string(file_name_buf), I2_full);
      // }
      resize(I2_full, I2_small, Size(), scale, scale, INTER_AREA);
      cvtColor(I2_full, I2_gray, CV_RGB2GRAY);
      resize(I2_gray, I2_int, Size(), scale, scale, INTER_AREA);
      I2_int.convertTo(I2, CV_64FC1);
      GaussianBlur(I2, I2, Size(11, 11), sigma, sigma);

      auto start = chrono::high_resolution_clock::now();
          
      Result<Homography> res = tr.step(I2, Mat::ones(I2.rows, I2.cols, CV_8UC1), cur_params);
          
      cur_params = res.params;
      // cout << cur_params.vec << endl;
      auto end = chrono::high_resolution_clock::now();

      chrono::duration<double> diff = end - start;
      times.push_back(diff.count());
      cout << "Processing frame " << frame_num++ << " ";
      cout << "iters: " << res.iterations << " err: " << res.final_err
           << " time: " << diff.count() << endl;
    

      Homography::ParameterStruct scaled_params = Homography::scale_params(cur_params, 1 / scale);
      warp_image<Homography>(I2_full, scaled_params, wI);
      // imshow("xxx",I2_full);
      
      // maskColorImage(wI, M_full, masked_wI);
      output_frame.create(wI.rows, I2_full.cols + wI.cols, wI.type());
      Mat left(output_frame, Rect(0, 0, I2_full.cols, I2_full.rows));
      Mat right(output_frame, Rect(I2_full.cols, 0, wI.cols, wI.rows));
      I2_full.copyTo(left);
      wI.copyTo(right);
    

      sprintf(file_name_buf, "%s/%06d.jpg", file_dir.c_str(), cnt);
      imwrite(string(file_name_buf), wI);
      cout << "img output" << endl;
      cnt++;
    }
    double tot = 0;
    for (int i = 0; i < times.size(); i++)
    {
      tot += times[i];
    }
    cout << "avg-time: " << tot / times.size() << " avg-fps: " << times.size() / tot << endl;
    return 0;
  }
