#include <test_util.h>

#include <array>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <unordered_map>

using namespace cv;
using namespace std;
extern std::vector<std::pair<std::string, std::array<float, 5>>>
post_process_nhwc(int orig_img_w, int orig_img_h, float bb13[1 * 13 * 13 * 255],
                  float bb26[1 * 26 * 26 * 255], float bb52[1 * 52 * 52 * 255]);
constexpr int N = 1;
constexpr int H = 416;
constexpr int W = 416;
constexpr int C = 3;

extern "C" {
void yolo_v3_init();
void yolo_v3_fini();
void yolo_v3(const float input[N * H * W * C],
             float out_conv2d_59[1 * 255 * 13 * 13],
             float out_conv2d_67[1 * 255 * 26 * 26],
             float out_conv2d_75[1 * 255 * 52 * 52]);
}

static unordered_map<string, Scalar> ClassColor;

static Mat detect(Mat& src_image, bool print_image_info) {
  float scale = std::min(416.0 / src_image.cols, 416.0 / src_image.rows);
  Size new_size{(int)(src_image.cols * scale), int(src_image.rows * scale)};
  Mat resized;
  resize(src_image, resized, new_size, 0, 0, INTER_CUBIC);
  Mat new_img(Size{416, 416}, CV_8UC3, Scalar{128, 128, 128});
  // std::cout<< src_image.cols << ", " << src_image.rows << "\n";
  // std::cout << resized.cols << ", " << resized.rows;

  Mat roi = new_img(Rect{(new_img.cols - resized.cols) / 2,
                         (new_img.rows - resized.rows) / 2, resized.cols,
                         resized.rows});
  resized.copyTo(roi);

  std::vector<float> input(416 * 416 * 3);
  for (int i = 0, e = input.size(); i < e; ++i) {
    input[i] = new_img.data[i] / 255.0;
  }
  std::vector<float> out_13(N * 255 * 13 * 13);
  std::vector<float> out_26(N * 255 * 26 * 26);
  std::vector<float> out_52(N * 255 * 52 * 52);

#ifdef USE_NCHW
  auto permute_2d = [](std::vector<float>* arg, int src_dim0, int src_dim1) {
    assert(arg->size() == src_dim0 * src_dim1);
    std::vector<float> buf(arg->size());
    for (int i = 0; i < src_dim1; ++i)
      for (int j = 0; j < src_dim0; ++j)
        buf[i * src_dim0 + j] = (*arg)[j * src_dim1 + i];
    arg->swap(buf);
  };

  permute_2d(&input, 416 * 416, 3);
#endif

  auto begin_time = Now();
  yolo_v3(input.data(), out_13.data(), out_26.data(), out_52.data());
  auto end_time = Now();

#ifdef USE_NCHW
  permute_2d(&out_13, 255, 13 * 13);
  permute_2d(&out_26, 255, 26 * 26);
  permute_2d(&out_52, 255, 52 * 52);
#endif

  auto ret = post_process_nhwc(src_image.cols, src_image.rows, out_13.data(),
                               out_26.data(), out_52.data());
  auto t = GetDuration(begin_time, end_time);
  auto rate = 1.0 / t;

  std::string vendor = "Unknown";
#define str(x) #x
#define xstr(x) str(x)
#ifdef VENDOR
  vendor = xstr(VENDOR);
#endif
  putText(src_image, "HALO with ODLA " + vendor, cvPoint(30, 50),
          FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 100, 150), 1, CV_AA);
  char buf[128];
  sprintf(buf, "FPS: %.2f", rate);
  putText(src_image, buf, cvPoint(src_image.cols - 150, 50),
          FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 100, 150), 1, CV_AA);

  for (auto& obj : ret) {
    if (print_image_info)
      std::cout << "[" << obj.first << "], pos:[" << obj.second[0] << ", "
                << obj.second[1] << ", " << obj.second[2] << ", "
                << obj.second[3] << "] score:" << obj.second[4] << std::endl;
    if (ClassColor.find(obj.first) == ClassColor.end()) {
      int n = (ClassColor.size() + 1) % 10;
      ClassColor[obj.first] = Scalar(n * 255 / 10, n * 255 / 10, n * 255 / 10);
    }
    auto color = ClassColor[obj.first];
    putText(src_image, obj.first, cvPoint(obj.second[1], obj.second[0] - 8),
            FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1, CV_AA);

    rectangle(src_image, Point(obj.second[1], obj.second[0]),
              Point(obj.second[3], obj.second[2]), color, 2);
  }

  return src_image;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    cout << " Usage: demo [image file or video file]" << endl;
    return -1;
  }
  const std::string input_file(argv[1]);
  auto ext = input_file.substr(input_file.size() - 4);
  bool is_video = ext == ".mp4" || ext == ".m4v";
  if (is_video) {
    const std::string output_file{"out/new.mp4"};
    std::cout << "Detecting for video, output: " << output_file << "\n";
    VideoCapture vid(input_file);
    if (!vid.isOpened()) {
      cout << "Could not open or find the video" << std::endl;
      return -1;
    }
    auto fourcc = int(vid.get(CAP_PROP_FOURCC));
    auto fps = vid.get(CAP_PROP_FPS);
    auto video_size = Size(int(vid.get(CAP_PROP_FRAME_WIDTH)),
                           int(vid.get(CAP_PROP_FRAME_HEIGHT)));
    int frames = vid.get(CV_CAP_PROP_FRAME_COUNT);
    cout << "fourCC: " << fourcc << std::endl;
    cout << "FPS: " << fps << std::endl;
    cout << "video size: W:" << video_size.width << " H:" << video_size.height
         << std::endl;
    cout << "frames: " << frames << std::endl;
    auto out = VideoWriter(output_file, fourcc, fps, video_size, true);

    {
      vector<float> dummy(416 * 416 * 3);
      vector<float> dummy1(13 * 13 * 256);
      vector<float> dummy2(26 * 26 * 256);
      vector<float> dummy3(52 * 52 * 256);
      yolo_v3(dummy.data(), dummy1.data(), dummy2.data(), dummy3.data());
    }

    cout << "Start inferencing\n";

    while (1) {
      Mat frame;
      vid >> frame;
      if (frame.empty()) break;
      auto dst = detect(frame, false);
      out.write(dst);
    }
    vid.release();
  } else {
    const std::string output_file{"out/result.jpg"};
    std::cout << "Detecting for image, output: " << output_file << std::endl;

    Mat src_image = imread(argv[1], CV_LOAD_IMAGE_COLOR); // Read the file
    if (!src_image.data) {
      cout << "Could not open or find the image" << std::endl;
      return -1;
    }
    auto out_img = detect(src_image, true);
    imwrite(output_file, out_img);
  }
  return 0;
}
