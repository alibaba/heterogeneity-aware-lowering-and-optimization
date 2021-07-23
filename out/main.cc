#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>

#include "model.h"

#define THREAD_COUNT 10

void inference(std::vector<std::array<char, 28 * 28>>* imgs, std::vector<char>* labels, int start_index, int count, std::mutex* correct_mutext, int* correct)
{
  int local_correct = 0;
  for(int i=start_index; i<start_index + count; i++){
    std::array<float, 28 * 28> input;
    std::array<float, 10> output;
    std::transform((*imgs)[i].begin(), (*imgs)[i].end(), input.begin(),
                  [](char x) { return ((unsigned char)x) / 255.0F; });
    //std::cout << "---------------------------------->" << std::endl;
    //for(int j=0; j<input.size(); j++)
    //  std::cout << j << ":" << input[j] << std::endl;
    //std::cout << "--------------------------------------------------weishabuxing" << std::endl;
    mnist_simple(input.data(), output.data());
    int pred = std::max_element(output.begin(), output.end()) - output.begin();
    local_correct += (pred == (*labels)[i]);
  }
  std::lock_guard<std::mutex> guard(*correct_mutext);
  *correct += local_correct;  
}

int main(int argc, char** argv) {
  struct ImageFileHead {
    int32_t magic;
    uint32_t count;
    uint32_t rows;
    uint32_t cols;
  };
  struct LabelFileHead {
    uint32_t magic;
    uint32_t count;
  };

  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " [MNIST image-idx3 file] [MNIST label-idx3 file]\n";
    return 1;
  }

  std::ifstream fs(argv[1], std::ios::binary);
  struct ImageFileHead fh {
    0, 0, 0, 0
  };
  struct LabelFileHead lh {
    0, 0
  };

  fs.read(reinterpret_cast<char*>(&fh), sizeof(fh));
  auto to_le = [](uint32_t& x) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    x = __builtin_bswap32(x);
#endif
  };
  to_le(fh.count);
  to_le(fh.rows);
  to_le(fh.cols);

  if (fh.rows != 28 || fh.cols != 28) {
    std::cerr << "Invalid data file (rows:" << fh.rows << " cols: " << fh.cols
              << ")\n";
    return 1;
  }

  std::vector<std::array<char, 28 * 28>> imgs(fh.count);
  for (unsigned i = 0; i < fh.count; ++i) {
    fs.read(imgs[i].data(), imgs[i].size());
    if (fs.fail()) {
      std::cerr << "Failed to read image file\n";
    }
  }

  std::ifstream f(argv[2], std::ios::binary);
  f.read(reinterpret_cast<char*>(&lh), sizeof(lh));

  to_le(lh.count);

  if (lh.count != fh.count) {
    std::cerr << "data mismatch\n";
    return 1;
  }
  std::vector<char> labels(lh.count);
  f.read(labels.data(), labels.size());

  std::mutex correct_mutex;
  int correct = 0;
  int nr_tests = lh.count;

  mnist_simple_init();

  // for (int i = 0; i < nr_tests; ++i) {
  //   std::array<float, 28 * 28> input;
  //   std::array<float, 10> output;
  //   std::transform(imgs[i].begin(), imgs[i].end(), input.begin(),
  //                  [](char x) { return ((unsigned char)x) / 255.0F; });
  //   mnist_simple(input.data(), output.data());
  //   int pred = std::max_element(output.begin(), output.end()) - output.begin();
  //   correct += (pred == labels[i]);
  // }

  //std::vector<std::thread> threads(THREAD_COUNT);
  std::thread threads[THREAD_COUNT];
  int start_index = 0;
  for(int i=0; i < THREAD_COUNT; i++){
    //threads.push_back(std::thread(inference, &imgs, &labels, start_index, 10000/THREAD_COUNT, &correct_mutex, &correct));
    threads[i] = std::thread(inference, &imgs, &labels, start_index, 10000/THREAD_COUNT, &correct_mutex, &correct);
    start_index += 10000/THREAD_COUNT;
  }
  std::cout << "main thread problem?" << std::endl;
  std::cout << "Join problem?" << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(10));
  for(int i=0; i < THREAD_COUNT; i++){
    threads[i].join();
  }
  // for(auto& thread : threads){
  //   thread.join();
  //   std::cout << "No, quite normal" << std::endl;
  // }
  mnist_simple_fini();
  std::cout << "Accuracy " << correct << "/" << nr_tests << " ("
            << correct * 100.0 / nr_tests << "%) \n";
}
