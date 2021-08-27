#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <vector>

#include "mnist_simple.h"

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

  int correct = 0;
  int nr_tests = lh.count;
  mnist_simple_init();
  for (int i = 0; i < nr_tests; ++i) {
    std::array<float, 28 * 28> input;
    std::array<float, 10> output;
    std::transform(imgs[i].begin(), imgs[i].end(), input.begin(),
                   [](char x) { return ((unsigned char)x) / 255.0F; });
    mnist_simple(input.data(), output.data());
    int pred = std::max_element(output.begin(), output.end()) - output.begin();
    correct += (pred == labels[i]);
  }
  mnist_simple_fini();
  std::cout << "Accuracy " << correct << "/" << nr_tests << " ("
            << correct * 100.0 / nr_tests << "%) \n";
}
