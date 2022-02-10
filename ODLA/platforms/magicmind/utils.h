#ifndef MAGICMIND_UTILS_H_
#define MAGICMIND_UTILS_H_

class EnvTime {
 public:
  static constexpr uint64_t kSecondsToNanos = 1000ULL * 1000ULL * 1000ULL;

  EnvTime() = default;
  virtual ~EnvTime() = default;

  static uint64_t NowNanos() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (static_cast<uint64_t>(ts.tv_sec) * kSecondsToNanos +
            static_cast<uint64_t>(ts.tv_nsec));
  }
};

#endif
