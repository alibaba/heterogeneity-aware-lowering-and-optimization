[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)




Build Unitest
===============

Use commond ODLA_BUILD_POPART_UNITEST to build unit test modelus and use ENABLE_COVERAGE  to generate code coverage report.

```bash
cmake -G Ninja .. -DODLA_BUILD_POPART_UNITEST=ON	\
                  -DENABLE_COVERAGE=ON 						\
                  -DCMAKE_BUILD_TYPE=Debug && ninja odla_popart
```



After building unit test modelus. runfiles are located on /build/bin. Run these files and generate code coverage report.

```bash
cd build
./bin/unitest_odla_compute
./bin/unitest_odla_config
./bin/unitest_odla_ops_math
./bin/unitest_odla_ops_nn
./bin/unitest_odla_ops
./bin/unitest_odla_ops_process
cd ..

mkdir tmp
find ./ -type f -name "*.gcno" | xargs cp -t ./tmp
find ./ -type f -name "*.gcda" | xargs cp -t ./tmp

cd tmp && lcov -d . -t 'odla' -o 'odla.info' -b . -c
genhtml -o result odla.info
tar -cvf result.tar result
```

