import os
import lit.formats
import urllib

# Download models if not exist
model_path = os.path.join(config.halo_build_dir, 'models')
urls = {'https://github.com/onnx/models/raw/master/vision/classification/'
        'resnet/model/resnet50-v2-7.onnx': 'resnet50_v2.onnx'}
for url, filename in urls.items():
    filename = os.path.join(model_path, filename)
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.isfile(filename):
        urllib.urlretrieve(url, filename)

# Download onnx op data
unittest_build_path = os.path.join(config.halo_build_dir, 'tests/unittests')
if not os.path.exists(unittest_build_path):
    os.makedirs(unittest_build_path)
data_path = os.path.join(config.halo_build_dir, 'tests/unittests/data')
if not os.path.exists(data_path):
    os.system('git clone -b rel-1.8.0 https://github.com/onnx/onnx.git')
    os.system('cp -r onnx/onnx/backend/test/data/node ' + data_path)
    os.system('rm -rf onnx')

config.name = "Halo Tests"

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Test suffixes.
config.suffixes = ['.cc']

tensorrt_dir = os.path.dirname(config.lib_cudart_path)
dnnl_dir = os.path.dirname(config.lib_dnnl_path)
xnnpack_dir = os.path.join(os.path.dirname(
    config.lib_xnnpack_path), os.path.pardir)

if len(tensorrt_dir):
    config.available_features.add('odla_tensorrt')
    tensorrt_dir = os.path.join(tensorrt_dir, os.path.pardir)

print(tensorrt_dir)

if len(dnnl_dir):
    config.available_features.add('odla_dnnl')
    dnnl_dir = os.path.join(dnnl_dir, os.path.pardir)

print(dnnl_dir)

test_include = os.path.sep.join((config.test_source_root, 'include'))

include_path = '-I %s -I %s -I %s -I %s -I %s' % (config.halo_header_src_dir,
                                                  config.halo_header_build_dir, test_include, config.llvm_src_dir, config.llvm_build_dir)

odla_path = os.path.sep.join((config.halo_src_dir, 'ODLA'))

link_path = '-Wl,-rpath=%s -L %s ' % (
    config.halo_lib_dir, config.halo_lib_dir)

flags = '-std=c++17'
if (config.build_type.lower() == "debug"):
    flags = flags + " -O0 -g"

config.substitutions.append(('%include', include_path))
config.substitutions.append(('%link', link_path + '-lhalo '))
config.substitutions.append(('%cxx', config.cxx))
config.substitutions.append(('%cc', config.cc))
config.substitutions.append(('%flags', flags))
config.substitutions.append(('%src_dir', config.halo_src_dir))
config.substitutions.append(('%build_dir', config.halo_build_dir))
config.substitutions.append(('%halo_compiler',
                             os.path.sep.join((config.halo_build_dir, 'bin', 'halo'))))
config.substitutions.append(('%halo_analyzer',
                             os.path.sep.join((config.halo_build_dir, 'bin', 'analyzer'))))
config.substitutions.append(('%halo_diagnostic',
                             os.path.sep.join((config.halo_build_dir, 'bin', 'diagnostic'))))
config.substitutions.append(('%odla_path', odla_path))
config.substitutions.append(('%tensorrt_path', tensorrt_dir))
config.substitutions.append(('%dnnl_path', dnnl_dir))
config.substitutions.append(('%xnnpack_path', xnnpack_dir))
config.substitutions.append(('%odla_link', link_path))
config.substitutions.append(('%models_dir', model_path))
config.substitutions.append(('%unittests_path',
                              os.path.join(config.halo_src_dir, 'tests/unittests')))
config.substitutions.append(('%onnx_path',
                              os.path.join(config.halo_build_dir, 'lib/parser/onnx')))
config.substitutions.append(('%data_path', data_path))

path = config.halo_lib_dir
if 'LD_LIBRARY_PATH' in config.environment:
    path = os.path.pathsep.join((config.environment['LD_LIBRARY_PATH'], path))
config.environment['LD_LIBRARY_PATH'] = path

config.environment['PATH'] = os.path.pathsep.join(
    (config.environment['PATH'],
     os.path.sep.join((config.halo_build_dir, 'llvm', 'bin'))))

config.environment['HALO_BASE_PATH'] = config.halo_build_dir
config.test_format = lit.formats.ShTest("0")
