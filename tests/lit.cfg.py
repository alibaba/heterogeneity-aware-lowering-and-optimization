import os
import lit.formats
import urllib

# Pass environment variables to tests
passthrough_env_vars = ['CUDA_VISIBLE_DEVICES']
for var in passthrough_env_vars:
    if var in os.environ:
        config.environment[var] = os.environ[var]

unittest_build_path = os.path.join(config.halo_build_dir, 'tests/unittests')
data_path = os.path.join(config.halo_build_dir, 'tests/unittests/data')
if not os.path.exists(unittest_build_path):
    os.makedirs(unittest_build_path)
if not os.path.exists(data_path):
    # TODO: read from /unittests directly.
    os.system('cp -r /unittests ' + data_path)

config.name = "Halo Tests"

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Test suffixes.
config.suffixes = ['.cc']

tensorrt_dir = os.path.dirname(config.lib_cudart_path)
dnnl_dir = os.path.dirname(config.lib_dnnl_path)
xnnpack_dir = os.path.join(os.path.dirname(
    config.lib_xnnpack_path), os.path.pardir)

if config.lib_rt.upper() == "ON":
    config.available_features.add('halo_rtlib')

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
config.substitutions.append(('%unittests_path',
                              os.path.join(config.halo_src_dir, 'tests/unittests')))
config.substitutions.append(('%onnx_path',
                              os.path.join(config.halo_build_dir, 'lib/parser/onnx')))
config.substitutions.append(('%data_path', data_path))
config.substitutions.append(('%models_root', '/models'))

path = config.halo_lib_dir
if 'LD_LIBRARY_PATH' in config.environment:
    path = os.path.pathsep.join((config.environment['LD_LIBRARY_PATH'], path))
config.environment['LD_LIBRARY_PATH'] = path

config.environment['PATH'] = os.path.pathsep.join(
    (config.environment['PATH'],
     os.path.sep.join((config.halo_build_dir, 'llvm', 'bin'))))

config.environment['HALO_BASE_PATH'] = config.halo_build_dir
config.test_format = lit.formats.ShTest("0")
