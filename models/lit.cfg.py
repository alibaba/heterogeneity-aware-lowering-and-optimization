import os
import lit.formats

# Pass environment variables to tests
passthrough_env_vars = ['CUDA_VISIBLE_DEVICES']
for var in passthrough_env_vars:
    if var in os.environ:
        config.environment[var] = os.environ[var]

config.name = "Halo Model Tests"

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Test suffixes.
config.suffixes = ['.sh']

config.excludes = ['run.all.sh', 'build_image.sh']
config.environment['MODELS_ROOT'] = '/models'
config.environment['SRC_DIR'] = config.halo_src_dir
config.environment['BUILD_DIR'] = config.halo_build_dir
config.environment['ODLA_INC'] = os.path.sep.join(
    (config.halo_src_dir, 'ODLA', 'include'))
config.environment['ODLA_LIB'] = os.path.sep.join(
    (config.halo_build_dir, 'lib'))
config.environment['TEST_TEMP_DIR'] = '/tmp'
if not os.path.isfile(config.lib_cudart_path):
    config.environment['TEST_WITH_GPU'] = '0'
else:
    config.environment['TEST_WITH_GPU'] = '1'
config.environment['HALO_BIN'] = os.path.sep.join(
    (config.halo_build_dir, 'bin', 'halo'))
config.environment['PATH'] = os.path.pathsep.join(
    (config.environment['PATH'],
     os.path.sep.join((config.halo_build_dir, 'llvm', 'bin'))))
config.substitutions.append(('%models_root', config.environment['MODELS_ROOT']))
config.substitutions.append(('%test_temp_dir', config.environment['TEST_TEMP_DIR']))

lit_config.parallelism_groups['modeltest'] = 8
config.parallelism_group = 'modeltest'

config.test_format = lit.formats.ShTest("0")
