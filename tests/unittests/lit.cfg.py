import os
import lit.formats

config.name = "Halo Unit Tests"

# Test suffixes.
config.suffixes = ['.sh']

config.excludes = ['run_single.sh']

# Request to run it early
config.is_early = True

# Setup source root.
config.test_source_root = os.path.join(config.halo_src_dir, 'tests/unittests')

config.environment['HALO_BUILD_DIR'] = config.halo_build_dir
config.environment['HALO_SRC_DIR'] = config.halo_src_dir
config.environment['HALO_UNITTESTS_PATH'] = config.halo_unittests_dir

config.substitutions.append(('%src_dir', config.halo_src_dir))
config.substitutions.append(('%build_dir', config.halo_build_dir))

config.test_format = lit.formats.ShTest("0")
