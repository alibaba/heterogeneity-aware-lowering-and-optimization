import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read npz file to check the content')
    # parser.add_argument('--tensor-names', metavar='N', type=str, nargs='+',
    #                 help='the tensor names should in the npz file')
    parser.add_argument('--npz-file', metavar='N', type=str,
                    help='the npz file to be read')
    args = parser.parse_args()
    # assert isinstance(args.tensor_names, list), "The tensor_names should be a list"
    npzfile = np.load(args.npz_file)
    for name in npzfile.files: # args.tensor_names:
        print(f'The content of [{name}] is:\n[{npzfile[name]}]\n \
               dtype: {npzfile[name].dtype}, shape: {npzfile[name].shape}')
        print('-'*80)