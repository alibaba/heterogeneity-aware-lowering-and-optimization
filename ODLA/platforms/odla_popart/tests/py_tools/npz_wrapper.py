import numpy as np
import argparse

np_dtypes = {'UINT32':np.uint32, 'INT32':np.int32, 
             'FP16':np.half, 'FP32':np.float32}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save files to one npz file')
    parser.add_argument('--file-names', metavar='N', type=str, nargs='+', 
                required=True, help='the files to be saved into a npz file')
    parser.add_argument('--dtypes', metavar='N', type=str, nargs='+',
                required=True, help='the dtypes for each file contains')
    parser.add_argument('--out-file', metavar='N', type=str,
                required=True, help='the files to be saved into a npz file')
    parser.add_argument('--batch-size', metavar='N', type=int, default=1,
                required=True, help='the batch size will used')
    parser.add_argument('--batches-per-step', metavar='N', type=int, default=1,
                required=True, help='the batches per step')                
    args = parser.parse_args()
    assert isinstance(args.file_names, list), "The file_names should be a list"
    assert len(args.file_names) == len(args.dtypes), \
           "count of files must match the count of dtypes"
    contents = dict()
    real_size = args.batch_size * args.batches_per_step
    real_content = []
    for file_name, dtype in zip(args.file_names, args.dtypes):
        real_content = []
        with open(file_name) as f:
            content = f.read()
            content = content.replace('\n', '').split(',')
            if 'INT' in dtype.upper():
                content = [int(i) for i in content]
            elif 'FP' in dtype.upper():
                content = [float(i) for i in content]
            else:
                raise ValueError(f'The dtype: {dtype} was unknown')
            for i in range(real_size):
                real_content.extend(content)
            # convert it to the int or float representation
            content = np.array(real_content, dtype=np_dtypes[dtype.upper()])
            contents[file_name] = content
    np.savez(args.out_file, **contents)
    print(f'Saved the files {args.file_names} to {args.out_file}')
