import argparse
import os
import numpy as np
import faiss
import time
import ctypes


parser = argparse.ArgumentParser()
parser.add_argument('--dstore_keys', type=str, default='', help='memmap where keys and vals are stored')
parser.add_argument('--dstore_vals', type=str, default='', help='memmap where keys and vals are stored')
parser.add_argument('--dstore_dir', type=str, default='', help='used to infer key/val names when they are None')
parser.add_argument('--infer_prefix', type=str, default='dstore', help='used to infer key/val names when they are None')
parser.add_argument('--dstore_size', type=int, default=0, help='number of items saved in the datastore memmap')
parser.add_argument('--actual_dstore_size', type=int, default=None,
    help='only the first actual_dstore_size in dstore is used for the index')
parser.add_argument('--dimension', type=int, default=1024, help='Size of each key')
parser.add_argument('--dstore_fp16', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=1, help='random seed for sampling the subset of vectors to train the cache')
parser.add_argument('--ncentroids', type=int, default=4096, help='number of centroids faiss should learn')
parser.add_argument('--code_size', type=int, default=64, help='size of quantized vectors')
parser.add_argument('--probe', type=int, default=8, help='number of clusters to query')
parser.add_argument('--faiss_index', type=str, help='file to write the faiss index')
parser.add_argument('--train_index', type=str, default='', help='the trained index')
parser.add_argument('--num_keys_to_add_at_a_time', default=1000000, type=int,
                    help='can only load a certain amount of data to memory at a time.')
parser.add_argument('--starting_point', type=int, help='index to start adding keys at')
parser.add_argument('--pca', type=int, default=0, help='apply pca transformation if this value is larger than 0')

args = parser.parse_args()


if args.train_index == '':
    args.train_index = args.faiss_index + ".trained"

if args.dstore_keys == '' or args.dstore_keys == 'none':
    for fname in os.listdir(args.dstore_dir):
        if fname.startswith(args.infer_prefix) and fname.endswith('keys.npy'):
            args.dstore_keys = os.path.join(args.dstore_dir, fname)

        if fname.startswith(args.infer_prefix) and fname.endswith('vals.npy'):
            args.dstore_vals = os.path.join(args.dstore_dir, fname)

    print(f'inferred file names from dir:\nkeys: {args.dstore_keys}\nvals: {args.dstore_vals}')

if args.dstore_keys == '' or args.dstore_vals == '':
    raise ValueError(f'keys and vals files not found')


if args.dstore_size == 0:
    # parse dstore size automatically
    fname = args.dstore_keys.split('/')[-1]
    for x in fname.split('_'):
        if x.startswith('size'):
            args.dstore_size = int(x.split('size')[-1])
            break

    print(f'inferred size: {args.dstore_size}')

if args.dstore_size == 0:
    raise ValueError(f'fail to parse size from {args.dstore_keys}')

if args.actual_dstore_size is None or args.actual_dstore_size == 0:
    args.actual_dstore_size = args.dstore_size

print(args)

if args.dstore_fp16:
    keys = np.memmap(args.dstore_keys, dtype=np.float16, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_vals, dtype=np.int, mode='r', shape=(args.dstore_size, 1))
else:
    keys = np.memmap(args.dstore_keys, dtype=np.float32, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_vals, dtype=np.int, mode='r', shape=(args.dstore_size, 1))


# from https://github.com/numpy/numpy/issues/13172
# to speed up access to np.memmap
madvise = ctypes.CDLL("libc.so.6").madvise
madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
madvise.restype = ctypes.c_int
assert madvise(keys.ctypes.data, keys.size * keys.dtype.itemsize, 1) == 0, "MADVISE FAILED" # 2 means MADV_SEQUENTIAL

if not os.path.exists(args.train_index):
    # Initialize faiss index
    index_dim = args.pca if args.pca > 0 else args.dimension
    quantizer = faiss.IndexFlatL2(index_dim)
    index = faiss.IndexIVFPQ(quantizer, index_dim,
        args.ncentroids, args.code_size, 8)
    index.nprobe = args.probe

    if args.pca > 0:
        pca_matrix = faiss.PCAMatrix(args.dimension, args.pca, 0, True)
        index = faiss.IndexPreTransform(pca_matrix, index)

    np.random.seed(args.seed)
    random_sample = np.random.choice(np.arange(args.actual_dstore_size), size=[min(1000000, vals.shape[0])], replace=False)

    # ensure sequential reading
    random_sample.sort()

    start = time.time()

    print('reading index', flush=True)
    # Faiss does not handle adding keys in fp16 as of writing this.
    x = keys[random_sample].astype(np.float32)
    print(f'reading indexing took {time.time() - start} seconds')

    print('Training Index begins', flush=True)
    start = time.time()
    index.train(x)
    print('Training took {} s'.format(time.time() - start), flush=True)

    print('Writing index after training', flush=True)
    start = time.time()
    faiss.write_index(index, args.train_index)
    print('Writing index took {} s'.format(time.time()-start), flush=True)

print('Adding Keys', flush=True)
index = faiss.read_index(args.train_index)
print(f'read trained index from {args.train_index}', flush=True)
start = args.starting_point
start_time = time.time()
while start < args.actual_dstore_size:
    end = min(args.actual_dstore_size, start+args.num_keys_to_add_at_a_time)
    to_add = keys[start:end].copy()
    index.add_with_ids(to_add.astype(np.float32), np.arange(start, end))
    start += args.num_keys_to_add_at_a_time

    if (start % 1000000) == 0:
        print('Added %d tokens so far' % start)
        print('Writing Index', start, flush=True)
        faiss.write_index(index, args.faiss_index)

print("Adding total %d keys" % start)
print('Adding took {} s'.format(time.time() - start_time))
print('Writing Index')
start = time.time()
faiss.write_index(index, args.faiss_index)
print('Writing index took {} s'.format(time.time()-start))
