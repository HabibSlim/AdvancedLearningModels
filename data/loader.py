"""
CSV data loaders.
"""
import pandas as pd
import numpy as np


TEST_PATH  = "./data/test/"
TRAIN_PATH = "./data/train/"


def read_train(fname):
	"""
	Reading input CSV file of DNA sequences.
	"""
	return pd.read_csv(TRAIN_PATH + fname + ".csv", encoding='utf-8').to_numpy()[:, 1]

def read_test(fname):
	return pd.read_csv(TEST_PATH + fname + ".csv", encoding='utf-8').to_numpy()[:, 1]

def split_train_test(X, Y, ratio):
	"""
	Splitting a dataset into a train and test sets.
	"""
	n_seq = len(X)
	chunk_1 = int(ratio*n_seq)

	return X[:chunk_1], X[chunk_1:], Y[:chunk_1], Y[chunk_1:]

def save_mat(M, fname):
	"""
	Saving a computed Gram matrix to a NPZ file.
	"""
	np.savez('./%s.npz' % fname, M)

def load_mat(fname):
	return np.load('./%s.npz' % fname)['arr_0']


# Loading files: Training set
__X0, __X1, __X2 = read_train('Xtr0'), read_train('Xtr1'), read_train('Xtr2')
__Y0, __Y1, __Y2 = read_train('Ytr0'), read_train('Ytr1'), read_train('Ytr2')
# Loading files: Test set
__X0_t, __X1_t, __X2_t = read_test('Xte0'), read_test('Xte1'), read_test('Xte2')


def load_train(dset, ratio=1.):
	"""
	Returning a train/test split of a labelled dataset.
	"""
	return {
		0: split_train_test(__X0, __Y0, ratio),
		1: split_train_test(__X1, __Y1, ratio),
		2: split_train_test(__X2, __Y2, ratio),
	}[dset]

def load_test(dset):
	"""
	Returning one of the three unlabelled test datasets.
	"""
	return {
		0: __X0_t,
		1: __X1_t,
		2: __X2_t,
	}[dset]
