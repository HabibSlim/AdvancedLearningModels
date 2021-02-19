"""
Performing data-augmentation based on reverse complements.
"""
import numpy as np


def reverse_comp(seq):
	"""
	Computing the reverse complement
	of an input sequence.
	"""
	comp = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
	return "".join(comp[b] for b in seq[::-1])

def reverse_mat(X):
	"""
	Building a reverse-complement equivalent of an input matrix.
	"""
	X_r = np.zeros_like(X, dtype=object)
	for i,seq in enumerate(X):
		X_r[i] = reverse_comp(seq)
	return X_r

def augment(X,Y):
	"""
	Performing data-augmentation on dataset (X,Y).
	"""
	return np.concatenate((X, reverse_mat(X))), np.concatenate((Y,Y))
