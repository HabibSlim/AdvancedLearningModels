"""
Spectrum kernel implementation [C.Leslie, 2002]
"""
import numpy as np
import multiprocessing as mp
from utils import kernel_worker


def sp_kernel(s1, s2, k):
	"""
	Spectrum kernel function.
	"""
	n = len(s1)
	k_dict = {}

	for i in range(n - k + 1):
		chunk = i + k
		sub1, sub2 = s1[i:chunk], s2[i:chunk]

		if sub1 in k_dict:
			k_dict[sub1][0] += 1
		else:
			k_dict[sub1] = [1, 0]

		if sub2 in k_dict:
			k_dict[sub2][1] += 1
		else:
			k_dict[sub2] = [0, 1]

	return sum([v[0] * v[1] for v in k_dict.values()])

def sp_kernel_comb(s1, s2, k_list, weights=None):
	"""
	Linear combination of spectrum kernels.
	"""
	prods = np.array([sp_kernel(s1, s2, k) for k in k_list])

	if weights is None:
		return np.sum(prods)
	else:
		return np.sum(weights*prods)
