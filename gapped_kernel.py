"""
Gapped kernel implementation [H.Lodhi, 2002]
"""
import itertools
import numpy as np


def gapped_k_mer(s1, s2, g=11, k=9):
	"""
	Gapped kernel function.
		- g: Length of g-mer
		- k: Informative positions with g-mer such that gaps m = g - k
	"""
	n = len(s1)
	gk_dict = {}
	m = g - k

	for gaps in itertools.combinations(range(g), m):
		for i in range(0, n - g + 1, 2):
			chunk = i + g
			gk1, gk2 = s1[i:chunk], s2[i:chunk]

			for pos in gaps:
				gk1 = gk1[:pos] + '_' + gk1[pos + 1:]
				gk2 = gk2[:pos] + '_' + gk2[pos + 1:]

			if gk1 in gk_dict:
				gk_dict[gk1][0] += 1
			else:
				gk_dict[gk1] = [1, 0]

			if gk2 in gk_dict:
				gk_dict[gk2][1] += 1
			else:
				gk_dict[gk2] = [0, 1]

	return sum([v[0] * v[1] for v in gk_dict.values()])
