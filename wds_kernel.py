"""
Weighted Degree Kernel [G.Ratsch - S.Sonnenburg, 2004]
"""


def wd_kernel(s1, s2, d = 5):
	n = len(s1)
	val = 0
	
	for k in range(1, d + 1):
		beta = 2 * (d - k + 1) / (d * (d + 1))
		s = 0
		for i in range(n - k + 1):
			chunk = i + k
			s += int(s1[i:chunk] == s2[i:chunk])
		val += (beta * s)
			
	return val

def wds_kernel(s1, s2, d=5, S=2):
	n = len(s1)
	val = 0

	for k in range(1, d + 1):
		beta = 2 * (d - k + 1) / (d * (d + 1))
		tot = 0
		for i in range(n - k + 1):
			for s in range(S + 1):
				i_s = i + s
				if i_s > n:
					break
				delta = 1 / (2 * (s + 1))
				tot += (delta * (int(s1[i_s: i_s + k] == s2[i: i + k]) + int(s2[i_s: i_s + k] == s1[i: i + k])))
		val += (beta * tot)

	return val
