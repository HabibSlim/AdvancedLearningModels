"""
Utility functions.
"""
import numpy as np
import multiprocessing as mp


def cross_validate(X, Y, k_folds, eval_fn):
	"""
	Perform k-folds cross-validation.
	- X,Y:     Dataset to use
	- k_folds: Number of folds
	- eval_fn: Prediction function to use
	"""
	n_train = X.shape[0]

	# Shuffling the dataset
	idx = np.arange(n_train)
	np.random.shuffle(idx)

	# Splitting the dataset
	folds = np.array(np.array_split(idx, k_folds))
	n_folds = folds.shape[0]

	print("Performing CV for %d iterations, with: [%d train sample size, %d test sample size]"
			% (k_folds, n_train-folds[0].shape[0], folds[0].shape[0]))

	tot_acc = []
	for i,fold_idx in enumerate(folds):
		# Initialize test/training chunks
		idx_train = folds[np.arange(n_folds) != i]
		idx_train = np.concatenate(idx_train)

		X_tr_it = X[idx_train]
		X_te_it = X[fold_idx]

		Y_tr_it = Y[idx_train]
		Y_te_it = Y[fold_idx]

		# Evaluate model
		acc = eval_fn(X_tr_it, Y_tr_it, X_te_it, Y_te_it)

		# Add accuracy results
		tot_acc += [acc]

	return np.mean(tot_acc), np.std(tot_acc)

def eval_acc(Y_pred, Y_real):
	"""
	Evaluating the accuracy of given predictions.
	"""
	return np.mean(Y_pred == Y_real)

def generate_sets(X, Y, fix_seed=True, ratio=0.95, n_models=5):
	"""
	Generating n random datasets from X.
		- X,Y: Input dataset
		- n:   Number of datasets to sample
		- r:   Ratio of data to use per subset of X
	"""
	SEED = 1337
	if fix_seed: np.random.seed(SEED)

	dsets = []
	data_size = int(ratio*X.shape[0])
	data_idx  = np.arange(X.shape[0])

	for _ in range(n_models):
		idx = np.random.choice(data_idx, data_size, replace=False)
		dsets += [(X[idx], Y[idx])]

	return dsets

def kernel_worker(f, i, X0, X1):
	"""
	Process worker to compute a Gram matrix.
	"""
	return (i, [f(X0[i], X1[j]) for j in range(X1.shape[0])])

def compute_gram(X, Xt=None, kernel_fun=None):
	"""
	Parallel computation of a spectrum kernel Gram matrix.
		- X:          List of sequences
		- Xt(opt):    Second matrix of sequences
		- kernel_fun: Kernel function to use,
									taking two sequences (x1,x2)
	"""
	n = X.shape[0]
	if Xt is None:
		K = np.zeros((n, n))
		Xt = X
	else:
		K = np.zeros((n, Xt.shape[0]))

	# Multiprocessing for a bit of speedup
	with mp.Pool(mp.cpu_count()) as p:
		sims = [p.apply_async(kernel_worker,
													args=(kernel_fun, i, X, Xt)) for i in range(n)]
		sims = [k.get() for k in sims]

		for i,data in sims:
			if Xt is None:
				K[i, i:n] = K[i:n, i] = data
			else:
				K[i, :] = data

	return K
