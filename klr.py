"""
Kernel Logistic Regression and KLR ensembling.
"""
import numpy as np
from scipy import linalg
from utils import generate_sets


def _s(x):
	"""
	Numerically stable sigmoid function.
	"""
	v = np.where(x >= 0, 
				 1 / (1 + np.exp(-x)), 
				 np.exp(x) / (1 + np.exp(x)))
	return v

def _solve_WKRR(K, w, z, l):
	"""
	Solving a Weighted Kernel Ridge Regression problem.
	"""
	# alpha = sqrt(W).(sqrt(W).K.sqrt(W) + nLI)^-1 .sqrt(W)z
	# Using the fact that W is a diagonal matrix
	W = np.diag(np.sqrt(w))
	nI = K.shape[0]*np.identity(K.shape[0])

	M = linalg.inv(W @ K @ W + l*nI)
	alpha = W @ M @ W @ z

	return np.nan_to_num(alpha)


class KLR:
	"""
	Kernel Logistic Regression using IRLS.
	"""
	def __init__(self, gram_fn=None, stop_criterion=0.1, max_it=5, l=1e-4):
		"""
			- gram_fn:	      Function to use to compute Gram matrices
			- stop_criterion: Minimum ||α(t)-α(t+1)|| value after which
							  				the algorithm is stopped.
			- max_it:	      	Maximum number of iterations
							  				(overrides the stop criterion).
			- l:		      		Regularization parameter
		"""
		self._gram_fn        = gram_fn
		self._stop_criterion = stop_criterion
		self._max_it         = max_it
		self._l              = l

	def fit(self, X, Y):
		"""
		Fitting the model given X and Y.
			returns: Gram matrix, alpha vector
		"""
		self._X = X

		# Computing the Gram matrix
		K = self._gram_fn(X).astype(float)
		self._Ktr = K

		return self.fit_K(K, Y)

	def fit_K(self, K, Y):
		"""
		Alternative with pre-computed Gram matrix.
		"""
		n = Y.shape[0]

		# Converting labels to the [1,-1] range
		Y_ = np.where(Y==0, -1, Y)

		# IRLS method implementation
		alpha = np.zeros(n)
		gap   = float('Inf')

		self._K = K
  
		# Iterating until convergence
		it = 0
		while (gap > self._stop_criterion):
			# m <- [Kα]
			m = K @ alpha

			# w <- o(m)o(-m) = o(m)(1 - o(m))
			sm = _s(m)
			w  = sm*(1-sm)

			# z <- m + y/o(-ym)
			z = m + Y_/_s(-Y_*m)

			# a' <- _solve_WKRR(K, w, z)
			alpha_p = _solve_WKRR(K, w, z, self._l)

			# Convergence criterion
			gap = np.linalg.norm(alpha_p - alpha)
			alpha = alpha_p

			it +=1
			if it>self._max_it:
				print("[KLR] Maximum number of iterations exceeded.")
				break

		self._alpha = alpha
		return K, self._alpha

	def predict(self, Xt):
		"""
		Predicting classes given a test matrix Xt.
			returns: prediction vector, probabilities vector
		"""
		# Computing the Gram matrix for X,Xt
		Kt = self._gram_fn(X=self._X, Xt=Xt)

		return self.predict_K(Kt)

	def predict_K(self, Kt):
		"""
		Alternative with pre-computed Gram matrix.
		"""
		return (self.predict_prob(Kt) > 0.5).astype(int)

	def predict_prob(self, K):
		"""
		Predicting class probabilities.
		"""
		pred = _s(self._alpha.T @ K)
		return pred


class KLRBag:
	"""
	Bagging KLR models.
	"""
	def __init__(self, n_models, ratio,
				 stop_criterion=0.1, max_it=5, l=1e-4,
				 gram_fn=None, gram_train=None, gram_test=None):
		"""
			- n_models:       Number of models in the bag
			- ratio:	      	Ratio of the full data to train each model with
			- stop_criterion: Minimum ||α(t)-α(t+1)|| value after which
							  				the algorithm is stopped.
			- max_it:	      	Maximum number of iterations
							  				(overrides the stop criterion).
			- l:		     			Regularization parameter
			- gram_fn:	      Function to use to compute Gram matrices
			- gram_train,     Pre-computed gram matrices
			  gram_test
		"""
		self._n_models   = n_models
		self._ratio      = ratio

		self._stop_crit  = stop_criterion
		self._max_it     = max_it
		self._l          = l

		self._gram_fn    = gram_fn
		self._gram_train = gram_train
		self._gram_test  = gram_test

	def fit_K(self, Y, verbose=False):
		"""
		Fitting n KLR models with data from K|Y.
		"""
		# Generating n random datasets
		dsets = generate_sets(np.zeros_like(Y), Y,
							  ratio=self._ratio,
							  n_models=self._n_models)
		self._bag = []

		for idx,(_, Yi) in enumerate(dsets):
			if verbose: print("Fitting KLR[%d]..." % idx, end="")

			K = self._gram_train[idx]
			klr = self._fit_klr_K(K, Yi)
			self._bag += [klr]

			if verbose: print("Done!")

	def predict_K(self, verbose=False):
		"""
		Making predictions based on the model bag.
		"""
		tot_preds = np.zeros_like(self._gram_test[0].shape[1])
		preds = []

		for idx,m in enumerate(self._bag):
			if verbose: print("Predicting KLR[%d]..." % idx, end="")

			Kt = self._gram_test[idx]
			pred = m.predict_prob(Kt)
			preds += [pred]
			tot_preds = np.add(pred, tot_preds)

			if verbose: print("Done!")

		return (tot_preds/self._n_models > 0.5).astype(int), preds

	def _fit_klr(self, X, Y):
		"""
		Fit a single SVM model using dataset X,Y.
		Write the train Gram matrix to disk (or load it).
		"""
		K = self._gram_fn(X)
		return self._fit_klr_K(K, Y)

	def _fit_klr_K(self, K, Y):
		"""
		Alternative using a train Gram matrix.
		"""
		# Fitting the model
		klr = KLR(stop_criterion=self._stop_crit,
			  max_it=self._max_it, l=self._l)
		klr.fit_K(K, Y)

		return klr
