"""
Kernel Ridge Regression and KRR ensembling.
"""
import numpy as np
from scipy import linalg
from utils import generate_sets


class KRR:
	"""
	Kernel Ridge Regression class.
	"""
	def __init__(self, gram_fn=None):
		self._gram_fn = gram_fn

	def fit(self, X, Y, l):
		"""
		Fitting the model given X and Y.
			returns: Gram matrix, alpha vector
		"""
		self._X = X

		# Computing the Gram matrix
		K = self._gram_fn(X).astype(float)

		return self.fit_K(K, Y, l)

	def fit_K(self, K, Y, l):
		"""
		Alternative with pre-computed Gram matrix.
		"""
		# Solving the ridge regression problem
		nI = K.shape[0]*np.identity(K.shape[0])
		A_n = K + l*nI
		self._alpha = linalg.solve(A_n, Y)

		return K, self._alpha

	def predict(self, Xt):
		"""
		Predicting classes given a test matrix Xt.
			returns: prediction vector
		"""
		# Computing the Gram matrix for X,Xt
		K = self._gram_fn(X=self._X, Xt=Xt)

		return self.predict_K(K)

	def predict_raw(self, Xt):
		"""
		Returning raw outputs from a test matrix Xt.
			returns: output vector
		"""
		# Computing the Gram matrix for X,Xt
		K = self._gram_fn(X=self._X, Xt=Xt)

		return (self._alpha @ K)

	def predict_K(self, K):
		"""
		Alternative with pre-computed Gram matrix.
		"""
		return (self._alpha @ K > 0.5).astype(int)


class KRRBag:
	"""
	Bagging regression models.
	"""
	def __init__(self, n_models, ratio, l,
				 gram_fn=None, gram_train=None, gram_test=None):
		"""
			- n_models:   Number of models in the bag
			- ratio:	  	Ratio of the full data to train each model with
			- gram_fn:	  Function to use to compute Gram matrices
			- gram_train, Pre-computed gram matrices
			  gram_test
			- l:		  		Regularization parameter
		"""
		self._n_models   = n_models
		self._ratio      = ratio
		self._l          = l
		self._gram_fn    = gram_fn
		self._gram_train = gram_train
		self._gram_test  = gram_test

	def fit(self, X, Y, verbose=False):
		"""
		Fitting n KRR models with data from X|Y.
		"""
		# Generating n random datasets
		dsets = generate_sets(X, Y,
							  ratio=self._ratio,
							  n_models=self._n_models)

		self._bag = []

		for idx,(Xi, Yi) in enumerate(dsets):
			if verbose: print("Fitting KRM[%d]..." % idx, end="")

			krr = KRR(self._gram_fn)
			krr.fit(Xi, Yi, self._l)

			self._bag += [krr]

			if verbose: print("Done!")

	def fit_K(self, Y, verbose=False):
		"""
		Fitting n KRR models with data from K|Y.
		"""
		# Generating n random datasets
		dsets = generate_sets(np.zeros_like(Y), Y,
							  ratio=self._ratio,
							  n_models=self._n_models)
		self._bag = []

		for idx,(_, Yi) in enumerate(dsets):
			if verbose: print("Fitting KRM[%d]..." % idx, end="")

			K = self._gram_train[idx]
			krr = KRR()
			krr.fit_K(K, Yi, self._l)

			self._bag        += [krr]

			if verbose: print("Done!")

	def predict(self, Xt, verbose=False):
		"""
		Making predictions based on the model bag.
		"""
		preds = []
		tot_preds = np.zeros_like(Xt.shape[0])

		for idx,m in enumerate(self._bag):
			if verbose: print("Predicting KRM[%d]..." % idx, end="")

			pred = m.predict(Xt)
			preds += [pred]
			tot_preds = np.add(pred, tot_preds)

			if verbose: print("Done!")
	
		return (tot_preds/self._n_models > 0.5).astype(int), preds

	def predict_K(self, verbose=False):
		"""
		Making predictions based on the model bag.
		"""
		preds = []
		tot_preds = np.zeros_like(self._gram_test[0].shape[1])

		for idx,m in enumerate(self._bag):
			if verbose: print("Predicting KRM[%d]..." % idx, end="")

			K = self._gram_test[idx]
			pred = m.predict_K(K)
			preds += [pred]
			tot_preds = np.add(pred, tot_preds)

			if verbose: print("Done!")
	
		return (tot_preds/self._n_models > 0.5).astype(int), preds
