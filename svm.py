"""
SVM implementation and SVM ensembling.
		SVM implementation following a tutorial
		by @mblondel (Google Brain).
"""
import numpy as np
import cvxopt
import cvxopt.solvers
from scipy import linalg
from utils import generate_sets


MIN_LG_MULTIPLIER = 1e-6

class SVMC(object):
	"""
	C-SVM implementation.
	"""
	def __init__(self, gram_fn=None, C=1.):
		self._gram_fn = gram_fn
		self._C = C

	def fit(self, X, Y):
		"""
		Fitting the model given X and Y.
			returns: Computed Gram matrix
		"""
		self._Xtrain = X

		# Computing the Gram matrix
		K = self._gram_fn(X)

		# Fitting the model
		self.fit_K(K, Y)

		return K

	def fit_K(self, K, Y):
		"""
		Alternative with pre-computed Gram matrix.
		"""
		n = len(Y)

		# Converting labels into {-1,1} for convenience
		Y = np.where(Y==0, -1, Y)

		# Solving the quadratic programming problem
		P = cvxopt.matrix(np.outer(Y,Y)*K)
		q = cvxopt.matrix(np.ones(n)*-1)
		A = cvxopt.matrix(Y, (1,n), 'd')
		b = cvxopt.matrix(0.0)

		tmp1 = np.diag(np.ones(n)*-1)
		tmp2 = np.identity(n)
		G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
		tmp1 = np.zeros(n)
		tmp2 = np.ones(n)*self._C
		h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

		cvxopt.solvers.options['show_progress'] = False
		a = np.ravel(cvxopt.solvers.qp(P, q, G, h, A, b)['x'])

		# Finding support vectors
		sv = a > MIN_LG_MULTIPLIER
		ind = np.arange(len(a))[sv]
		self._alpha = a[sv]

		self._sv   = np.argwhere(sv)
		self._sv_y = Y[sv]

		# Computing the intercept
		self._b = 0
		for n in range(len(self._alpha)):
				self._b += self._sv_y[n]
				self._b -= np.sum(self._alpha*self._sv_y*K[ind[n], sv])
		self._b /= len(self._alpha)

	def predict(self, Xt):
		"""
		Predicting classes given a test matrix Xt.
			returns: prediction vector
		"""
		# Computing the Gram matrix for X,Xt
		K = self._gram_fn(self._Xtrain, Xt)

		return self.predict_K(K)

	def predict_K(self, K):
		"""
		Alternative with pre-computed Gram matrix.
		"""
		pred = np.zeros(K.shape[1])
		for i in range(K.shape[1]):
				pred[i] = sum(alpha*sv_y*K[sv,i] for alpha, sv, sv_y
											in zip(self._alpha, self._sv, self._sv_y))
		pred = pred + self._b

		return (pred >= 0.).astype(int)

#=======================================

class SVMBag:
	"""
	Bagging SVM models.
	"""
	def __init__(self, n_models, ratio, svm, C=1.,
				 gram_fn=None, gram_train=None, gram_test=None):
		"""
			- n_models:   Number of models in the bag
			- ratio:	  	Ratio of the full data to train each model with
			- svm:		  	Function to use to create SVM instances
			- C:		  		Regularization parameter (default 1)
			- gram_fn:	  Function to use to compute Gram matrices
			- gram_train, Pre-computed gram matrices
				gram_test
		"""
		self._n_models   = n_models
		self._ratio      = ratio
		self._svm	       = svm
		self._C          = C
		self._gram_fn    = gram_fn
		self._gram_train = gram_train
		self._gram_test  = gram_test

	def fit(self, X, Y, verbose=False):
		"""
		Fitting n SVM models with data from X|Y.
		"""
		# Generating n random datasets
		dsets = generate_sets(X, Y,
								ratio=self._ratio,
								n_models=self._n_models)

		self._bag = []
		self._train_sets = []
		for idx,(Xi, Yi) in enumerate(dsets):
			if verbose: print("Fitting SVM[%d]..." % idx, end="")

			svm = self._fit_svm(Xi, Yi)
			self._bag        += [svm]
			self._train_sets += [Xi]

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
			if verbose: print("Fitting SVM[%d]..." % idx, end="")

			K = self._gram_train[idx]
			svm = self._fit_svm_K(K, Yi)
			self._bag += [svm]

			if verbose: print("Done!")

	def predict(self, Xt, verbose=False):
		"""
		Making predictions based on the model bag.
		"""
		tot_preds = np.zeros_like(Xt.shape[0])
		preds = []

		for idx,Xi in enumerate(self._train_sets):
			if verbose: print("Predicting SVM[%d]..." % idx, end="")

			pred = self._pred_svm(Xi, Xt, idx)
			preds += [pred]
			tot_preds = np.add(pred, tot_preds)

			if verbose: print("Done!")

		return (tot_preds/self._n_models > 0.).astype(int), preds

	def predict_K(self, verbose=False):
		"""
		Making predictions based on the model bag.
		"""
		tot_preds = np.zeros_like(self._gram_test[0].shape[1])
		preds = []

		for idx,m in enumerate(self._bag):
			if verbose: print("Predicting SVM[%d]..." % idx, end="")

			Kt = self._gram_test[idx]
			pred = m.predict_K(Kt)
			preds += [pred]
			tot_preds = np.add(pred, tot_preds)

			if verbose: print("Done!")

		return (tot_preds/self._n_models > 0.).astype(int), preds

	def _fit_svm(self, X, Y):
		"""
		Fit a single SVM model using dataset X,Y.
		Write the train Gram matrix to disk (or load it).
		"""
		K = self._gram_fn(X)
		return self._fit_svm_K(K, Y)

	def _fit_svm_K(self, K, Y):
		"""
		Alternative using a train Gram matrix.
		"""
		# Fitting the model
		svc = self._svm(C=self._C)
		svc.fit_K(K, Y)

		return svc

	def _pred_svm(self, X, Xt, idx):
		"""
		Computing the test Gram matrix associated to X ~ Xt.
		Write the test Gram matrix to disk (or load it).
		"""
		Kt = self._gram_fn(X, Xt)
		svc = self._bag[idx]
		return svc.predict(Kt)
