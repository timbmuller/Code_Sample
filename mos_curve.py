import abc
import numbers

import numpy
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import sklearn.base
import sklearn.utils.validation
import pandas as pd
from . import base
import prob_spline
import matplotlib.pyplot as pyplot

#msq_file = "Vector_Data(NoZeros).csv"

class MosCurve():
	'''
	Utilizing the PoissonSpline code and a datafile consisting of
	the sampled mosquito counts to generate the individual splines for each bird
	
	NOTE : We do not intend to use this in our final model, rather as a method to test the ODE
	'''


	def __init__(self, data_file, MosClass,sigma = 0, period=prob_spline.period(), sample=0):

		msg = 'datafile must be a string'
		assert isinstance(data_file, str), msg
		
		self.data_file = data_file

		self.read_data()
		self.X=prob_spline.time_transform(self.time)

		assert (sigma >= 0), 'sigma must be nonnegative.'

		self.curves = MosClass(data_file,sigma=sigma,period=period,sample=sample)

	def read_data(self):

		count_data = pd.read_csv(self.data_file,index_col=0)
		self.birdnames = count_data.index
		self.time = numpy.array([int(x) for x in count_data.columns])
		self.Y = count_data.as_matrix()
		return()
	
	def evaluate(self,X):			# Evaluate the splines at given values X
		return(numpy.array(self.curves(X)))

	__call__ = evaluate

	def derivative(self,X):
		return(numpy.array(self.curves.derivative(X)))

	def pos_der(self,X):
		return(numpy.array(numpy.max((self.derivative(X),0))))
	
	def neg_der(self,X):
		return(numpy.array(numpy.min((self.derivative(X),0))))
	

