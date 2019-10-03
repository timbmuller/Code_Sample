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



class Execute(abc.ABC):

	def Host_Spline(self,data_file,sigma):		# Set up as subclass with ability to call?
		time,mat = self.read_data(data_file)
		X=prob_spline.time_transform(time)
		splines = self.get_host_splines(X,mat,sigma)
		return(splines)

	def Vector_Spline(self,data_file,sigma):	# Set up as a subclass with the ability to call?
		time,mat = self.read_data(data_file)
		X = prob_spline.time_transform(time)
		spline = self.get_vector_splines(self,X,mat.T,sigma)

	def get_vector_splines(self,X,Y,sigma):
		multinomial_spline = prob_spline.MultinomialSpline(sigma = sigma,period = prob_spline.period())
		multinomial_spline.fit(X, Y)
		return(multinomial_spline)


	def read_data(self,data_file):
		count_data = pd.read_csv(data_file,index_col=0)
		time = numpy.array([int(x) for x in count_data.columns])
		mat = count_data.as_matrix()
		return(time,mat)

	def get_host_splines(self,X,Y_mat,sigma):
		splines=[]
		for i in range(len(Y_mat)):
			Y = numpy.squeeze(Y_mat[i,:])
			poisson_spline = prob_spline.PoissonSpline(sigma = sigma, period=prob_spline.period())
			poisson_spline.fit(X, Y)
			splines.append(poisson_spline)
		return(splines)

	def evaluate(self,splines,X):			# Evaluate the splines at given values X
		return(numpy.array([splines[i](X) for i in range(len(splines))]))

	def derivative(self,splines,X):
		return(numpy.array([splines[i].derivative(i) for i in range(len(splines))]))

	def generate_samples(self,data,n_samples,distribution = 'p'):
		return (numpy.random.poisson(lam=data,size = (n_samples,len(data))))


