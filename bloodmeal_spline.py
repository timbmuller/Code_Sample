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
import scipy.stats
import joblib
from time import gmtime, strftime
'''
bm_file = "Days_BloodMeal.csv"
'''
class BloodmealSpline():
	'''
	Utilizing the MultinomialSpline code and a datafile consisting of
	the sampled bird counts to generate the individual splines for each bird
	'''


	def __init__(self, data_file, sigma = 0, period=prob_spline.period(),sample=0,combine_index=[],remove_index=[],seed=None):
		if seed is not None:
			numpy.random.seed(seed)
			
		self.data_file = data_file

		msg = 'datafile must be a string'
		assert isinstance(data_file, str), msg
		
		msg = 'combine_index and remove_index cannot both be non-empty'
		if combine_index!=[] and remove_index!=[]:
			assert False, msg

		self.remove_index=remove_index
		self.combine_index=combine_index
		self.read_data()


		self.X=prob_spline.time_transform(self.time)

		assert (sigma >= 0), 'sigma must be nonnegative.'

		if sample == 1:
			self.generate_samples()
			self.splines = self.get_vector_spline(self.X,self.samples,sigma,period)
		else:
			self.splines = self.get_vector_spline(self.X,self.Y,sigma,period)

	def read_data(self):	
		
		'''
		Read the given data_file to pull the required data
		'''

		count_data = pd.read_csv(self.data_file,index_col=0)
		self.time = numpy.array([int(x) for x in count_data.columns])
		if self.combine_index==[] and self.remove_index==[]:		
			self.birdnames = list(count_data.index)
			self.Y = count_data.as_matrix()
			self.p = len(self.birdnames)
		elif self.combine_index!=[]:
			birdnames = list(count_data.index)
			Y = count_data.as_matrix()
			self.p = len(birdnames)
			p=len(birdnames)
			holder_matrix = numpy.zeros((p-len(self.combine_index)+1,len(self.time)),dtype=int)
			k=0
			for j in range(p):
				if j in self.combine_index:
					holder_matrix[-1]+=Y[j]
				else:
					holder_matrix[k] = Y[j]
					k+=1
			
			for j in sorted(self.combine_index,reverse=True):
				del birdnames[j]
			
			birdnames.append('Other Birds')
			self.birdnames=birdnames
			self.Y = holder_matrix
		else:
			data = count_data.as_matrix()
			self.Y = numpy.delete(data,(self.remove_index),axis=0)
			birdnames = list(count_data.index)
			birdnames.pop(self.remove_index)
			self.birdnames = birdnames
			self.p = len(self.birdnames)
	
	def get_vector_spline(self,X,Y,sigma,period):
		if len(self.combine_index)==self.p:
			spline= prob_spline.PoissonSpline(sigma = sigma,period=period)
		else:
			spline = prob_spline.MultinomialSpline(sigma = sigma,period = period)
		spline.fit(X,Y.T)
		return(spline)

	def evaluate(self,X):			# Evaluate the splines at given values X
		return(self.splines(X))

	__call__ = evaluate

	def plot(self):
		# Note to self, fix axis values so that all the graphs display the same range
		x = numpy.linspace(numpy.min(self.X), numpy.max(self.X), 1001)
		p = len(self.Y)
		grid = numpy.ceil(numpy.sqrt(p))
		Y = self.Y/numpy.sum(self.Y,axis=0)
		plot_counter = 1
		for j in range(len(Y)):
			pyplot.subplot(3,3,j+1)
			handles=[]
			s = pyplot.scatter(prob_spline.inv_time_transform(self.X),Y[j],label = self.birdnames[j])
			handles.append(s)
			l = pyplot.plot(prob_spline.inv_time_transform(x), self.splines(x)[j],
				label = 'Fitted MultinomialSpline($\sigma =$ {:g})'.format(self.splines.sigma))
			if j==0:
				handles.append(l[0])
			pyplot.legend(handles, [h.get_label() for h in handles])

		pyplot.show()
		return()

	def generate_samples(self):
		p_vals = self.Y/numpy.sum(self.Y,axis=0)
		total_bites = numpy.random.poisson(numpy.sum(self.Y,axis=0),size=(len(numpy.sum(self.Y,axis=0)))) # the sampled total number of bites
		bm_temp = numpy.zeros(self.Y.shape)
		for j in range(len(self.Y.T)):	# number of time points
				bm_temp[:,j] = numpy.random.multinomial(total_bites[j],p_vals[:,j])
		self.samples =bm_temp

		return()

