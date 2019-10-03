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
from time import gmtime, strftime
import joblib

#bc_file = "Days_BirdCounts.csv"

class HostSpline():
	'''
	Utilizing the PoissonSpline code and a datafile consisting of
	the sampled bird counts to generate the individual splines for each bird
	'''


	def __init__(self, data_file, sigma = 0, period=prob_spline.period(), sample = 0, combine_index=[], remove_index=[],
		seed=None):
		if seed is not None:
			numpy.random.seed(seed)


		msg = 'datafile must be a string'
		assert isinstance(data_file, str), msg
		
		msg = 'combine_index and remove_index cannot both be non-empty'
		if combine_index!=[] and remove_index!=[]:
			assert False, msg

		self.remove_index=remove_index
		self.data_file = data_file
		self.combine_index=combine_index

		self.read_data()
		self.X=prob_spline.time_transform(self.time)
		
		'''
		Index here is a list of values correlating to which bird species you want combined
		An empty list indicates each line in the datafile will be examined independently
		A list with specified indices will combine those species, and return a matrix with
		only the non-specified indices, and a final index of the combined species.
		'''

		

		if hasattr(sigma,"__len__"):
			for j in sigma:
				assert (j >=0 ), 'sigma must be nonnegative'
		else:
			assert (sigma >= 0), 'sigma must be nonnegative.'
			sigma = sigma*numpy.ones(len(self.Y))

		if sample==1:
			self.generate_samples()
			self.splines = self.get_host_splines(self.X,self.samples,sigma,period)
		else:
			self.splines = self.get_host_splines(self.X,self.Y,sigma,period)


	def read_data(self):

		count_data = pd.read_csv(self.data_file,index_col=0)
		self.time = numpy.array([int(x) for x in count_data.columns])
		if self.combine_index==[] and self.remove_index==[]:		
			self.birdnames = list(count_data.index)
			self.Y = count_data.as_matrix()
		elif self.combine_index!=[]:
			birdnames = list(count_data.index)
			Y = count_data.as_matrix()
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


		return()
	
	def get_host_splines(self,X,Y_mat,sigma,period):
		splines=[]
		for i in range(len(Y_mat)):
			Y = numpy.squeeze(Y_mat[i,:])
			poisson_spline = prob_spline.PoissonSpline(sigma = sigma[i], period=period)
			poisson_spline.fit(X, Y)
			splines.append(poisson_spline)
		return(splines)

	def evaluate(self,X):			
		'''
		Evalute the splines at the given time X.
		If multiple samples have been generated, evalute the index'th sampled splines at time X
		'''
		return(numpy.array([self.splines[i](X) for i in range(len(self.splines))]))

	__call__ = evaluate

	def log_derivative(self,X):
		return(numpy.array([self.splines[i].log_derivative(X) for i in range(len(self.splines))]))

	def pos_der(self,X):
		return(numpy.clip(self.log_derivative(X),0,numpy.inf))
	
	def neg_der(self,X):
		return(numpy.clip(self.log_derivative(X),-numpy.inf,0))
	

	def plot(self):
		'''
		A function to plot the data and spline fit of the specified species
		Defaults to all species given, but allows for input of specified species index
		'''
		p=range(len(self.Y))
		val = len(p)
		x = numpy.linspace(numpy.min(self.X), numpy.max(self.X), 1001)
		grid = numpy.ceil(numpy.sqrt(val))
		plot_counter = 1
		for j in p:
			handles = []
			pyplot.subplot(grid,grid,plot_counter)
			s = pyplot.scatter(prob_spline.inv_time_transform(self.X), self.Y[j,:], color = 'black',
	                   label = self.birdnames[j])
			handles.append(s)
			l = pyplot.plot(prob_spline.inv_time_transform(x), self.splines[j](x),
				label = 'Fitted PoissonSpline($\sigma =$ {:g})'.format(self.splines[j].sigma))
			handles.append(l[0])
			pyplot.xlabel('$x$')
			pyplot.legend(handles, [h.get_label() for h in handles],fontsize = 'xx-small',loc=0)
			plot_counter+=1
		pyplot.show()
		return()

	def generate_samples(self):
		self.samples = numpy.random.poisson(lam=self.Y,size = (len(self.Y),len(self.Y.T))) 
		return()
