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
from time import gmtime, strftime

class Seasonal_Spline_ODE():

	def __init__(self, bc_splines, bm_splines, mos_curve,tstart,tend,beta_1=1,find_beta=0,eps=.001,counter=0):
		self.tstart = tstart
		self.tend = tend
		self.time_trans = 365./prob_spline.period()
		self.eps = eps  # The rate at which birds entering the population enter already infected with EEE
		if find_beta==1:
			val = scipy.optimize.minimize(self.findbeta,beta_1,args=(bm_splines,bc_splines,mos_curve),method='COBYLA',options={"disp":False})
			self.beta_1=val.x
			print(counter)
		else:
			self.beta_1 = beta_1
		self.bc_splines = bc_splines
		self.bm_splines = bm_splines
		self.mos_curve = mos_curve
		self.Y = self.run_ode(self.beta_1,bm_splines,bc_splines,mos_curve)

	def alpha_calc(self,bm,counts):   #If 0 bm returns 0, atm if 0 counts returns inf
		bm_rat = bm / numpy.sum(bm)
		count_rat = counts / numpy.sum(counts)
		with numpy.errstate(divide="ignore"):
			#alpha = numpy.where(count_rat>10**-5,bm_rat/count_rat,0)
			alpha = numpy.where(count_rat>0,bm_rat/count_rat,0)
			weight = numpy.sum(alpha*counts,axis=0)
			return(alpha/weight)

	def rhs(self,Y,t, bc_splines, bm_splines, mos_curve):
		# Consider adding epsilon term , for proportion infected entering population eps = .001
		p=self.p
		eps = self.eps
		s=Y[0:p]
		i=Y[p:2*p]
		r=Y[2*p:3*p]
		sv=Y[3*p]
		iv=Y[3*p+1]
		c = Y[3*p+1:4*p+1]   #cumulative infections
		e = Y[4*p+1:5*p+1]	#exposure
		
		transform_constant = 365./prob_spline.period()

		alpha_val = self.alpha_calc(bm_splines(t),bc_splines(t))
		N=bc_splines(t)
		N_v = mos_curve(t)
		denom = numpy.dot(N,alpha_val)
		lambdab = self.beta1*self.v*iv*numpy.array(alpha_val)*N_v/denom
		lambdav = self.v*(numpy.dot(self.beta2*i*N,alpha_val))/denom

		'''
		Note that bc_splines.pos_der returns the normalized dervivative of the spline, that is
		the derivative of the spline at the given time, divided by the value of the spline at
		that given time.
		'''

		ds = bc_splines.pos_der(t)*(1-eps-s) - lambdab*s
		di = bc_splines.pos_der(t)*(eps-i) + lambdab*s - self.gammab*i
		dr = self.gammab*i - r*bc_splines.pos_der(t)
		dsv = mos_curve.pos_der(t)*iv-lambdav*sv + self.dv*iv  
		div = lambdav*sv - mos_curve.pos_der(t)*iv - self.dv*iv

		dc = lambdab*s*N 		#cumulative infections eq
		#de = numpy.sum(s*N)        			#exposure eq
		de = bc_splines.pos_der(t)*N		# proposed change


		dY = numpy.hstack((ds,di,dr,dsv,div,dc,de))  # the 365/2 is the rate of change of the time transform
		return dY

	def run_ode(self,beta1,bm_splines,bc_splines,mos_curve):
		self.p = len(bm_splines.Y)
		self.beta2 = 1
		self.gammab = .1*numpy.ones(self.p)*self.time_trans
		self.v=.14*self.time_trans		# Biting Rate of Vectors on Hosts
		self.b=0*self.time_trans			# Bird "Recruitment" Rate
		self.d=0*self.time_trans			# Bird "Death" Rate
		self.dv=.10*self.time_trans			# Mosquito Mortality Rate
		self.dEEE= 0*self.time_trans	
		self.beta1= beta1
		 # Run for ~ 6 Months
		
		T = scipy.linspace(self.tstart,self.tend,1001)
		Sv = .99
		Iv = .01
		S0 = 1*numpy.ones(self.p)
		I0 = .00*numpy.ones(self.p)
		R0 = 0*numpy.ones(self.p)
		C0 = 0*numpy.ones(self.p)
		E0 = bc_splines(self.tstart)

		Y0 = numpy.hstack((S0, I0, R0, Sv, Iv,C0,E0))
		Y = scipy.integrate.odeint(self.rhs,Y0,T,args = (bc_splines,bm_splines,mos_curve),mxstep = 0, full_output=0)
		return(Y)
		
	def get_SIR_vals(self,Y):		# Takes the values from scipy.integrate.odeint and returns the SIR vals
		p=self.p
		S=Y[:,0:p]
		I=Y[:,p:2*p]
		R=Y[:,2*p:3*p]
		sv=Y[:,3*p]
		iv=Y[:,3*p+1]
		c = Y[:,3*p+2:4*p+2]
		e = Y[:,4*p+2:5*p+2]
		return(S,I,R,sv,iv,c,e)

	def eval_ode_results(self,alpha=1):	
		import pylab
		import seaborn
		self.birdnames = self.bc_splines.birdnames
		colors = seaborn.color_palette('Dark2')+['black']
		seaborn.set_palette(colors)
		name_list = list(self.birdnames)
		name_list.append('Vector')
		T = scipy.linspace(self.tstart,self.tend,1001)
		p = self.p
		s,i,r,sv,iv,c,e = self.get_SIR_vals(self.Y)
		bc = numpy.zeros((p,len(T)))
		bm = numpy.zeros((p,len(T)))
		alpha_val = numpy.zeros((p,len(T)))
		mos_pop = numpy.zeros(len(T))
		bc = self.bc_splines(T)
		bm = self.bm_splines(T)
		alpha_val = self.alpha_calc(self.bm_splines(T),self.bc_splines(T))*self.bc_splines(T)
		mos_pop = self.mos_curve(T)	
		sym = ['b','g','r','c','m','y','k','--','g--']
		pylab.figure(1)
		for k in range(self.p):
			pylab.plot(prob_spline.inv_time_transform(T),bc[k],alpha=alpha)
		pylab.title("Populations")
		pylab.legend(name_list)
		N=s+i+r
		N=numpy.clip(N,0,numpy.inf)
		pylab.figure(2)
		for k in range(self.p):
			temp=i[:,k]
			pylab.plot(prob_spline.inv_time_transform(T),temp,alpha=alpha)	
		pylab.legend(self.birdnames)
		pylab.title("Infected Birds")
		pylab.figure(3)
		for k in range(self.p):
			pylab.plot(prob_spline.inv_time_transform(T),alpha_val[k],alpha=alpha)
		pylab.legend(self.birdnames)
		pylab.title("Feeding Index Values")
		return()

	def findbeta(self,beta1,bm_splines,bc_splines,mos_curve):  
		print(beta1)
		Y = self.run_ode(beta1,bm_splines,bc_splines,mos_curve)
		s,i,r,sv,iv,c,e = self.get_SIR_vals(Y)
		finalrec = numpy.where(numpy.sum(e[-1])>0,numpy.sum(c[-1])/numpy.sum(e[-1]),0)
		final = finalrec-.13
		#print(numpy.abs(final))
		return numpy.abs(final)

			