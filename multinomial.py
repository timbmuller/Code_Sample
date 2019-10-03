import warnings

import numpy
import scipy.stats

from . import base


class MultinomialSpline(base.ProbSpline):
    # Poisson parameter is nonnegative.
    _parameter_min = 0
    
    _parameter_max = 1
    
    _alpha = 1e-8
   
    @staticmethod
    def _loglikelihood(Y, mu):
        V = scipy.stats.multinomial.logpmf(Y.T, numpy.sum(Y,axis=0), mu.T.clip(10**-10))
        return V

    @classmethod
    def _transform(cls, Y):
        p = Y/numpy.sum(Y,axis=0)
        return numpy.log(p[0:-1]+cls._alpha)-numpy.log(p[-1]+cls._alpha)

    @classmethod
    def _transform_inverse(cls, q):
        # Silence warnings.
        p = numpy.zeros(q.shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j in range(len(q)):
                p[j] = 1/(numpy.exp(-q[j])+numpy.sum(numpy.exp(q-q[j]),axis=0))
            val = 1/(1+numpy.sum(numpy.exp(q),0))
            if numpy.ndim(q) == 1:
                p = numpy.hstack((p,val))
            else:
                p = numpy.row_stack((p,val))
            
            return p