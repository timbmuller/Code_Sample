import abc
import numbers

import numpy
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import sklearn.base
import sklearn.utils.validation


def identity(Y):
    '''
    Null transform.
    '''
    return Y


class ProbSpline(sklearn.base.BaseEstimator, abc.ABC):
    '''
    A smoothing spline using a loglikelihood as the error metric
    instead of the usual squared distance.
    '''
    _parameter_min = - numpy.inf
    _parameter_max = numpy.inf

    _transform = _transform_inverse = staticmethod(identity)

    def __init__(self, degree = 3, sigma = 0, period=1):
        msg = 'degree must be a nonnegative integer.'
        assert isinstance(degree, numbers.Integral) and (degree >= 0), msg
        self.degree = degree
        assert (sigma >= 0), 'sigma must be nonnegative.'
        self.sigma = sigma
        self.period = period

    @abc.abstractmethod
    def _loglikelihood(self, Y, mu):
        '''
        This must be defined by subclasses.
        '''

    def fit(self, X, Y):
        Y = Y.T
        self.knots_ = X
        if self.sigma == 0:
            self._fit_interpolating_spline(X, Y)
        else:
            self._fit_smoothing_spline(X, Y)

    def log_derivative(self,X):
        '''
        The Y values of the derivative of the spline at X.
        '''
        # Make sure _coef is defined.
        sklearn.utils.validation.check_is_fitted(self, 'coef_')
        # Find which interval the x values are in.
        ix = (numpy.searchsorted(self.knots_-self.knots_[0], (X-self.knots_[0]) % self.period, side='right') - 1)
        # Handle scalar vs vector x.
        if numpy.isscalar(ix):
            ix = [ix]
        # Get the coefficients in those intervals.
        # Make sure this gets stacked correctly in the multinomial case, want new dimension in the front
        #numpy.stack doesn't work, column/h stack

        coef = numpy.stack(
            [numpy.polyder(self.coef_[... , (self.degree + 1) * i : (self.degree + 1) * (i + 1)].T)
             for i in ix],axis=-1)
        # Evaluate the polynomials in those intervals at the X values.

        dz = numpy.polyval(coef, ((X - self.knots_[0]) % self.period) - (self.knots_[ix]-self.knots_[0]))
        mu = self._transform_inverse_log_der(X,dz)

        if numpy.isscalar(X):
            mu =numpy.squeeze(mu)
        return mu

    def predict(self, X):
        '''
        The Y values of the spline at X.
        '''
        # Make sure _coef is defined.
        sklearn.utils.validation.check_is_fitted(self, 'coef_')
        # Find which interval the x values are in.
        ix = (numpy.searchsorted(self.knots_-self.knots_[0], (X-self.knots_[0]) % self.period, side='right') - 1)
        # Handle scalar vs vector x.
        if numpy.isscalar(ix):
            ix = [ix]
        # Get the coefficients in those intervals.
        # Make sure this gets stacked correctly in the multinomial case, want new dimension in the front
        #numpy.stack doesn't work, column/h stack

        coef = numpy.stack(
            [self.coef_[... , (self.degree + 1) * i : (self.degree + 1) * (i + 1)].T
             for i in ix],axis=-1)
        # Evaluate the polynomials in those intervals at the X values.
        z = numpy.polyval(coef, ((X - self.knots_[0]) % self.period) - (self.knots_[ix]-self.knots_[0]))
        mu = self._transform_inverse(z).clip(self._parameter_min,
                                             self._parameter_max)
        if numpy.isscalar(X):
            mu =numpy.squeeze(mu)
        return mu

    __call__ = predict

    def score(self, X, Y):
        Y=Y.T
        '''
        The likelihood.
        '''
        mu = self.predict(X)
        return numpy.exp(numpy.sum(self._loglikelihood(Y, mu)))

    def _fit_interpolating_spline(self, X, Y, continuity_matrix = None):
        if continuity_matrix is None:
            if len(Y.shape)==1:
                continuity_matrix = self._get_continuity_matrix(X, Y)
            else:
                continuity_matrix = self._get_continuity_matrix(X, Y[0])
        # The continuity conditions.
        # A_C.dot(coef) = 0.
        if len(Y.shape)==1:
            A_C = continuity_matrix
            m_C, n_C = A_C.shape
            b_C = numpy.zeros(m_C)
            # The interpolating conditions.
            # A_I.dot(coef) = Y.
            m_I = len(self.knots_)
            n_I = (self.degree + 1) * len(self.knots_)
            A_I = scipy.sparse.lil_matrix((m_I, n_I))
            # At all the X values except the last one,
            # we just have that the constant term in the
            # polynomial is equal to the corresponding
            # (transformed) Y value.
            rows = numpy.arange(len(self.knots_), dtype = int)
            # The constant terms.
            cols = rows * (self.degree + 1) + self.degree
            A_I[rows, cols] = 1
            b_I = self._transform(Y)
            A = scipy.sparse.vstack((A_C, A_I))
            b = numpy.hstack((b_C, b_I))
            self.coef_ = scipy.sparse.linalg.spsolve(A.tocsr(), b)
        else:  # Code to handle the grouped probabilities for a multinomial distribution
            p = len(Y)
            A_C = continuity_matrix
            m_C, n_C = A_C.shape
            b_C = numpy.hstack([numpy.zeros(m_C)]*(p-1))
            # The interpolating conditions.
            # A_I.dot(coef) = Y.
            m_I = len(self.knots_)
            n_I = (self.degree + 1) * len(self.knots_)
            A_I = scipy.sparse.lil_matrix((m_I, n_I))
            # At all the X values except the last one,
            # we just have that the constant term in the
            # polynomial is equal to the corresponding
            # (transformed) Y value.
            rows = numpy.arange(len(self.knots_), dtype = int)
            # The constant terms.
            cols = rows * (self.degree + 1) + self.degree
            A_I[rows, cols] = 1
            A_I_block = scipy.sparse.block_diag([A_I]*(p-1))
            A_C_block = scipy.sparse.block_diag([A_C]*(p-1))
            A = scipy.sparse.vstack((A_I_block,A_C_block))
            b_I = self._transform(Y)
            b_I = numpy.hstack(b_I)
            b = numpy.hstack((b_I,b_C))       
            #placeholder for the coefficients so we can unpack them
            coef = scipy.sparse.linalg.spsolve(A.tocsr(), b)
            #Return a matrix with the coefficients as the rows?
            print(coef.shape)
            #self.coef_ = coef
            self.coef_ = numpy.reshape(coef,(p-1,len(self.knots_)*(self.degree+1)))


    def _get_continuity_matrix(self, X, Y):
        '''
        Build the matrix for the continuity conditions
        at the internal knots.
        The product of the resulting matrix and the vector of coeffecients
        is 0 when the conditions are all satisfied.
        '''
        m = self.degree * len(self.knots_)
        n = (self.degree + 1) * len(self.knots_)
        continuity_matrix = scipy.sparse.lil_matrix((m, n))
        dX = numpy.diff(numpy.hstack((X,X[0]+self.period)))
        # At each internal knot,
        # the jth derivatives are continuous
        # for j = 0, 1, ..., degree - 1.
        for j in range(self.degree):
            constants = numpy.polyder(numpy.ones(self.degree + 1),
                                      m = j)
            exponents = numpy.arange(self.degree - j, -1, -1)
            for i in range(len(self.knots_)):
            #for i in range(len(self.knots_) - 1):
                #row = self.degree * i + j + 1
                row = self.degree * i + j
                # The columns of the terms in
                # the jth derivative of the ith polynomial.
                cols = slice((self.degree + 1) * i,
                             (self.degree + 1) * (i + 1) - j)
                continuity_matrix[row, cols] = constants * dX[i] ** exponents
                # The column of the constant term in
                # the jth derivative of the (i + 1)st polynomial.
                # This is set up to be modular so that it wraps around for our boundary condition
                # col = ((self.degree + 1) * (i + 2) - 1 - j)
                col = ((self.degree + 1) * (i + 2) - 1 - j) % n
                continuity_matrix[row, col] = - constants[-1]
        return continuity_matrix

    def _fit_smoothing_spline(self, X, Y):
        '''
        Fit the smoothing spline by minimizing
        - loglikelihood + sigma * \int |f''(x)| dx.
        '''
        # The continuity matrix is needed to get the initial
        # guess for the coefficients (_fit_interpolating_spline())
        # and by _continuity_constraints, so build it once now.
        if len(Y.shape)==1:
            continuity_matrix = self._get_continuity_matrix(X, Y)
        else:
            continuity_matrix = self._get_continuity_matrix(X,Y[0])
        #    A_C = self._get_continuity_matrix(X, Y[0])
        #    continuity_matrix = scipy.sparse.block_diag([A_C]*(p-1))
        # Get the initial guess for the coefficients
        # from the interpolating spline.
        self._fit_interpolating_spline(X, Y,
                                       continuity_matrix = continuity_matrix)
        coef_initial_guess = numpy.ravel(self.coef_)
        # Set some constants so that the objective function
        # doesn't recompute them every time it is called.
        
        # I think we need this to be the complete matrix?
        if len(Y.shape)!=1:
            p = len(Y)
            continuity_matrix = scipy.sparse.block_diag([continuity_matrix]*(p-1))

        dX = numpy.diff(numpy.hstack((X,X[0]+self.period)))
        deriv1 = numpy.polyder(numpy.ones(self.degree + 1),
                               m = self.degree - 2)
        deriv2 = numpy.polyder(numpy.ones(self.degree + 1),
                               m = self.degree - 1)
        adjustment_constant = (deriv2[1] / deriv2[0]
                               * (deriv1[1]
                                  - deriv1[0] * deriv2[1] / deriv2[0]))
        objective_args = (X, Y, dX, deriv1, deriv2, adjustment_constant)
        constraints = dict(fun = self._continuity_constraints,
                           args = (continuity_matrix, ),
                           type = 'eq')
        options = dict(maxiter = 1000)
        result = scipy.optimize.minimize(self._objective,
                                         coef_initial_guess,
                                         constraints = constraints,
                                         args = objective_args,
                                         options = options)
        if not result.success:
            # Don't leave self.coef_ set.
            del self.coef_
            msg = 'Optimization failed: {}.'.format(result.message)
            print(msg)
            #raise RuntimeError(msg)
        if len(Y.shape)>1:
            self.coef_ = numpy.reshape(result.x,(p-1,len(self.knots_)*(self.degree+1)))
        else:
            self.coef_ = result.x

    def _objective(self, coef, X, Y, dX, deriv1, deriv2, adjustment_constant):
        '''
        The objective function for optimizing the smoothing spline,
        - loglikelihood + sigma * \int |f''(x)| dx.
        '''
        if len(Y.shape)>1:
            p = len(Y)
            coef =  numpy.reshape(coef,(p-1,len(self.knots_)*(self.degree+1)))
        self.coef_ = coef
        mu = self.predict(X)
        loglikelihood = numpy.sum(self._loglikelihood(Y, mu))
        # Calculate the variation integral.
        a = coef[... , 0 : : self.degree + 1]
        b = coef[... , 1 : : self.degree + 1]
        # Check if 2nd derivative is 0 inside each interval.
        condition1  = (a * b < 0)
        condition2 = (numpy.abs(deriv2[1] * b) < numpy.abs(deriv2[0] * a * dX))
        haszero = condition1 & condition2
        variations = dX * (deriv1[0] * a * dX + deriv1[1] * b)
        adjustments = numpy.ma.divide(2 * adjustment_constant * b ** 2, a)
        variations[haszero] += adjustments[haszero]
        # Sum over pieces of the spline.
        variation = numpy.sum(numpy.abs(variations))
        #consider floor of loglikelihood/box constraints on coefficients
        return - loglikelihood + self.sigma * variation

    def _continuity_constraints(self, coef, continuity_matrix):
        '''
        The continuity constraints function for optimizing the smoothing spline.
        '''
        return continuity_matrix.dot(coef)
