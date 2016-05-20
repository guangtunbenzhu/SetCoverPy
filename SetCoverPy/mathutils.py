__license__ = "MIT"
__author__ = "Guangtun Ben Zhu (BGT) @ Johns Hopkins University"
__startdate__ = "2015.01.11"
__name__ = "mathutils"
__module__ = "SetCoverPy"
__python_version__ = "3.5.1"
__numpy_version__ = "1.11.0"
__scipy_version__ = "0.17.0"

__lastdate__ = "2016.05.13"
__version__ = "0.9.0"


__all__ = ['quick_amplitude', 'quick_totalleastsquares']

""" 
mathutils.py

    This piece of software is developed and maintained by Guangtun Ben Zhu, 

    Copyright (c) 2015-2016 Guangtun Ben Zhu

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
    and associated documentation files (the "Software"), to deal in the Software without 
    restriction, including without limitation the rights to use, copy, modify, merge, publish, 
    distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or 
    substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
    BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


import numpy as np
import scipy.optimize as op

def quick_amplitude(x, y, x_err, y_err, niter=5):
    """
    Purpose - Assume y = Ax, calculate the amplitude with an approximate maximum-likelihood estimator
            - chi2_i = Sum_j (y_ij - A_i*x_ij)*weight_ij, where weight_ij = 1/(yerr_ij^2 + A^2*(xerr_ij^2))
            - Vectorized
            - Fast 
            - Robust if errors of the independent variable (x_err) are relatively small 
    Input - 2D arrays: x, y, x_err, y_err
          - The first dimension is the number of samples/vectors
          - The second dimension is the number of data points (or the dimension) of each sample/vector
          - niter: number of iterations, default 5
    Output - amplitude, chi2_squared
    Caveats - Both x and y should have (mostly) the same sign
            - The MLE is an approximate and iterative estimator, the default niter is 5
            - Use quick_totalleastsquares() if y_err are large
            - See arXiv:1008.4686 by Hogg et al. for a (correct) Bayesian treament instead
    """

    xy = x*y
    xx = x*x
    # we need x and y to have the same sign
    xy[xy<0] = 1E-10

    # begin iteration
    A = np.ones(x.shape[0])
    for i in np.arange(niter):
        weight = 1./(np.square(y_err)+np.square(A).reshape(A.size,1)*np.square(x_err))
        A = np.einsum('ij, ij->i', xy, weight)/np.einsum('ij, ij->i', xx, weight)

    chi2 = np.einsum('ij, ij->i', np.square(A.reshape(A.size,1)*x - y), weight)

    return (A, chi2)

def quick_totalleastsquares(x, y, x_err, y_err, niter=5):
    """
    Purpose - Assume y = Ax, calculate the amplitude with an approximate maximum-likelihood estimator
            - chi2_i = Sum_j (y_ij - A_i*x_ij)*weight_ij, where weight_ij = 1/(yerr_ij^2 + A^2*(xerr_ij^2))
            - Vectorized
            - Slow, because it uses scipy's optimization code minimize()
            - Robust even if errors of the independent variable (x_err) are large
    Input - 2D arrays: x, y, x_err, y_err
          - The first dimension is the number of samples/vectors
          - The second dimension is the number of data points (or the dimension) of each sample/vector
          - niter: number of iterations, default 5
    Output - amplitude, chi2_squared
    Caveats - Both x and y should have (mostly) the same sign
            - The MLE is an approximate and iterative estimator, the default niter is 5
            - Use quick_amplitude() if y_err are small for speed-up. It is much faster.
            - See arXiv:1008.4686 by Hogg et al. for a (correct) Bayesian treament instead
    """
    # chi2 function for op.minimize
    chi2 = lambda A: np.einsum('ij, ij->i', np.square(y-A.reshape(A.size,1)*x), \
                        1./(np.square(y_err)+np.square(A).reshape(A.size,1)*np.square(x_err)))

    xy = x*y
    xx = x*x
    # we need x and y to have the same sign
    xy[xy<0] = 1E-10

    # begin iteration, to get a good initial guess
    A0 = 0.
    for i in np.arange(n_iter):
        weight = 1./(np.square(y_err)+np.square(A0)*np.square(x_err))
        A0 = np.einsum('ij, ij->i', xy, weight)/np.einsum('ij, ij->i', xx, weight)

    # optimization in case of large errors for y
    res = op.minimize(chi2, A0)

    return (res, chi2(res['x']))
