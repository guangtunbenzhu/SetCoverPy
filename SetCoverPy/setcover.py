__license__ = "MIT"
__author__ = "Guangtun Ben Zhu (BGT) @ Johns Hopkins University"
__startdate__ = "2015.12.10"
__name__ = "setcover"
__module__ = "SetCoverPy"
__python_version__ = "3.5.1"
__numpy_version__ = "1.11.0"
__scipy_version__ = "0.17.0"

__lastdate__ = "2016.05.13"
__version__ = "0.9.0"


__all__ = ['SetCover']

""" 
setcover.py

    This piece of software is developed and maintained by Guangtun Ben Zhu, 
    It is designed to find an (near-)optimal solution to the set cover problem (SCP) as 
    fast as possible. It employs an iterative heuristic approximation method, combining 
    the greedy and Lagrangian relaxation algorithms.

    For the standard tests (4,5,6, A-H instances from Beasley's OR Library), the code yields 
    solutions that are 96%-100% optimal (see scp_BeasleyOR_test_results.txt). 

    As all the codes, this code can always be improved and any feedback will be greatly appreciated.

    And here are some small tips using this SCP solver:
      - If you are not satisfied with the solution, just run it a few more times
        (when your servers are free) and select the best solution.
        Or you can also select a larger maxiters at the instantiation:

        >> g = setcover.SetCover(a_matrix, cost, maxiters=100)

      - If you are really really lazy and don't want to wait for a near-optimal solution,
        you can just run the greedy solver, which takes no time:

        >> g.greedy()


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


import warnings

import numpy as np
# Using the dot product in the sparse module can speed up set intersection operation drastically
from scipy import sparse
from time import time

# Some magic numbes
_stepsize = 0.1
_largenumber = 1E5
_smallnumber = 1E-5

class SetCover:
    """
    Set Cover Problem - Find a set of columns that cover all the rows with minimal cost.

    Algorithm:
        -- greedy: 
        -- Lagrangian relaxation:
    Input: 
        -- a_matrix[mrows, ncols], the covering binary matrix, 
           a_matrix[irow, jcol] = True if jcol covers irow
        -- cost[ncols], the cost of columns. 
           I recommend using normalized cost: cost/median(cost)

    (Use A.4 instance from Beasley's OR library,
     http://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html, as an example)
    Instantiation: 
        >> a_matrix = np.load('./BeasleyOR/scpa4_matrix.npy')
        >> cost = np.load('./BeasleyOR/scpa4_cost.npy')
        >> g = setcover.SetCover(a_matrix, cost)
    Run the solver: 
        >> solution, time_used = g.SolveSCP()

    Output:
        -- g.s, the (near-optimal) minimal set of columns, a binary 1D array, 
           g.s[jcol] = True if jcol is in the solution
        -- g.total_cost, the total cost of the (near-optimal) minimal set of columns

    Comments:
        -- 

    References:
        -- Guangtun Ben Zhu, 2016
           A New View of Classification in Astronomy with the Archetype Technique: 
           An Astronomical Case of the NP-complete Set Cover Problem
           AJ/PASP, (to be submitted)
        -- Caprara, A., Fischetti, M. and Toth, P., 1999 
           A heuristic method for the set covering problem 
           Operations research, 47(5), pp.730-743.
        -- Fisher, M.L., 2004 
           The Lagrangian relaxation method for solving integer programming problems 
           Management science, 50(12_supplement), pp.1861-1871.
        -- Balas, E. and Carrera, M.C., 1996 
           A dynamic subgradient-based branch-and-bound procedure for set covering 
           Operations Research, 44(6), pp.875-890.
        -- Beasley, J.E., 1990  
           OR-Library: distributing test problems by electronic mail  
           Journal of the operational research society, pp.1069-1072.
        -- And many more, please see references in Zhu (2016).

    To_do:
        -- Better converging criteria needed (to cut time for small instances)
    History:
        -- 14-May-2016, Alternate initialization methods between random and greedy, BGT, JHU
        -- 19-Apr-2016, Documented, BGT, JHU
        -- 10-Dec-2015, Started, BGT, JHU
        -- DD-MMM-2010, Conceived, BGT, NYU
    """

    def __init__(self, amatrix, cost, maxiters=20, subg_nsteps=15, subg_maxiters=100):
        """
        Initialization
        Required argument:
            amatrix
            cost
        Keywords:
            maxiters - the maximum number of iterations for the re-initialization, 
                       20 by default. This is the parameter you may want to increase
                       to get a better solution (at the expense of time)
            subg_nsteps - how many steps for each subgradient iteration, 15 by default
                          Note each step includes _subg_nadaptive (hard-coded at 20) mini steps
            subg_maxiters - the maximum number of iterations for the subgradient phase, 
                            100 by default

        """
        self.a = np.copy(amatrix)
        # Compressed sparse row
        self.a_csr = sparse.csr_matrix(amatrix, copy=True)
        # Compressed sparse column (transposed for convenience)
        self.a_csc = sparse.csr_matrix(amatrix.T, copy=True)
        self.c = np.copy(cost)
        self.mrows = amatrix.shape[0]
        self.ncols = amatrix.shape[1]

        # Some Magic Numbers based on the Beasley's test bed

        ## subgradient method magic numbers
        self._stepsize = _stepsize
        # how many steps we look back to decide whether to increase or decrease the stepsize
        self._subg_nadaptive = 20
        self._subg_nsteps = self._subg_nadaptive*subg_nsteps
        # Maximum iterations we want to perturb the best u and then recalculate
        self._subg_maxiters = subg_maxiters
        self._subg_maxfracchange = 0.000020 # convergence criteria, fractional change
        self._subg_maxabschange = 0.010 # convergence criteria, absolute change
        self._max_adapt = 0.06 # threshold to half the stepsize
        self._min_adapt = 0.002 # threshold to increase the stepsize by 1.5
        self._u_perturb = 0.06 # perturbations

        ## re-initialization magic numbers
        self._maxiters = maxiters
        self._maxfracchange = 0.001 # convergence criterion, fractional change
        self._LB_maxfracchange = 0.050 # update criterion for Lower Bound

        # setting up
        self.f_uniq = self._fix_uniq_col() # fix unique columns
        self.f = np.copy(self.f_uniq) # fixed columns, only using f_uniq for now
        self.f_covered = np.any(self.a[:,self.f], axis=1) # rows covered by fixed columns
        self.s = np.copy(self.f_uniq) # (current) solution, selected column
        self.u = self._u_naught() # (current best) Lagrangian multiplier

    @property
    def total_cost(self):
        """
        Total cost of a given set s
        """
        return np.einsum('i->', self.c[self.s])

    @property
    def fixed_cost(self):
        """
        Total cost of a given set s
        """
        return np.einsum('i->', self.c[self.f])

    def reset_all(self):
        """
        Reset the parameters to start over
        """
        self._stepsize = _stepsize
        self.reset_f()
        self.reset_s()
        self.reset_u()

    def reset_s(self):
        """
        Reset s, the selected columns
        """
        self.s = np.copy(self.f_uniq) # (current) solution, selected column

    def reset_f(self):
        """
        Reset f, the fixed columns
        """
        self.f = np.copy(self.f_uniq)
        self.f_covered = np.any(self.a[:,self.f], axis=1)

    def reset_u(self, random=False):
        """
        Reset u, the Lagrangian multiplier
        """
        if (random):
            self.u = self._u_naught_simple()
        else:
            self.u = self._u_naught()

    def _u_naught_simple(self):
        """
        Initial guess of the Lagrangian multiplier with random numbers
        """
        # Random is better to give different multipliers in the subgradient phase
        return np.random.rand(self.mrows)*1. 

    def _u_naught(self):
        """
        Initial guess of the Lagrangian multiplier with greedy algorithm
        This is the default initializer
        """
        adjusted_cost = self.c/self.a_csc.dot(np.ones(self.mrows))
        cost_matrix = adjusted_cost*self.a + np.amax(adjusted_cost)*(~self.a)
        return adjusted_cost[np.argmin(cost_matrix, axis=1)]

    def _fix_uniq_col(self):
        """
        Fix the unique columns that have to be in the minimal set
        """
        # subgradient; for two boolean arrays, multiplication seems to be the best way 
        # (equivalent to logical_and)
        n_covered_col = self.a_csr.dot(np.ones(self.ncols)) 
        ifix = np.zeros(self.ncols, dtype=bool)
        if (np.count_nonzero(n_covered_col) != self.mrows):
           raise ValueError("There are uncovered rows! Please check your input!")
        if (np.any(n_covered_col==1)):
           inonzero = self.a_csr[n_covered_col==1,:].nonzero()
           ifix[inonzero[1]] = True

        return ifix


    def greedy(self, u=None, niters_max=1000):
        """
        Heuristic greedy method to select a set of columns to cover all the rows
        start from the initial set
        run the following first if you want to reset the initial selection with fixed columns
            - self.reset_s() or 
            - self.s = np.logical_or(self.s, self.f)
        """

        niters = 1
        if (u is None):
            u = self.u

        utmp = np.copy(u)
        iuncovered = ~np.any(self.a[:,self.s], axis=1)
        
        score = np.zeros(self.ncols)
        while (np.count_nonzero(iuncovered) > 0) and (niters <= niters_max):
            # It's 5 times faster without indexing, the advantage is made possible by csc_matrix.dot
            mu = (self.a_csc.dot((iuncovered).astype(int))).astype(float) 
            mu[mu<=_smallnumber] = _smallnumber

            utmp[~iuncovered] = 0
            gamma = (self.c - self.a_csc.dot(utmp))
            select_gamma = (gamma>=0)

            if (np.count_nonzero(select_gamma)>0):
                score[select_gamma] = gamma[select_gamma]/mu[select_gamma]
            if (np.count_nonzero(~select_gamma)>0):
                score[~select_gamma] = gamma[~select_gamma]*mu[~select_gamma]

            inewcolumn = (np.nonzero(~self.s)[0])[np.argmin(score[~self.s])]
            self.s[inewcolumn] = True
            iuncovered = ~np.logical_or(~iuncovered, self.a[:,inewcolumn])
            niters = niters+1
        if (niters == niters_max): 
           warnings.warn("Iteration in Greedy reaches maximum = {0}".format(niters_max))
        return self.total_cost

    def update_core(self):
        """
        Removing fixed columns
        """
        if (~np.any(self.f)):
           a_csr = sparse.csr_matrix(self.a, copy=True) # Compressed sparse row
           a_csc = sparse.csr_matrix(self.a.T, copy=True) # Compressed sparse column (transposed)
        else:
           a_csr = sparse.csr_matrix(self.a[:,~self.f][~self.f_covered,:], copy=True)
           a_csc = sparse.csr_matrix(self.a[:,~self.f][~self.f_covered,:].T, copy=True)
        return (a_csr, a_csc)

    def subgradient(self):
        """
        Subgradient step for the core problem N\S. 
        """
        
        UB_full = self.total_cost
        ufull = np.copy(self.u)

        # Update core: possible bottleneck
        (a_csr, a_csc) = self.update_core()
        mrows = a_csr.shape[0]
        ncols = a_csr.shape[1]
        u_this = self.u[~self.f_covered]
        # np.einsum is 20% faster than np.sum ...
        UB_fixed = self.fixed_cost
        UB = UB_full - UB_fixed
        cost = self.c[~self.f]

        # save nsteps calculations (Lagrangian multipliers and lower bounds)
        u_sequence = np.zeros((mrows, self._subg_nsteps)) 
        Lu_sequence = np.zeros(self._subg_nsteps)
        # update u
        x = np.zeros(ncols, dtype=bool)
        niters_max = self._subg_maxiters
        maxfracchange = self._subg_maxfracchange
        maxabschange = self._subg_maxabschange

        # initialization
        f_change = _largenumber
        a_change = _largenumber
        niters = 0
        Lu_max0 = 0
        while ((f_change>maxfracchange) or (a_change>maxabschange)) and (niters<niters_max):
            u_this = (1.0+(np.random.rand(mrows)*2.-1)*self._u_perturb)*u_this
            u_sequence[:,0] = u_this
            cost_u = cost - a_csc.dot(u_sequence[:,0]) # Lagrangian cost
            # next lower bound of the Lagrangian subproblem
            Lu_sequence[0] = np.einsum('i->', cost_u[cost_u<0])+np.einsum('i->', u_sequence[:,0]) 

            for i in np.arange(self._subg_nsteps-1):
                # current solution to the Lagrangian subproblem
                x[0:] = False
                x[cost_u<0] = True

                # subgradient; for two boolean arrays, multiplication seems to be the best way 
                # (equivalent to logical_and)
                s_u = 1. - a_csr.dot(x.astype(int)) 
                s_u_norm = np.einsum('i,i',s_u,s_u) # subgradient's norm squared

                # Update
                # next Lagrangian multiplier
                u_temp = u_sequence[:,i]+self._stepsize*(UB - Lu_sequence[i])/s_u_norm*s_u 
                u_temp[u_temp<0] = 0

                u_sequence[:,i+1] = u_temp
                cost_u = cost - a_csc.dot(u_sequence[:,i+1]) # Lagrangian cost
                # next lower bound of the Lagrangian subproblem
                Lu_sequence[i+1] = np.einsum('i->', cost_u[cost_u<0])+np.einsum('i->', u_sequence[:,i+1]) 
            
                #print(UB_full, UB, Lu_sequence[i+1])
                # Check the last nadaptive steps and see if the step size needs to be adapted
                if (np.mod(i+1,self._subg_nadaptive)==0):
                    Lu_max_adapt = np.amax(Lu_sequence[i+1-self._subg_nadaptive:i+1])
                    Lu_min_adapt = np.amin(Lu_sequence[i+1-self._subg_nadaptive:i+1])
                    if (Lu_max_adapt <= 0.):
                        Lu_max_adapt = _smallnumber
                    f_change_adapt = (Lu_max_adapt-Lu_min_adapt)/np.fabs(Lu_max_adapt)
                    if f_change_adapt > self._max_adapt:
                        self._stepsize = self._stepsize*0.5
                    elif (f_change_adapt < self._min_adapt) and (self._stepsize<1.5):
                        self._stepsize = self._stepsize*1.5
                    # swap the last multiplier with the optimal one
                    i_optimal = np.argmax(Lu_sequence[i+1-self._subg_nadaptive:i+1])
                    if (i_optimal != (self._subg_nadaptive-1)):
                       u_temp = u_sequence[:,i]
                       u_sequence[:,i] = u_sequence[:,i+1-self._subg_nadaptive+i_optimal]
                       u_sequence[:,i+1-self._subg_nadaptive+i_optimal] = u_temp
                       Lu_sequence[i+1-self._subg_nadaptive+i_optimal] = Lu_sequence[i]
                       Lu_sequence[i] = Lu_max_adapt

            i_optimal = np.argmax(Lu_sequence)
            Lu_max = Lu_sequence[i_optimal]
            u_this = u_sequence[:,i_optimal]
            niters = niters + 1
            a_change = Lu_max - Lu_max0
            f_change = a_change/np.fabs(Lu_max)
            Lu_max0 = Lu_max # Just a copy. Not the reference (It's a number object)
            # save current u_this???

            if (niters == niters_max): 
                warnings.warn("Iteration in subgradient reaches maximum = {0}".format(niters))

        # update multipliers
        self.u[~self.f_covered] = u_this

        # return the last nsteps multipliers
        # save nsteps calculations (Lagrangian multipliers and lower bounds)
        u_sequence_full = np.zeros((self.mrows, self._subg_nsteps)) 
        Lu_sequence_full = np.zeros(self._subg_nsteps)
        u_sequence_full[self.f_covered,:] = self.u[self.f_covered][:, np.newaxis]
        u_sequence_full[~self.f_covered,:] = u_sequence

        Lu_sequence_full = Lu_sequence + self.fixed_cost

        return (u_sequence_full, Lu_sequence_full)

    def subgradient_solution(self, u=None):
        """
        """
        if (u is None):
            u = np.copy(self.u)
        cost_u = self.c - self.a_csc.dot(u) # Lagrangian cost
        x = np.zeros(self.ncols, dtype=bool) # has to be integer to use scipy sparse matrix
        x[cost_u<0] = True # current solution to the Lagrangian subproblem
        return x

    def SolveSCP(self):
        """
        The wrapper, Solve the SCP
        """

        t0 = time()

        # Some predicates
        Lu_min = 0.
        niters_max = self._maxiters
        maxfracchange = self._maxfracchange

        # initialization, resetting ...
        self.reset_all() # including _u_naught(), first application
        scp_min = self.greedy()

        # re-initialization iteration; col fixing ignored for the moment
        niters = 0
        f_change = _largenumber
        while (f_change>maxfracchange) and (niters<niters_max):
            # re-initialize u
            if (np.mod(niters, 2)==0): 
                self.reset_u(random=True)
            else:
                self.reset_u()
            u_tmp, Lu_tmp = self.subgradient() # find a near-optimal solution 
            u, Lu = self.subgradient() # rerun subgradient to get a set of Lagrangian multipliers

            scp_all = np.zeros(self._subg_nsteps)
            for i in np.arange(self._subg_nsteps):
                #self.reset_s()
                self.s = np.copy(self.f)
                scp_all[i] = self.greedy(u=u[:,i])

            # check if the solution is gettting better
            imin_tmp = (np.where(scp_all==np.amin(scp_all)))[0]
            imin = imin_tmp[np.argmax(Lu[imin_tmp])]
            imax = np.argmax(Lu)
            if (np.mod(niters, 5)==0):
                print("This Best solution: UB={0}, LB={1}, UB1={2}, LB1={3}".format(scp_all[imin], Lu[imin], scp_all[imax], Lu[imax]))
            if (niters==0) or ((scp_all[imin]<=scp_min) and ((Lu[imin]-Lu_min)>-(np.fabs(Lu_min)*self._LB_maxfracchange))):
                scp_min = scp_all[imin]
                u_min = np.copy(u[:,imin])
                Lu_min = Lu[imin]
                self.stepsize = _stepsize

            LB = Lu_min

            # final step, needs to get u_min back
            self.u = np.copy(u_min)
            self.s = np.copy(self.f)
            UB = self.greedy()

            # Which is better? absolute change or fractional change? 
            # Both are fine, but cost should be normalized over the mean/median.
            GAP = (UB-LB)/np.fabs(UB)
            f_change = GAP
            if (np.mod(niters, 5)==0):
                print("Current Best Solution: UB={0}, LB={1}, change={2}% @ niters={3}".format(UB,LB,f_change*100.,niters))
            niters = niters + 1
            if (niters == niters_max): 
                #warnings.warn("Iteration reaches maximum = {0}".format(niters))
                print("Iteration in re-initialization reaches maximum number = {0}".format(niters))

        # Need to remove redundant columns
        # self.remove_redundant() # this itself is NP-hard ...

        print("Current Best Solution: UB={0}, LB={1}, change={2}% @ niters={3}".format(UB,LB,f_change*100.,niters))
        print("Final Best solution: {0}".format(UB))
        time_used = (time()-t0)/60.
        print("Took {0:.3f} minutes to reach current solution.".format(time_used))

        return (UB,time_used) 
