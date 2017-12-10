#!/usr/bin/env python
"""
Author: Anatoly Makarevich
Laboratory work #4

Variant 8

Use Monte Carlo methods to solve:

1. Integral (0, pi, exp(-x^4) * sqrt(1+x^2) dx)
2. Integral(0, 1, Integral(0, 2, (x^2+y^2)dx)dy)

3. Solve system:
A = 
 1.2 -0.3  0.4
 0.4  0.7 -0.2
 0.2 -0.3  0.9
f = (-4, 2, 0)^T

Find area:
4. 1 <= x^2+y^2 <= 4
5. { y>x^2, y<sqrt(x) }



"""

import numpy as np
import matplotlib.pyplot as plt

import Lab1, Lab2, Lab3
from sympy import *
from sympy.abc import *

class Uniform2D(Lab1.RandGenerator):
    """(x,y) uniform generator."""

    def __init__(self, xrange = (0,1), yrange = (0,1)):
        self.xrange = xrange
        self.yrange = yrange
        pass

    def __next__(self):
        x = np.random.uniform(*self.xrange)
        y = np.random.uniform(*self.yrange)
        return (x, y)

    @property
    def area(self):
        xr = self.xrange[1]-self.xrange[0]
        yr = self.yrange[1]-self.yrange[0]
        return xr*yr

    pass


# INTEGRATION

def mc_integral_1d(expr, gen, N=1000, conf_int=0):
    """Find integral of $expr given $gen."""

    vals = np.empty(N)
    total = 0.0
    for i in range(N):
        _x = next(gen)
        vals[i] = float(expr.subs(x,_x))
        total += vals[i]
        pass
    res = total / N * (gen.hi - gen.lo)

    if conf_int==0: return res
    # Variance
    var = np.var(vals) / sqrt(N) * (gen.hi - gen.lo)
    return (float(res - conf_int * var), float(res + conf_int * var))

def mc_integral_2d(expr, gen, N=1000, conf_int=0):
    """Find integral of $expr given $gen."""

    vals = np.empty(N)
    total = 0.0
    for i in range(N):
        _x, _y = next(gen)
        vals[i] = float(expr.subs(x,_x).subs(y,_y))
        total += vals[i]
        pass
    res = total / N * gen.area

    if conf_int==0: return res
    # Variance
    var = np.var(vals) / sqrt(N) * gen.area
    return (float(res - conf_int * var), float(res + conf_int * var))


def ex1(N=1000, n_sigma=3):
    """Calculate Integral (0, pi, exp(-x^4)sqrt(1+x^2) dx)"""
    func = ( exp(-x**4)*sqrt(1+x*x)  )
    gen = Lab3.ContinuousUniform(0, np.pi)
    res = mc_integral_1d(func, gen, N=N, conf_int=n_sigma)
    print("Integral is approx. %s\n (%s-sigma conf interval is %s ) " % 
          ((res[0]+res[1])/2, n_sigma, res))
    actual_res = float( integrate(func,(x,0,np.pi)) )
    print("Actual integral should be %s" % (actual_res))
    pass

def ex2(N=1000, n_sigma=3):
    """Calculate Integral(0,1,Integral(0,2,(x^2+y^2)dx)dy)"""
    func = (x*x + y*y)
    gen = Uniform2D((0,1),(0,2))
    res = mc_integral_2d(func, gen, N=N, conf_int=n_sigma)
    print("Integral is approx. %s\n (%s-sigma conf interval is %s ) " % 
          ((res[0]+res[1])/2,n_sigma, res))
    actual_res = float( integrate(func,(x,0,1),(y,0,2)) )
    print("Actual integral should be %s" % (actual_res))
    pass


# LINEAR ALGEBRA SYSTEM

class MarkovChain(object):
    """Defines a Markov chain."""

    def __init__(self, P, pi=None, stop_states=[], max_iters=np.inf):
        self.P = P
        self.pi = pi
        
        if stop_states is None or isinstance(stop_states,  list):
            self.stop_states = stop_states
        else: self.stop_states = [stop_states]
        self.max_iters = max_iters

        self.curr_state = None
        pass

    def __iter__(self): 
        self.curr_iter = 0

        if self.pi is None:
            self.curr_state = int(np.random.uniform(low=0, high=sz))
        else:
            self.curr_state = np.random.choice(len(self.P), p=self.pi)
        return self

    def __next__(self):
        if self.curr_iter>self.max_iters: raise StopIteration()
        if self.curr_state is None: raise StopIteration()
        res = self.curr_state
        if self.curr_state in self.stop_states: 
            self.curr_state=None
        else: self.curr_state = int(np.random.choice( len(self.P), 
                    p = np.array(self.P[self.curr_state, :]).squeeze()) )
        self.curr_iter += 1
        return res
    pass



def mc_solve_linsystem(A, f, l=10000, N=1000):
    """Solves a linear system of the form x=Ax+f."""
    
    _do_check = True
    if _do_check and np.max(np.abs(np.linalg.eigvals(A)))>1:
        raise ValueError("Eigenvalues must be <1 for matrix T!")

    sz = len(A)

    res = np.empty(sz)
    for j in range(sz): # j is coordinate
        # unit vector per coordinate
        h = np.zeros(sz)
        h[j] = 1

        #Markov chain:
        # pi (initial probs) are 1/sz for all states.
        pi = np.full(shape=(sz,), fill_value=1.0/sz)
    
        # P (transition matrix) have equal probs, except absorbing state
        P = np.matrix(np.full(shape=(sz, sz), fill_value = 1.0/(sz) ))

        # g[i], G[i,j]
        g = h/pi
        g[pi==0] = 0

        G = A/P
        G[P==0] = 0

        def ksi(N):
            MC = MarkovChain(P, pi=pi).__iter__()
            im = next(MC)

            Qm = g[im]
            res = Qm * f[im,0]

            for m in range(1, N):
                im1 = im
                im = next(MC)

                Qm = Qm * G[im1, im]
                res += Qm * f[im,0]
            return res

        res[j] = sum(ksi(N) for i in range(l))/l
        pass
    return np.matrix(res).transpose()

def ex3(l=10, N=100):
    """Solve system Ax+f=x for x, given:
    A = 
     1.2 -0.3  0.4
     0.4  0.7 -0.2
     0.2 -0.3  0.9
    f = (-4, 2, 0)^T
    """

    A = np.matrix(
        [[-0.2,0.3,-0.4],
         [-0.4,0.3,0.2],
         [-0.2,0.3,0.1]]
        )
    #A = np.matrix([[0.4, -0.3, 0.4], [0.4, 0.7, -0.2], [0.2, -0.3, 0.9]])
    f = np.matrix([-4, 2, 0]).transpose()

    x = np.matrix(
        Matrix(A - np.diag([1 for i in range(len(A))])).solve(Matrix(-f)),
        ).astype('float')
    print("Actual solution:\n%s" % x)
    print("Delta:\n%s\n" % (A * x + f - x))

    res = mc_solve_linsystem(A.transpose(), f, l=l, N=N)
    print("Approximate solution:\n%s" % res)
    delta = A * res + f - res
    print("Delta:\n%s\n" % delta)
    return res

# AREA

def find_area(expr, gen, N=1000):
    """Find area of $expr (x,y) using 2D $gen in a Monte Carlo sim"""

    accepted = 0
    for i in range(N):
        _x, _y = next(gen)
        if bool(expr.subs(x,_x).subs(y,_y)):
            accepted += 1
        pass

    return float(accepted) / N * gen.area

def ex4(N=1000):
    """ Find area: 1 <= x^2+y^2 <= 4 """
    fig = (x*x+y*y<=4) & (x*x+y*y>=1)
    gen = Uniform2D((-2,2), (-2,2))

    area = find_area(fig, gen, N)
    print("Area is approx. %s" % area)
    print("Actual area should be %s" % (np.pi * (4-1)))
    pass

def ex5(N=1000):
    """ Find area: 5. { y>x^2, y<sqrt(x) } """
    fig = (y>x*x) & (y<sqrt(x))
    gen = Uniform2D((0,2), (0,2))

    area = find_area(fig, gen, N)
    print("Area is approx. %s" % area)
    expected = integrate(sqrt(x),(x,0,1)) - integrate(x*x,(x,0,1))
    print("Actual area should be %s" % (expected))
    pass

if __name__ == "__main__":
    def_n = 1000

    print("*****\nIntegrals:")
    #ex1(def_n)
    #ex2(def_n)

    #print("*****\nLinear algebra:")
    #ex3(l=def_n, N=100)

    print("*****\nAreas:")
    #ex4(def_n)
    #ex5(def_n)

    pass
else:
    pass


