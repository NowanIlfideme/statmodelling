#!/usr/bin/env python
"""
Author: Anatoly Makarevich
Laboratory work #1

"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt


class RandGenerator(object):
    """Random generator base object. Can be used as an iterable."""

    def __init__(self):pass
    def __iter__(self): return self
    def __next__(self): raise StopIteration()
    def gen(self, n): return [next(self) for i in range(n)]
    pass

class ArrayGen(RandGenerator):
    """Not a generator, but a convenience wrapper."""
    def __init__(self, arry, idx=0):
        self.arry = arry
        self.idx = idx
    def __next__(self):
        res = self.arry[self.idx]
        self.idx += 1
        if self.idx>len(self.arry): self.idx = 0
        return res
    pass

class NpUniform(RandGenerator):
    """Wrapping numpy generator as an iteratable."""
    def __next__(self): return np.random.uniform(0,1)
    def gen(self, n): return np.random.uniform(0,1,n).tolist()
    pass

class mult_congruent(RandGenerator):
    """Multiplicative-congruent pseudo-random generator.
    Returns a continuous uniform distribution on [0;1]."""

    def __init__(self, beta=65539, M=2**31, alpha_star_0=None):
        self.beta = beta
        self.M = M
        if alpha_star_0 is None: alpha_star_0=beta
        self.alpha_star = alpha_star_0
        pass
    def __next__(self):
        self.alpha_star = (self.beta * self.alpha_star) % self.M
        return float(self.alpha_star) / float(self.M)
    pass

class mclaren_marcelli(RandGenerator):
    """Mclaren-Marcelli pseudo-random generator.
    Returns a continuous uniform distribution on [0;1]."""

    def __init__(self, K, base_b, base_c):
        self.base_b = base_b
        self.base_c = base_c
        self.K = K
        self.V = self.base_b.gen(K)
        pass
    def __next__(self):
        s = int(next(self.base_c) * self.K)
        res = self.V[s]
        self.V[s] = next(self.base_b)
        return res
    pass




def test_bsv_moment(data, eps=None):
    A = np.array(data)
    n = len(A)

    m = np.mean(A)
    s2 = 1/float(n-1)*np.sum((A-m)**2)

    c1 = np.sqrt(12*n)
    c2 = float(n-1)/n * (0.0056/n+0.0028/n/n-0.0083/(n**3))**(-0.5)

    P_val_1 = 2*(1-ss.norm.cdf( c1*abs(m-0.5) ))
    P_val_2 = 2*(1-ss.norm.cdf( c2*abs(s2-1.0/12.0) ))

    if eps is None: return (P_val_1, P_val_2) #test1 and test2

    test1 = (P_val_1 > eps)
    test2 = (P_val_2 > eps)
    return test1 and test2

def test_bsv_covar(data, eps=None):
    A = np.array(data)
    n = len(A)

    m = np.mean(A)
    R_hat = lambda j: 1/float(n-j-1) * np.sum(A[:n-j])*A[j] - n/float(n-1)*m*m
    R = lambda j: 1.0/12.0 if j==0 else 0
    c = lambda j: np.sqrt(2.0) if j==0 else 1.0

    t = int(n**0.5)
    res = []
    for j in range(t):
        zj = 12*np.sqrt(n-1) * abs( R_hat(j) - R(j) ) / c(j)
        Pj = 2*( 1 - ss.norm.cdf(zj) )
        res.append(Pj)
    if eps is None: return tuple(res)
    for j in range(t):
        if ( res[j] > eps ): return False
    return True

def test_bsv_2d(data, eps=None):
    A = np.array(data)
    n = len(A)

    m = n//2
    A2 = A[:m*2]
    A2 = A2.reshape((m,2))

    k = 10

    r = ([0 if i==0 else float((i)*0.5/k) for i in range(k)]) #for [0; k-1]
    p = ([(1-np.pi*r[k-1]**2) if i==k else (np.pi*(r[i]**2-r[i-1]**2)) for i in range(k+1)]) #0th is trash

    bin = [0 for i in range(k+1)]
    for i in range(m):
        found = False
        for j in range(1, k):
            if np.sqrt(np.sum( (A2[i]-[0.5,0.5])**2 ))<r[j]:
                bin[j] += 1
                found = True
                break
        if not found: bin[k] +=1
        pass

    chi_sq = 0
    for i in range(1,k+1): chi_sq += (bin[i]-m*p[i])**2 / (m*p[i])
    #

    P_val = 1 - ss.chi2.cdf(chi_sq, k-1) #k-1 degress of freedom

    if eps is None: return P_val
    return P_val > eps

def test_bsv_pearson(data, eps=None):
    K = 10
    # K bins in a histogram
    # p[k] = 1.0/K
    step = 1.0/K

    A = np.array(data)
    n = len(A)

    nu = np.zeros((K,))
    for k in range(K):
        nu[k] = np.sum( (A>=step*k) & (A<step*(k+1)) )
        
    chi_sq = np.sum( (nu-n*step)**2 / (n*step) )

    P_val = 1.0 - ss.chi2.cdf(chi_sq, K-1)
    if eps is None: return P_val
    return P_val > eps

def test_bsv_kolmogorov(data, eps=None):

    A = np.array(data)
    n = len(A)
    B = np.sort(A)
    step = 1.0/n

    # B is sorted, basically empirical CDF
    D = 0.0
    for i in range(n):
        dia = abs(B[i] - step*i)
        dib = abs(step*(i+1) - B[i])
        if dia>D: D = dia
        if dib>D: D = dib
        pass

    #K  - kolmogorov distro
    tau = n #???
    K = lambda y : 1 - 2 * sum( (-1)**(j-1) * np.exp(-2*j*j*y*y) for j in range(1, tau+1) )

    P_val = 1.0 - K(np.sqrt(n)*D)
    if eps is None: return P_val
    return P_val > eps


def test_bsv_all(data, eps):
    if eps is None: print("[ Printing test P-values. ]")
    else: print("[ Passes at epsilon = %s? ]" % eps)

    # Moment test
    print("Moment test: %s" % (test_bsv_moment(data, eps),) )
    # Covar test
    print("Covar test: %s" % (test_bsv_covar(data, eps),) )
    # 2d test
    print("2d distro test: %s" % (test_bsv_2d(data, eps),) )
    # Pearson test
    print("Pearson test: %s" % (test_bsv_pearson(data, eps),) )
    # Kolmogorov test
    print("Kolmogorov test: %s" % (test_bsv_kolmogorov(data, eps),) )

    pass

def z1(m, eps=None):

    #Mult-congr
    print("\nMultiplicative-congruent method:")
    g1 = mult_congruent(2**m + 3)
    d1 = g1.gen(1000)
    plt.hist(d1)
    test_bsv_all(d1, eps)
    plt.show()


    #Maclaren-Marcelli
    print("\nMaclaren-Marcelli method:")
    K = 1000
    g2 = mclaren_marcelli(K, g1, NpUniform())
    d2 = g2.gen(1000)
    plt.hist(d2)
    test_bsv_all(d2, eps)
    plt.show()


    return (d1, d2)


if __name__ == "__main__":

    res1 = z1(24, 0.05)


    print("Finished.")
else:
    pass


