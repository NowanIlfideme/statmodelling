#!/usr/bin/env python
"""
Author: Anatoly Makarevich
Laboratory work #2

5 discrete distributions (Bernoulli, Binomial, Geometric, Poisson, Discrete Uniform)

Pearson chi-squared test for each

"""

from Lab1 import RandGenerator
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as ss

# 1. Bernoulli
# 2. Binomial
# 3. Geometric
# 4. Negative binomial <- not done
# 5. Hypergeometric <- not done
# 6. Poisson
# 7. Discrete uniform

def chisq_test(bins_theory, bins_actual, eps=None):

    K = len(bins_theory)

    # calculating the statistic
    v = ((bins_actual-bins_theory)**2) / (bins_theory)
    chi_sq = np.sum( v[~np.isnan(v) & ~np.isinf(v)] )

    P_val = 1.0 - ss.chi2.cdf(chi_sq, K-1)
    if eps is None: return P_val
    return P_val > eps


class BernoulliGen(RandGenerator):

    def __init__(self, p=0.5):
        self.p = p
        pass

    def __next__(self):
        return 0 if np.random.uniform()>self.p else 1


    def test(self, data, eps=None):
        """Performs chi-squared test on data."""

        A = np.array(data)
        N = len(A)

        #
        bins_theory = np.zeros(shape=(2,)) #len = K
        bins_theory[0] = N * (1-self.p)
        bins_theory[1] = N * self.p

        bins_actual = np.zeros(shape=(2,)) #len = K
        bins_actual[0] = np.sum(A==0)
        bins_actual[1] = np.sum(A==1)

        return chisq_test(bins_theory, bins_actual, eps=eps)


    pass

class BinomialGen(RandGenerator):

    def __init__(self, n=1, p=0.5):
        self.n = n
        #self.p = p #actually now a property
        self._bernoulli = BernoulliGen(p=p)
        pass

    @property
    def p(self): return self._bernoulli.p

    def __next__(self):
        return sum(self._bernoulli.gen(self.n))
    
    def test(self, data, eps=None):
        """Performs chi-squared test on data."""

        A = np.array(data)
        K = np.max(A) + 1
        N = len(data)

        bins_actual = np.array([np.sum(A==k) for k in range(K)])
        bins_theory = np.array([N*ss.binom.pmf(k, n=self.n, p=self.p) for k in range(K)])

        return chisq_test(bins_theory, bins_actual, eps=eps)


    pass

class GeometricGen(RandGenerator):

    def __init__(self, p=0.5):
        self.p = p 
        # #self._bernoulli = BernoulliGen(p=p)
        pass

    #@property
    #def p(self): return self._bernoulli.p

    def __next__(self):
        if self.p==1.0: return 1
        
        a = np.random.uniform()
        return int(math.ceil(math.log(a, 1.0-self.p)))
        #return np.random.geometric(self.p)

    def test(self, data, eps=None):
        """Performs chi-squared test on data."""

        A = np.array(data)
        K = np.max(A) + 1
        N = len(data)

        bins_actual = np.array([np.sum(A==k) for k in range(K)])
        bins_theory = np.array([N*ss.geom.pmf(k, p=self.p) for k in range(K)])

        # Note: arbitrarily stacking first for better testing
        K_max = 11
        bins_actual = bins_actual[1:K_max]
        bins_theory = bins_theory[1:K_max]

        return chisq_test(bins_theory, bins_actual, eps=eps)

    pass

"""
# Not implemented!

class NegBinomialGen(RandGenerator):

    pass

class HypergeometricGen(RandGenerator):

    pass
"""

class PoissonGen(RandGenerator):

    def __init__(self, lamb=2):
        self.lamb = lamb
        pass

    def __next__(self):
        lim = math.exp(-self.lamb)
        n = 0
        prod = np.random.uniform()
        while prod>lim:
            prod *= np.random.uniform()
            n += 1
        return n

    def test(self, data, eps=None):
        """Performs chi-squared test on data."""

        A = np.array(data)
        K = np.max(A) + 1
        N = len(data)

        bins_actual = np.array([np.sum(A==k) for k in range(K)])
        bins_theory = np.array([N*ss.poisson.pmf(k, mu=self.lamb) for k in range(K)])

        return chisq_test(bins_theory, bins_actual, eps=eps)

    pass

class DiscreteUniformGen(RandGenerator):

    def __init__(self, vals=range(2)):
        self.vals = vals

        pass

    def __next__(self):
        a = np.random.uniform()
        n = len(self.vals)
        return self.vals[int(math.floor(n*a))]

    def test(self, data, eps=None):
        """Performs chi-squared test on data."""

        A = np.array(data)
        bins_actual = np.array([np.sum(A==v) for v in self.vals])
        bins_theory = np.full(bins_actual.shape, float(len(A))/len(bins_actual))
        return chisq_test(bins_theory, bins_actual, eps)

    pass





def gen_hist(Gen, n=1000, eps=None):
    print("Generating %s of %s" % (n, Gen.__class__.__name__))
    res = Gen.gen(n)
    tst = Gen.test(res, eps=eps)
    print("Test result: %s" % (tst,))
    plt.hist(res)
    plt.show()
    return res


if __name__ == "__main__":
    # """
    a1 = gen_hist(BernoulliGen(0.8))
    a2 = gen_hist(BinomialGen(7, 0.8))
    a3 = gen_hist(GeometricGen(0.8))
    a4 = gen_hist(PoissonGen(0.8))
    a5 = gen_hist(DiscreteUniformGen([x for x in range(8)]))
    # """
    pass
else:
    pass


