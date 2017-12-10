#!/usr/bin/env python
"""
Author: Anatoly Makarevich
Laboratory work #3

5 continuous distributions

Kolmogorov test for each
(using numeric integration, e.g. middle rectangles,
to find CDF given a PDF)

"""

from Lab1 import RandGenerator
from Lab2 import gen_hist

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as ss
import scipy.special


# 1. Continuous uniform
# 2. Normal (univariate)
# 3. Log-normal
# 4. Exponential
# 5. Weibul-Gnedenko
# 6. Laplace
# 7. Gamma
# 8. Beta
# 9. Cauchy
# 10. Logistic
# 11. Chi-squared
# 12. Student
# 13. Fisher

def kolmogorov_smirnov_test(data, cdf, eps=None):
    # TODO: Implement
    P_val = 0

    srt = np.sort(data)
    n = len(srt)
    step = float(1.0/n)
    max_delta = 0.0

    for i in range(n):
        # EDF
        cx = cdf(srt[i])

        # lower of current 'step'
        d = abs(cx - i*step)
        if d>max_delta: max_delta = d

        # upper of current 'step'
        d = abs(cx - (i+1)*step)
        if d>max_delta: max_delta = d
        pass

    P_val = scipy.special.kolmogorov(max_delta)

    if eps is None: return P_val
    return P_val > eps

#def get_cdf(pdf):
#





class ContinuousUniform(RandGenerator):

    def __init__(self, lo=0, hi=1):
        self.lo = lo
        self.hi = hi
        pass

    def __next__(self):
        return np.random.uniform(self.lo, self.hi)

    def gen(self, n):
        return np.random.uniform(self.lo, self.hi, n)

    def test(self, data, eps=None):
        def cdf(x):
           if x<self.lo: return 0
           if x>self.hi: return 1
           return (x-self.lo)/(self.hi-self.lo)
        return kolmogorov_smirnov_test(data, cdf, eps=eps)

    pass


class NormalUnivariate(RandGenerator):

    # ЦПТ
    # box-mueller

    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
        pass

    def __next__(self):
        # Box-Muller transform
        t_u1 = math.sqrt(-2 * math.log( np.random.uniform() ))        
        t_u2 = 2 * math.pi * np.random.uniform()
        z1 = t_u1*math.cos(t_u2)
        #z2 = t_u1*math.sin(t_u2)
        return z1


    def test(self, data, eps=None):
        cdf = lambda x: ss.norm.cdf(x, loc=self.mu, scale=self.sigma)
        return kolmogorov_smirnov_test(data, cdf, eps=eps)


    pass


class LogNormal(RandGenerator):

    def __init__(self, mu=0, sigma=1):
        self.normgen = NormalUnivariate(mu=mu, sigma=sigma)
        pass

    @property
    def mu(self): return self.normgen.mu

    @property
    def sigma(self): return self.normgen.sigma

    def __next__(self):
        return math.exp(next(self.normgen))

    def test(self, data, eps=None):
        cdf = lambda x: ss.lognorm.cdf(x, self.mu, self.sigma)
        return kolmogorov_smirnov_test(data, cdf, eps=eps)

    pass


class ExponentialDist(RandGenerator):

    def __init__(self, lamb=1):
        self.lamb = 1
        pass

    def __next__(self):
        u = np.random.uniform()
        #return ss.expon.ppf(u, scale = 1.0 / self.lamb)
        return -math.log(u)/self.lamb

    def test(self, data, eps=None):
        cdf = lambda x: ss.expon.cdf(x, scale=1.0/self.lamb)
        return kolmogorov_smirnov_test(data, cdf, eps=eps)

    pass


class GammaDist(RandGenerator):

    def __init__(self, a=1):
        self.a = a
        self._normgen = NormalUnivariate()
        pass

    def __next__(self):
        # Marsaglia and Tsang’s Method
        # Link: http://www.hongliangjie.com/2012/12/19/how-to-generate-gamma-random-variables/

        if self.a<1: a = self.a + 1
        else: a = self.a

        cnt = 100
        val = 0
        while cnt>0:
            u = np.random.uniform()
            z = next(self._normgen)

            d = self.a - 1.0/3
            c = 1.0/math.sqrt(9*d)
            if z>-1.0/c:
                V = (1+c*z)**3
                #print("%s < %s" % (math.log(u),(z**2)/2 +d*(1-V+math.log(V))))
                if math.log(u) < (z**2)/2 +d*(1-V+math.log(V)):
                    val = d*V
                    if self.a<1: return val * u**(1/self.a)
                    return val
            cnt -= 1
            pass
        print("FAIL in 100")
        return ss.gamma.ppf(u, self.a) #fail

    def test(self, data, eps=None):
        cdf = lambda x: ss.norm.cdf(x, self.a)
        return kolmogorov_smirnov_test(data, cdf, eps=eps)

    pass


class BetaDist(RandGenerator):
    # x in [0;1]

    def __init__(self, a=1, b=1): #a=b=1 -> uniform lol
        self._x_gen = GammaDist(a)
        self._y_gen = GammaDist(b)
        pass

    @property
    def a(self): return self._x_gen.a

    @property
    def b(self): return self._y_gen.a

    def __next__(self):
        #u = np.random.uniform()
        #return ss.beta.ppf(u, a=self.a, b=self.b)
        x = next(self._x_gen)
        y = next(self._y_gen)
        return x/(x+y)


    def test(self, data, eps=None):
        cdf = lambda x: ss.norm.cdf(x, self.a, self.b)
        return kolmogorov_smirnov_test(data, cdf, eps=eps)

    pass


if __name__ == "__main__":
    a1 = gen_hist(ContinuousUniform(3,5))
    a2 = gen_hist(NormalUnivariate(3,2))
    a3 = gen_hist(LogNormal(3,2))
    a4 = gen_hist(ExponentialDist(2))
    a5 = gen_hist(GammaDist(2))
    a6 = gen_hist(BetaDist(a=1,b=2))
    pass
else:
    pass


