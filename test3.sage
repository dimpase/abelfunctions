from abelfunctions.puiseux_series_ring import PuiseuxSeriesRing

P.<x> = PuiseuxSeriesRing(QQbar, 'x')
L.<t> = LaurentSeriesRing(QQbar, 't')

p = 2*x**(-2) + 1 + x + 2*x**2 + 5*x**5
l = 2*t**(-2) + 1 + t + 2*t**2 + 5*t**5

print 'Puiseux:'
timeit('p*p')

print 'Laurent:'
timeit('l*l')

