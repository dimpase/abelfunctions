import abelfunctions
from abelfunctions import RiemannSurface
from abelfunctions.abelmap import Jacobian, AbelMap

from sage.all import QQ, QQbar, I

import numpy
from numpy.linalg import norm

print '''
===========================
===== Testing Sageify =====
===========================
'''
###############################################################################
from abelfunctions.integralbasis import integral_basis
import cProfile
import pstats

R = QQ['x,y']
x,y = R.gens()
f = -y**8 + x**6 + 2*x**5

print '\nf = %s\n'%f

b = integral_basis(f)
cProfile.run('b = integral_basis(f)', 'test.profile')
p = pstats.Stats('test.profile')
p.strip_dirs().sort_stats('time').print_stats(20)
p.strip_dirs().sort_stats('cumtime').print_stats(20)

print 'b ='
print b

###############################################################################

###############################################################################
# R = QQ['x,y']
# x,y = R.gens()
# f = y**2 - (x-1)*(x+1)*(x-2)*(x+2)*(x-3)*(x+3)*(x-4)*(x+4)
# X = RiemannSurface(f)

# print 'f =', f
# print 'Omega =\n'

# import cProfile
# import pstats
# cProfile.run('Omega = X.riemann_matrix()', 'test.profile')

# print Omega
# print

# p = pstats.Stats('test.profile')
# p.strip_dirs().sort_stats('time').print_stats(20)
###############################################################################

###############################################################################
# R = QQ['x,y']
# x,y = R.gens()
# f11 = x**2*y**3 - x**4 + 1
# X11 = RiemannSurface(f11)

# X11_Jacobian = Jacobian(X11)
# X11_P = X11(0)[0]
# X11_Q = X11(1)[0]
# X11_R = X11(I)[0]
# X11_P0 = X11.base_place()

# J = X11_Jacobian
# P = X11_P
# D = 3*P
# val1 = AbelMap(D)
# val2 = sum(ni*AbelMap(Pi) for (Pi,ni) in D)
# error = norm(J(val1-val2))
# assert error < 1e-7
###############################################################################

# #f = y**3 + 2*x**3*y - x**7
# f = x**2*y**3 - x**4 + 1
# print 'f =', f

# print 'Constructing Riemann surface:'
# X = RiemannSurface(f)

# print '\nDiscriminant points:'
# b = X.discriminant_points()
# print b


# # Riemann Matrix Timings
# import cProfile
# import pstats
# cProfile.run('Omega = X.riemann_matrix()', 'test.profile')
# p = pstats.Stats('test.profile')
# p.strip_dirs().sort_stats('time').print_stats(20)

# print '\nRiemann matrix:'
# print Omega



# # Path timings
# import cProfile
# import pstats
# P = X(-2)[0]
# gamma = X.path(P)
# t = numpy.linspace(0,1,128)
# cProfile.run('[gamma.get_y(ti) for ti in t]', 'test.profile')
# p = pstats.Stats('test.profile')
# p.strip_dirs().sort_stats('time').print_stats(20)


# # print '\nDifferentials:'
# # omega = X.holomorphic_differentials()
# # print omega

# print '\nPeriod Matirx:'
# Omega = X.riemann_matrix()
# print Omega

# # print '\nLocalizing a differential:'
# # places = X('oo')
# # print 'places:', places
# # P = places[0]
# # print 'place: ', P
# # P.puiseux_series.extend(12)
# # omegat = omega[0].localize(P)
# # print omegat
