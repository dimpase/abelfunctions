from abelfunctions.riemann_surface import RiemannSurface

from sage.all import QQ, QQbar

import numpy

print '''
===========================
===== Testing Sageify =====
===========================
'''


R = QQ['x,y']
x,y = R.gens()
f = y**3 + 2*x**3*y - x**7
print 'f =', f


print 'Constructing Riemann surface:'
X = RiemannSurface(f)

print '\nDiscriminant points:'
b = X.discriminant_points()
print b


# print '\nDifferentials:'
# omega = X.holomorphic_differentials()
# print omega

# print '\nPeriod Matirx:'
# Omega = X.riemann_matrix()
# print Omega

# print '\nLocalizing a differential:'
# places = X('oo')
# print 'places:', places
# P = places[0]
# print 'place: ', P
# P.puiseux_series.extend(12)
# omegat = omega[0].localize(P)
# print omegat
