from abelfunctions.polynomials cimport *
from sage.all import QQ, QQbar, CC
import numpy

print '''
===========================
===== Testing Sageify =====
===========================
'''

from abelfunctions.riemann_surface cimport RiemannSurface
from abelfunctions.riemann_surface_path cimport RiemannSurfacePathLine, RiemannSurfacePathArc
from abelfunctions.singularities import singularities


R = QQ['x,y']
x,y = R.gens()
f = y**3 + 2*x**3*y - x**7
print 'f =', f



# # POLYNOMIAL TESTS
# cdef MultivariatePolynomial p
# cdef UnivariatePolynomial q
# p = MultivariatePolynomial(f)
# cdef complex z1 = 1.0 + 1.0j
# cdef complex z2 = -0.5 + 0.5j
# cdef complex curr = 0
# cdef complex prev = p.eval(z1,z2)
# for i in range(2000):
#     p = MultivariatePolynomial(f)    
#     curr = p.eval(z1,z2)
#     if abs(curr - prev) > 1e-12:
#         print 'FUCK'
#     prev = curr

#     # print 'p(1,2) =', p.eval(1,2)
#     # q = UnivariatePolynomial(f(1,y).univariate_polynomial())
#     # print 'q =', q


# Path Tests
X = RiemannSurface(f)

print 'Constructing path:'
a = X.base_point()
b = numpy.array(X.base_sheets(), dtype=numpy.complex)
for i in range(1):
    print '\n========================='
    print '=== CONSTRUCTING PATH ==='
    print '========================='
    gamma = RiemannSurfacePathArc(X, a, b, 3, 0, numpy.pi, -numpy.pi)
    print 'RESULT:', numpy.array(gamma.get_y(1.0), dtype=numpy.complex)

import matplotlib.pyplot as plt
gamma.plot_x()
plt.savefig('foo.png')
gamma.plot_y(color='green')
plt.savefig('bar.png')




print 'Constructing Riemann surface:'
X = RiemannSurface(f)

print 'Discriminant points:', X.discriminant_points()
print 'Singularities:      ', singularities(f)

print '\nPeriod matrix:'
Omega = X.riemann_matrix()








