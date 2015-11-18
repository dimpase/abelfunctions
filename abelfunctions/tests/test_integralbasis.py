import unittest

from abelfunctions.puiseux import PuiseuxXSeries
from abelfunctions.integralbasis import (
    integral_basis,
    evaluate_polynomial_at_puiseux_series,
)

from sage.all import SR
from sage.rings.big_oh import O
from sage.rings.laurent_series_ring import LaurentSeriesRing
from sage.rings.rational_field import QQ
from sage.rings.qqbar import QQbar


class TestEvaluatePolynomial(unittest.TestCase):

    def test_identity(self):
        R = QQ['x,y']
        x,y = R.gens()
        L = LaurentSeriesRing(QQ, 't')
        t = L.gen()

        p = PuiseuxXSeries(L(1) + O(t**5), 0, 1)
        q = PuiseuxXSeries(t + O(t**5), 0, 1)
        r = PuiseuxXSeries(t + t**3 + O(t**5), 0, 1)
        s = PuiseuxXSeries(t**(-1) + t**2 + O(t**5), 0, 1)

        poly = y
        expr = evaluate_polynomial_at_puiseux_series(poly, p)
        self.assertEqual(expr, p)
        expr = evaluate_polynomial_at_puiseux_series(poly, q)
        self.assertEqual(expr, q)
        expr = evaluate_polynomial_at_puiseux_series(poly, r)
        self.assertEqual(expr, r)
        expr = evaluate_polynomial_at_puiseux_series(poly, s)
        self.assertEqual(expr, s)

    def test_identity_ramified(self):
        R = QQ['x,y']
        x,y = R.gens()
        L = LaurentSeriesRing(QQ, 't')
        t = L.gen()

        p = PuiseuxXSeries(L(1) + O(t**5), 0, 2)
        q = PuiseuxXSeries(t + O(t**5), 0, 2)
        r = PuiseuxXSeries(t + t**3 + O(t**5), 0, 2)
        s = PuiseuxXSeries(t**(-1) + t**2 + O(t**5), 0, 2)

        poly = y
        expr = evaluate_polynomial_at_puiseux_series(poly, p)
        self.assertEqual(expr, p)
        expr = evaluate_polynomial_at_puiseux_series(poly, q)
        self.assertEqual(expr, q)
        expr = evaluate_polynomial_at_puiseux_series(poly, r)
        self.assertEqual(expr, r)
        expr = evaluate_polynomial_at_puiseux_series(poly, s)
        self.assertEqual(expr, s)

    def test_monomial(self):
        R = QQ['x,y']
        x,y = R.gens()
        L = LaurentSeriesRing(QQ, 't')
        t = L.gen()

        # example puiseux series
        p = PuiseuxXSeries(L(1) + O(t**5), 0, 1)
        q = PuiseuxXSeries(t + O(t**5), 0, 1)
        r = PuiseuxXSeries(t + t**3 + O(t**6), 0, 1)
        s = PuiseuxXSeries(t**(-1) + t**2 + O(t**5), 0, 1)

        poly = x*y
        expr = evaluate_polynomial_at_puiseux_series(poly, p)
        p0 = PuiseuxXSeries(t + O(t**5), 0, 1)
        self.assertEqual(expr, p0)
        expr = evaluate_polynomial_at_puiseux_series(poly, q)
        q0 = PuiseuxXSeries(t**2 + O(t**5), 0, 1)
        self.assertEqual(expr, q0)
        expr = evaluate_polynomial_at_puiseux_series(poly, r)
        r0 = PuiseuxXSeries(t**2 + t**4 + O(t**6), 0, 1)
        self.assertEqual(expr, r0)
        expr = evaluate_polynomial_at_puiseux_series(poly, s)
        s0 = PuiseuxXSeries(1 + t**3, 0, 1, order=5)
        self.assertEqual(expr, s0)

        poly = x*y**2
        expr = evaluate_polynomial_at_puiseux_series(poly, p)
        p0 = PuiseuxXSeries(t, 0, 1, order=5)
        self.assertEqual(expr, p0)
        expr = evaluate_polynomial_at_puiseux_series(poly, q)
        q0 = PuiseuxXSeries(t**3, 0, 1, order=5)
        self.assertEqual(expr, q0)
        expr = evaluate_polynomial_at_puiseux_series(poly, r)
        r0 = PuiseuxXSeries(t**3 + 2*t**5, 0, 1, order=6)
        self.assertEqual(expr, r0)
        expr = evaluate_polynomial_at_puiseux_series(poly, s)
        s0 = PuiseuxXSeries(t**(-1) + 2*t**2, 0, 1, order=5)
        self.assertEqual(expr, s0)

        poly = x**2*y
        expr = evaluate_polynomial_at_puiseux_series(poly, p)
        p0 = PuiseuxXSeries(t**2, 0, 1, order=5)
        self.assertEqual(expr, p0)
        expr = evaluate_polynomial_at_puiseux_series(poly, q)
        q0 = PuiseuxXSeries(t**3, 0, 1, order=5)
        self.assertEqual(expr, q0)
        expr = evaluate_polynomial_at_puiseux_series(poly, r)
        r0 = PuiseuxXSeries(t**3 + t**5, 0, 1, order=6)
        self.assertEqual(expr, r0)
        expr = evaluate_polynomial_at_puiseux_series(poly, s)
        s0 = PuiseuxXSeries(t + t**4, 0, 1, order=5)
        self.assertEqual(expr, s0)


    def test_monomial_ramified(self):
        R = QQ['x,y']
        x,y = R.gens()
        L = LaurentSeriesRing(QQ, 't')
        t = L.gen()

        # example puiseux series
        p = PuiseuxXSeries(L(1) + O(t**5), 0, 2)
        q = PuiseuxXSeries(t + O(t**5), 0, 2)
        r = PuiseuxXSeries(t + t**3 + O(t**6), 0, 2)
        s = PuiseuxXSeries(t**(-1) + t**2 + O(t**5), 0, 2)

        poly = x*y
        expr = evaluate_polynomial_at_puiseux_series(poly, p)
        p0 = PuiseuxXSeries(t, 0, 1, order=5)
        self.assertEqual(expr, p0)
        expr = evaluate_polynomial_at_puiseux_series(poly, q)
        q0 = PuiseuxXSeries(t**3, 0, 2, order=5)
        self.assertEqual(expr, q0)
        expr = evaluate_polynomial_at_puiseux_series(poly, r)
        r0 = PuiseuxXSeries(t**3 + t**5, 0, 2, order=6)
        self.assertEqual(expr, r0)

        poly = x*y**2
        expr = evaluate_polynomial_at_puiseux_series(poly, p)
        p0 = PuiseuxXSeries(t**2, 0, 2, order=5)
        self.assertEqual(expr, p0)
        expr = evaluate_polynomial_at_puiseux_series(poly, q)
        q0 = PuiseuxXSeries(t**4, 0, 2, order=8)
        self.assertEqual(expr, q0)
        expr = evaluate_polynomial_at_puiseux_series(poly, r)
        r0 = PuiseuxXSeries(t**4 + 2*t**6, 0, 2, order=6)
        self.assertEqual(expr, r0)

        poly = x**2*y
        expr = evaluate_polynomial_at_puiseux_series(poly, p)
        p0 = PuiseuxXSeries(t**4, 0, 2, order=5)
        self.assertEqual(expr, p0)
        expr = evaluate_polynomial_at_puiseux_series(poly, q)
        q0 = PuiseuxXSeries(t**5, 0, 2, order=7)
        self.assertEqual(expr, q0)
        expr = evaluate_polynomial_at_puiseux_series(poly, r)
        r0 = PuiseuxXSeries(t**5 + t**7, 0, 2, order=8)
        self.assertEqual(expr, r0)

    def test_monomial_ramified_SR(self):
        R = SR['x,y']
        x,y = R.gens()
        L = LaurentSeriesRing(SR, 't')
        t = L.gen()

        # example puiseux series
        p = PuiseuxXSeries(L(1) + O(t**5), 0, 2)
        q = PuiseuxXSeries(t + O(t**5), 0, 2)
        r = PuiseuxXSeries(t + t**3 + O(t**6), 0, 2)
        s = PuiseuxXSeries(t**(-1) + t**2 + O(t**5), 0, 2)

        poly = x*y
        expr = evaluate_polynomial_at_puiseux_series(poly, p)
        p0 = PuiseuxXSeries(t, 0, 1, order=5)
        self.assertEqual(expr, p0)
        expr = evaluate_polynomial_at_puiseux_series(poly, q)
        q0 = PuiseuxXSeries(t**3, 0, 2, order=5)
        self.assertEqual(expr, q0)
        expr = evaluate_polynomial_at_puiseux_series(poly, r)
        r0 = PuiseuxXSeries(t**3 + t**5, 0, 2, order=6)
        self.assertEqual(expr, r0)

        poly = x*y**2
        expr = evaluate_polynomial_at_puiseux_series(poly, p)
        p0 = PuiseuxXSeries(t**2, 0, 2, order=5)
        self.assertEqual(expr, p0)
        expr = evaluate_polynomial_at_puiseux_series(poly, q)
        q0 = PuiseuxXSeries(t**4, 0, 2, order=8)
        self.assertEqual(expr, q0)
        expr = evaluate_polynomial_at_puiseux_series(poly, r)
        r0 = PuiseuxXSeries(t**4 + 2*t**6, 0, 2, order=6)
        self.assertEqual(expr, r0)

        poly = x**2*y
        expr = evaluate_polynomial_at_puiseux_series(poly, p)
        p0 = PuiseuxXSeries(t**4, 0, 2, order=5)
        self.assertEqual(expr, p0)
        expr = evaluate_polynomial_at_puiseux_series(poly, q)
        q0 = PuiseuxXSeries(t**5, 0, 2, order=7)
        self.assertEqual(expr, q0)
        expr = evaluate_polynomial_at_puiseux_series(poly, r)
        r0 = PuiseuxXSeries(t**5 + t**7, 0, 2, order=8)
        self.assertEqual(expr, r0)



# class TestIntegralBasis(AbelfunctionsTestCase):

#     def test_f1(self):
#         a = integral_basis(self.f1,x,y)
#         b = [1, y*(x**2-x+1)/x**2]
#         self.assertEqual(a,b)

#     def test_f2(self):
#         a = integral_basis(self.f2,x,y)
#         b = [1, y/x, y**2/x**3]
#         self.assertEqual(a,b)

#     # def test_f3(self):
#     #     a = integral_basis(self.f3,x,y)
#     #     b = [1, y, (y**2-1)/(x-1), -y*(x - 4*y**2 + 3)/(4*x*(x - 1))]
#     #     self.assertEqual(a,b)

#     def test_f4(self):
#         a = integral_basis(self.f4,x,y)
#         b = [1, y/x]
#         self.assertEqual(a,b)

#     def test_f5(self):
#         a = integral_basis(self.f5,x,y)
#         b = [1, y, y**2, y**3, y*(y**3-1)/x, y**2*(y**3-1)/x**2]
#         self.assertEqual(a,b)

#     def test_f6(self):
#         a = integral_basis(self.f6,x,y)
#         b = [1, y, y**2/x, y**3/x]
#         self.assertEqual(a,b)

#     def test_f7(self):
#         a = integral_basis(self.f7,x,y)
#         b = [1, y, y**2]
#         self.assertEqual(a,b)

#     def test_f8(self):
#         a = integral_basis(self.f8,x,y)
#         b = [1, x*y, x*y**2, y**3*x, x**2*y**4, y**5*x**2]
#         self.assertEqual(a,b)

#     def test_f9(self):
#         a = integral_basis(self.f9,x,y)
#         b = [1, y, y**2]
#         self.assertEqual(a,b)

#     def test_f10(self):
#         a = integral_basis(self.f10,x,y)
#         b = [1, x*y, x**2*y**2, x**3*y**3]
#         self.assertEqual(a,b)

#     def test_rcvexample(self):
#         f = x**2*y**3 - x**4 + 1
#         a = integral_basis(f,x,y)
#         b = [1, x*y, x**2*y**2]
#         self.assertEqual(a,b)

#     def test_issue86(self):
#         f = -x**7 + 2*x*z**5 + z**4
#         a = integral_basis(f,x,z)
#         b = [1, 2*x*z, 2*z*(2*x*z + 1)/x, 4*z**2*(2*x*z + 1)/x**3,
#              8*z**3*(2*x*z + 1)/x**5]
#         self.assertEqual(a,b)
