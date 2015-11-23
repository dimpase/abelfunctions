import unittest

from abelfunctions.puiseux_series_ring import PuiseuxSeriesRing
from .test_abelfunctions import AbelfunctionsTestCase

from sage.all import SR
from sage.rings.laurent_series_ring import LaurentSeriesRing
from sage.rings.rational_field import QQ
from sage.rings.qqbar import QQbar
from sage.rings.infinity import infinity

class TestPuiseuxSeriesRing(unittest.TestCase):

    def test_construction_QQ(self):
        R = PuiseuxSeriesRing(QQ)
        x = R.gen()

    def test_construction_SR(self):
        R = PuiseuxSeriesRing(SR)
        x = R.gen()

    def test_construction_QQbar(self):
        R = PuiseuxSeriesRing(QQbar)
        x = R.gen()


class TestPuiseuxSeries(unittest.TestCase):
    def setUp(self):
        pass

    def test_repr(self):
        R = PuiseuxSeriesRing(QQ, 't')
        t = R.gen()

        p = R(1)
        s = '1'
        self.assertEqual(str(p), s)

        p = t
        s = 't'
        self.assertEqual(str(p), s)

        p = t**2
        s = 't^2'
        self.assertEqual(str(p), s)

        p = t**(QQ(1)/2)
        s = 't^(1/2)'
        self.assertEqual(str(p), s)

        p = t**(-QQ(1)/2) + 1
        s = 't^(-1/2) + 1'
        self.assertEqual(str(p), s)

    def test_add(self):
        R = PuiseuxSeriesRing(QQ, 't')
        t = R.gen()

        p = 1 + t
        q = 1 + t + t**2
        r = 2 + 2*t + t**2
        self.assertEqual(p + q, r)

        p = 1 + t**(QQ(1)/2)
        q = 1 + t**(QQ(1)/2) + t
        r = 2 + 2*t**(QQ(1)/2) + t
        self.assertEqual(p + q, r)

    def test_sub(self):
        R = PuiseuxSeriesRing(QQ, 't')
        t = R.gen()

        p = 1 + t
        q = 1 + t + t**2
        r = t**2
        self.assertEqual(q - p, r)

        p = 1 + t**(QQ(1)/2)
        q = 1 + t**(QQ(1)/2) + t
        r = t
        self.assertEqual(q - p, r)

    def test_mul(self):
        L = LaurentSeriesRing(QQ, 't')
        t = L.gen()

        R = PuiseuxSeriesRing(QQ, 't')
        t = R.gen()

        p = 1 + t
        q = 1 + t + t**2
        r = 1 + 2*t + 2*t**2 + t**3
        self.assertEqual(p * q, r)

        p = 1 + t**(QQ(1)/2)
        q = 1 + t**(QQ(1)/2) + t
        r = t
        self.assertEqual(p * q, r)

    # def test_div(self):
    #     L = LaurentSeriesRing(QQ, 't')
    #     t = L.gen()

    #     p = PuiseuxXSeries(L(1), e=1)
    #     q = PuiseuxXSeries(t, e=1)
    #     r = PuiseuxXSeries(t, e=1)
    #     s = PuiseuxXSeries(t**(-1), e=1)
    #     self.assertEqual(q/p, r)
    #     self.assertEqual(p/q, s)

    #     p = PuiseuxXSeries(t, e=1)
    #     q = PuiseuxXSeries(t**2, e=1)
    #     r = PuiseuxXSeries(t, e=1)
    #     s = PuiseuxXSeries(t**(-1), e=1)
    #     self.assertEqual(q/p, r)
    #     self.assertEqual(p/q, s)

    #     p = PuiseuxXSeries(L(1), e=3)
    #     q = PuiseuxXSeries(t, e=3)
    #     r = PuiseuxXSeries(t, e=3)
    #     s = PuiseuxXSeries(t**(-1), e=3)
    #     self.assertEqual(q/p, r)
    #     self.assertEqual(p/q, s)

    #     p = PuiseuxXSeries(t, e=3)
    #     q = PuiseuxXSeries(t**2, e=3)
    #     r = PuiseuxXSeries(t, e=3)
    #     s = PuiseuxXSeries(t**(-1), e=3)
    #     self.assertEqual(q/p, r)
    #     self.assertEqual(p/q, s)

    #     p = PuiseuxXSeries(t, e=3)
    #     q = PuiseuxXSeries(t, e=2)
    #     r = PuiseuxXSeries(t**1, e=6)
    #     s = PuiseuxXSeries(t**(-1), e=6)
    #     self.assertEqual(q/p, r)
    #     self.assertEqual(p/q, s)

    # def test_pow(self):
    #     from sage.rings.big_oh import O
    #     L = LaurentSeriesRing(QQ, 't')
    #     t = L.gen()
    #     n = 2

    #     p = PuiseuxXSeries(1 + t, e=1)
    #     r = PuiseuxXSeries(1 + 2*t + t**2, e=1)
    #     self.assertEqual(p**n, r)

    #     p = PuiseuxXSeries(1 + t + O(t**2), e=1)
    #     r = PuiseuxXSeries(1 + 2*t, e=1)
    #     self.assertEqual(p**n, r)

    # def test_valuation(self):
    #     L = LaurentSeriesRing(QQ, 't')
    #     t = L.gen()

    #     p = PuiseuxXSeries(t + t**3, e=1)
    #     self.assertEqual(p.valuation(), QQ(1))

    #     p = PuiseuxXSeries(t + t**3, e=2)
    #     self.assertEqual(p.valuation(), QQ(1)/2)

    # def test_prec(self):
    #     L = LaurentSeriesRing(QQ, 't')
    #     t = L.gen()

    #     p = PuiseuxXSeries(t + t**3, e=1)
    #     self.assertEqual(p.valuation(), QQ(1))

    #     p = PuiseuxXSeries(t + t**3, e=2)
    #     self.assertEqual(p.valuation(), QQ(1)/2)

    # def test_list(self):
    #     L = LaurentSeriesRing(QQ, 't')
    #     t = L.gen()

    #     p = PuiseuxXSeries(2*t**(-1) + 3 + 5*t**3 + 7*t**8, e=3)
    #     exponents = p.exponents()
    #     coefficients = p.coefficients()
    #     list = p.list()
    #     self.assertEqual(exponents, [QQ(-1)/3, 0, 1, QQ(8)/3])
    #     self.assertEqual(coefficients, [2, 3, 5, 7])
    #     self.assertEqual(list, [2,3,0,0,5,0,0,0,0,7])

    # def test_different_parents(self):
    #     L = LaurentSeriesRing(QQ, 't')
    #     t = L.gen()
    #     M = LaurentSeriesRing(SR, 't')
    #     s = M.gen()

    #     p = PuiseuxXSeries(t, e=1)
    #     q = PuiseuxXSeries(s, e=1)
    #     print p
    #     print q

    #     r = PuiseuxXSeries(2*t, e=1)
    #     print r
    #     print p+q
    #     self.assertEqual(p+q, r)

    #     # r = PuiseuxXSeries(t**2, e=1)
    #     # self.assertEqual(p*q, r)

    #     # r = PuiseuxXSeries(L(1), e=1)
    #     # self.assertEqual(p/q, r)

    # def test_symbolic(self):
    #     from sage.all import SR
    #     L = LaurentSeriesRing(SR, 't')
    #     t = L.gen()
    #     a = SR('a')
    #     p = PuiseuxXSeries(t**(-1) + a + 5*t + t**3 + 9*t**5, e=3, a=1)
    #     self.assertTrue(a in p.list())

    # def test_prec(self):
    #     L = LaurentSeriesRing(QQ, 't')
    #     t = L.gen()

    #     p = PuiseuxXSeries(L(1), 0, 1, order=5)
    #     q = PuiseuxXSeries(t**3, 0, 1, order=5)
    #     r = PuiseuxXSeries(t**(-1) + t**2, 0, 1, order=5)
    #     s = PuiseuxXSeries(t**(-2) + t**(-1), 0, 1, order=5)

    #     self.assertEqual((p*p).prec(), 5)
    #     self.assertEqual((q*q).prec(), 8)
    #     self.assertEqual((r*r).prec(), 4)
    #     self.assertEqual((s*s).prec(), 3)

    #     # ramified
    #     p = PuiseuxXSeries(L(1), 0, 2, order=5)
    #     q = PuiseuxXSeries(t**3, 0, 2, order=5)
    #     r = PuiseuxXSeries(t**(-1) + t**2, 0, 2, order=5)
    #     s = PuiseuxXSeries(t**(-2) + t**(-1), 0, 2, order=5)

    #     self.assertEqual((p*p).prec(), QQ(5)/2)
    #     self.assertEqual((q*q).prec(), QQ(8)/2)
    #     self.assertEqual((r*r).prec(), QQ(4)/2)
    #     self.assertEqual((s*s).prec(), QQ(3)/2)

    # def test_prec_bigoh(self):
    #     from sage.rings.big_oh import O
    #     L = LaurentSeriesRing(QQ, 't')
    #     t = L.gen()

    #     # same as test_prec, but using bigoh notation instead
    #     p = PuiseuxXSeries(L(1) + O(t**5), 0, 2)
    #     q = PuiseuxXSeries(t**3 + O(t**5), 0, 2)
    #     r = PuiseuxXSeries(t**(-1) + t**2 + O(t**5), 0, 2)
    #     s = PuiseuxXSeries(t**(-2) + t**(-1) + O(t**5), 0, 2)

    #     self.assertEqual((p*p).prec(), QQ(5)/2)
    #     self.assertEqual((q*q).prec(), QQ(8)/2)
    #     self.assertEqual((r*r).prec(), QQ(4)/2)
    #     self.assertEqual((s*s).prec(), QQ(3)/2)

if __name__ == '__main__':
    unittest.main()
