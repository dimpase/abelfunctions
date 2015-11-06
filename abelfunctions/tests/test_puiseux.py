import unittest

from abelfunctions.puiseux import (
    almost_monicize,
    newton_data,
    newton_iteration,
    newton_polygon,
    newton_polygon_exceptional,
    puiseux,
    puiseux_rational,
    transform_newton_polynomial,
    PuiseuxXSeries,
)
from .test_abelfunctions import AbelfunctionsTestCase

from sage.calculus.functional import taylor
from sage.calculus.var import var
from sage.rings.arith import xgcd
from sage.rings.laurent_series_ring import LaurentSeriesRing
from sage.rings.rational_field import QQ
from sage.rings.qqbar import QQbar
from sage.rings.infinity import infinity
from sympy import Poly, Point, Segment, Polygon, RootOf, sqrt, S

# every example will be over QQ[x,y]. consider putting in setup?
R = QQ['x,y']
S = QQ['t']
x,y = R.gens()
t = S.gens()


class TestNewtonPolygon(unittest.TestCase):

    def test_segment(self):
        H = y + x
        self.assertEqual(newton_polygon(H),
                         [[(0,1),(1,0)]])

        H = y**2 + x**2
        self.assertEqual(newton_polygon(H),
                         [[(0,2),(2,0)]])

    def test_general_segment(self):
        H = y**2 + x**4
        self.assertEqual(newton_polygon(H),
                         [[(0,2),(2,0)]])

    def test_colinear(self):
        H = x**4 + x**2*y**2 + y**4
        self.assertEqual(newton_polygon(H),
                         [[(0,4),(2,2),(4,0)]])

        H = x**4 + x**2*y**2 + x*y**3 + y**4
        self.assertEqual(newton_polygon(H),
                         [[(0,4),(2,2),(3,1),(4,0)]])

    def test_multiple(self):
        H = 2*x**2 + 3*x*y + 5*y**3
        self.assertEqual(newton_polygon(H),
                         [[(0,2),(1,1)], [(1,1),(3,0)]])

    def test_general_to_colinear(self):
        H = x**5 + x**2*y**2 + y**4
        self.assertEqual(newton_polygon(H),
                         [[(0,4),(2,2),(4,0)]])

        H = x**5 + x**3*y + x**2*y**2 + y**4
        self.assertEqual(newton_polygon(H),
                         [[(0,4),(1,3),(2,2),(4,0)]])

        H = x**5 + x**2*y**2 + x*y**3 + y**4
        self.assertEqual(newton_polygon(H),
                         [[(0,4),(2,2),(3,1),(4,0)]])

    def test_exceptional(self):
        H = x**5 + x**2*y**2 + y**4
        EN = newton_polygon_exceptional(H)
        self.assertEqual(EN, [[(0,0),(4,0)]])

        # issue 111
        H = -x**8 + y**4 + x*y**2
        EN = newton_polygon_exceptional(H)
        self.assertEqual(EN, [[(0,0),(4,0)]])

    def test_issue111(self):
        H = -x**8 + y**4 + x*y**2
        N = newton_polygon(H)

class TestNewtonData(unittest.TestCase):

    def test_segment(self):
        H = 2*x + 3*y
        self.assertEqual(newton_data(H),
                         [(1, 1, 1, 3*x + 2)])

        H = 2*x**2 + 3*y**2
        self.assertEqual(newton_data(H),
                         [(1, 1, 2, 3*x**2 + 2)])

        H = 2*x**2 + 3*y**3
        self.assertEqual(newton_data(H),
                         [(3, 2, 6, 3*x + 2)])

        H = 2*x**2 + 3*y**4
        self.assertEqual(newton_data(H),
                         [(2, 1, 4, 3*x**2 + 2)])

    def test_general_segment(self):
        H = x**2 + y
        self.assertEqual(newton_data(H),
                         [(1, 1, 1, x)])

        H = x**3 + y**2
        self.assertEqual(newton_data(H),
                         [(1, 1, 2, x**2)])

        H = x**5 + y**3
        self.assertEqual(newton_data(H),
                         [(1, 1, 3, x**3)])

    def test_colinear(self):
        H = 2*x**4 + 3*x**2*y**2 + 5*y**4
        self.assertEqual(newton_data(H),
                         [(1, 1, 4, 5*x**4 + 3*x**2 + 2)])

        H = 2*x**4 + 3*x**3*y + 5*x**2*y**2 + 7*y**4
        self.assertEqual(newton_data(H),
                         [(1, 1, 4, 7*x**4 + 5*x**2 + 3*x + 2)])

        H = 2*x**4 + 3*x**2*y**2 + 5*x*y**3 + 7*y**4
        self.assertEqual(newton_data(H),
                         [(1, 1, 4, 7*x**4 + 5*x**3 + 3*x**2 + 2)])

    def test_multiple(self):
        H = 2*x**2 + 3*x*y + 5*y**3
        self.assertEqual(newton_data(H),
                         [(1, 1, 2, 3*x + 2),
                          (2, 1, 3, 5*x + 3)])

    def test_general_to_colinear(self):
        H = 2*x**5 + 3*x**2*y**2 + 5*y**4
        self.assertEqual(newton_data(H),
                         [(1, 1, 4, 5*x**4 + 3*x**2)])

        H = 2*x**5 + 3*x**3*y + 5*x**2*y**2 + 7*y**4
        self.assertEqual(newton_data(H),
                         [(1, 1, 4, 7*x**4 + 5*x**2 + 3*x)])

        H = 2*x**5 + 3*x**3*y + 5*x**2*y**2 + 7*y**6
        self.assertEqual(newton_data(H),
                         [(1, 1, 4, 5*x**2 + 3*x),
                          (2, 1, 6, 7*x**2 + 5)])


class TestNewPolynomial(unittest.TestCase):

    def setUp(self):
        self.p = R.random_element(degree=5)
        self.q = R.random_element(degree=5)

    def test_null_transform(self):
        H = self.p*self.q
        q,m,l,xi = 1,0,0,0
        Hprime = transform_newton_polynomial(H,q,m,l,xi)
        Htest = H
        self.assertEqual(Hprime, Htest)

    def test_yshift(self):
        H = self.p*self.q
        q,m,l,xi = 1,0,0,1
        Hprime = transform_newton_polynomial(H,q,m,l,xi)
        Htest = H(y=1+y)
        self.assertEqual(Hprime, Htest)

    def test_y(self):
        H = y**2
        q,m,l,xi = 0,1,0,1
        Hprime = transform_newton_polynomial(H,q,m,l,xi)
        Htest = x**2*(1 + y)**2
        self.assertEqual(Hprime, Htest)

        q,m,l,xi = 0,1,2,1
        Hprime = transform_newton_polynomial(H,q,m,l,xi)
        Htest = (1 + y)**2
        self.assertEqual(Hprime, Htest)


class TestNewtonIteration(unittest.TestCase):

    def test_trivial(self):
        G = y - x
        S = newton_iteration(G,3)
        self.assertEqual(S,x)

        G = y - x**2
        S = newton_iteration(G,3)
        self.assertEqual(S,x**2)

    def test_sqrt(self):
        # recenter sqrt(x) at x+1
        G = (y+1)**2 - (x+1)
        S = newton_iteration(G,9).truncate(x,10) + 1

        z = var('z')
        series = taylor(sqrt(z),z,1,9)
        series = R(series.subs({z:x+1}))
        self.assertEqual(S,series)

    def test_cuberoot(self):
        # recenter cuberoot(x) at x+1
        G = (y+1)**3 - (x+1)
        S = newton_iteration(G,9).truncate(x,10) + 1

        z = var('z')
        series = taylor(z**(QQ(1)/QQ(3)),z,1,9)
        series = R(series.subs({z:x+1}))
        self.assertEqual(S,series)

    def test_geometric(self):
        G = (1-x)*y - 1
        S = newton_iteration(G,9).truncate(x,10)

        z = var('z')
        series = taylor(1/(1-z),z,0,9)
        series = R(series.subs({z:x}))
        self.assertEqual(S,series)

    def test_n(self):
        # test if the solution is indeed given to the desired terms
        G = y - x**2
        S = newton_iteration(G,0)
        self.assertEqual(S,0)

        G = y - x**2
        S = newton_iteration(G,3)
        self.assertEqual(S,x**2)


class TestPuiseuxRational(AbelfunctionsTestCase):
    def is_G_vanishing(self, fmonic):
        # each G should satisfy G(0,0) = 0 and G_y(0,0) = 0
        for G,P,Q in puiseux_rational(fmonic):
            self.assertEqual(G(x=0,y=0),0)
    def test_G_vanishing2(self):
        self.is_G_vanishing(self.f2)
    def test_G_vanishing4(self):
        self.is_G_vanishing(self.f4)
    def test_G_vanishing5(self):
        self.is_G_vanishing(self.f5)
    def test_G_vanishing6(self):
        self.is_G_vanishing(self.f6)
    def test_G_vanishing7(self):
        self.is_G_vanishing(self.f7)
    def test_G_vanishing9(self):
        self.is_G_vanishing(self.f9)


class TestAlmostMonicize(AbelfunctionsTestCase):
    def test_monic(self):
        f = y**2 + x
        g,transform = almost_monicize(f)
        self.assertEqual(f,g)

    def test_partially_monic(self):
        g,transform = almost_monicize(self.f1)
        self.assertEqual(self.f1,g)

        f = (x**2 + 1)*y
        g,transform = almost_monicize(f)
        self.assertEqual(f,g)

    def test_not_monic_simple(self):
        f = x**2*y
        g,transform = almost_monicize(f)
        self.assertEqual(g,y)
        self.assertEqual(transform,x**2)

        f = x**2*y + 1
        g,transform = almost_monicize(f)
        self.assertEqual(g,y + 1)
        self.assertEqual(transform,x**2)

        f = x**2*y + x
        g,transform = almost_monicize(f)
        self.assertEqual(g,y + x)
        self.assertEqual(transform,x**2)

        f = x*y**2 + y + x
        g,transform = almost_monicize(f)
        self.assertEqual(g,y**2 + y + x**2)
        self.assertEqual(transform,x)

        f = x**3*y**2 + y + x
        g,transform = almost_monicize(f)
        self.assertEqual(g,y**2 + y + x**4)
        self.assertEqual(transform,x**3)

    def test_not_monic(self):
        f = x**7*y**3 + 2*y - x**7
        g,transform = almost_monicize(f)

        self.assertEqual(g,y**3 + 2*x*y - x**12)
        self.assertEqual(transform,x**4)

        g,transform = almost_monicize(self.f8)
        self.assertEqual(g,-x**4 + 2*x**2*y**5 + y**6)
        self.assertEqual(transform,x)

    def test_issue70(self):
        f = -x**5 + x*y**4 + y**2
        g, transform = almost_monicize(f)
        self.assertEqual(g,y**4 + y**2*x - x**8)
        self.assertEqual(transform,x)


class TestPuiseux(AbelfunctionsTestCase):
    def setUp(self):
        self.f22 = y**3 - x**5
        self.f23 = (y - 1 - 2*x - x**2)*(y - 1 - 2*x - x**7)
        self.f27 = (y**2 - 2*x**3)*(y**2-2*x**2)*(y**3-2*x)
        super(TestPuiseux,self).setUp()

    def get_PQ(self,f, a=0):
        p = puiseux(f,a)
        if p:
            series = [(P._xpart,P._ypart) for P in p]
        else:
            series = []
        return series

    def test_PQ_f1(self):
        series = self.get_PQ(self.f1)
        x = QQbar['x,y'](self.x)
        y = QQbar['x,y'](self.y)
        self.assertItemsEqual(
            series,
            [(x**2, x**4*(x*(y + 1) + 1))])

    def test_PQ_f2(self):
        series = self.get_PQ(self.f2)
        x = QQbar['x,y'](self.x)
        y = QQbar['x,y'](self.y)
        self.assertItemsEqual(
            series,
            [(x, x**2*y),
             (-x**2/2, -x**3*(y + 1)/2)])

    def test_PQ_f2_oo(self):
        series = self.get_PQ(self.f2, a=infinity)
        x = QQbar['x,y'](self.x)
        y = QQbar['x,y'](self.y)
        self.assertItemsEqual(
            series,
            [(1/x**3, (x**2*y + x**2)/x**9)])

    def test_PQ_f3(self):
        # awaiting RootOf simplification issues
        pass

    def test_PQ_f4(self):
        series = self.get_PQ(self.f4)
        x = QQbar['x,y'](self.x)
        y = QQbar['x,y'](self.y)
        self.assertItemsEqual(
            series,
            [(x, x*(y + 1)),
             (x, x*(y - 1))])

    def test_PQ_f4_oo(self):
        series = self.get_PQ(self.f4, a=infinity)
        x = QQbar['x,y'](self.x)
        y = QQbar['x,y'](self.y)
        self.assertItemsEqual(
            series,
            [(1/(-x**2), (x*y + x)/x**4)])

    def test_PQ_f7(self):
        S = QQ['t']
        t = S.gen()
        r0,r1,r2 = (t**3 - t**2 + 1).roots(QQbar, multiplicities=False)

        series = self.get_PQ(self.f7)
        x = QQbar['x,y'](self.x)
        y = QQbar['x,y'](self.y)
        self.assertItemsEqual(
            series,
            [(x, y + r0),
             (x, y + r1),
             (x, y + r2)])

    def test_PQ_f22(self):
        series = self.get_PQ(self.f22)
        x = QQbar['x,y'](self.x)
        y = QQbar['x,y'](self.y)
        self.assertItemsEqual(
            series,
            [(x**3, x**5*(y + 1))])

    def test_PQ_f22_oo(self):
        series = self.get_PQ(self.f22, a=infinity)
        x = QQbar['x,y'](self.x)
        y = QQbar['x,y'](self.y)
        self.assertItemsEqual(
            series,
            [(1/x**3, (x*y + x)/x**6)])

    def test_PQ_f23(self):
        series = self.get_PQ(self.f23)
        x = QQbar['x,y'](self.x)
        y = QQbar['x,y'](self.y)
        self.assertItemsEqual(
            series,
            [(x, x*(x*y + 2) + 1),
             (x, x*(x*(y + 1) + 2) + 1)])

    def test_PQ_f23_oo(self):
        series = self.get_PQ(self.f23, a=infinity)
        x = QQbar['x,y'](self.x)
        y = QQbar['x,y'](self.y)
        self.assertItemsEqual(
            series,
            [(1/x,y/x**7), (1/x,(1+y)/x**7)])

    def test_PQ_f27(self):
        S = QQ['t']
        t = S.gen()
        sqrt2 = (t**2 - 2).roots(QQbar, multiplicities=False)[0]

        series = self.get_PQ(self.f27)
        x = QQbar['x,y'](self.x)
        y = QQbar['x,y'](self.y)
        self.assertItemsEqual(
            series,
            [(x, x*(y + sqrt2)),
             (x, x*(y - sqrt2)),
             (x**2/2, x**3*(y + 1)/2),
             (x**3/2, x*(y + 1))])

    def test_hyperelliptic_oo(self):
        x = self.x
        y = self.y
        f = y**2 - (x**2 - 9)*(x**2 - 4)*(x**2 - 1)
        series = self.get_PQ(f,a=infinity)
        x = QQbar['x,y'](self.x)
        y = QQbar['x,y'](self.y)
        self.assertItemsEqual(
            series,
            [(1/x, (y+1)/x**3),
             (1/x, (y-1)/x**3)])

# class TestPuiseuxTSeries(unittest.TestCase):
#     def test_instantiation(self):
#         # test that the x- and y-parts are instantiated correctly given
#         # the output of puiseux()
#         pass


class TestPuiseuxXSeries(unittest.TestCase):
    def setUp(self):
        pass

    def test_repr(self):
        L = LaurentSeriesRing(QQ, 't')
        t = L.gen()

        p = PuiseuxXSeries(L(1))
        s = '1'
        self.assertEqual(str(p), s)

        p = PuiseuxXSeries(t)
        s = 'x'
        self.assertEqual(str(p), s)

        p = PuiseuxXSeries(t**2)
        s = 'x^2'
        self.assertEqual(str(p), s)

        p = PuiseuxXSeries(t, e=2)
        s = 'x^(1/2)'
        self.assertEqual(str(p), s)

        p = PuiseuxXSeries(t**2, e=2)
        s = 'x'
        self.assertEqual(str(p), s)

        p = PuiseuxXSeries(t**(-1) + 1, e=2)
        s = 'x^(-1/2) + 1'
        self.assertEqual(str(p), s)

    def test_add(self):
        L = LaurentSeriesRing(QQ, 't')
        t = L.gen()

        p = PuiseuxXSeries(1 + t, e=1)
        q = PuiseuxXSeries(1 + t + t**2, e=1)
        r = PuiseuxXSeries(2 + 2*t + t**2, e=1)
        self.assertEqual(p + q, r)

        p = PuiseuxXSeries(1 + t, e=2)
        q = PuiseuxXSeries(1 + t + t**2, e=2)
        r = PuiseuxXSeries(2 + 2*t + t**2, e=2)
        self.assertEqual(p + q, r)

        p = PuiseuxXSeries(1 + t, e=3)
        q = PuiseuxXSeries(1 + t + t**2, e=3)
        r = PuiseuxXSeries(2 + 2*t + t**2, e=3)
        self.assertEqual(p + q, r)

        # mixed ramification indices
        p = PuiseuxXSeries(t, e=1)
        q = PuiseuxXSeries(t**2, e=1)
        r = PuiseuxXSeries(t**2 + t**4, e=2)
        self.assertEqual(p + q, r)

    def test_sub(self):
        L = LaurentSeriesRing(QQ, 't')
        t = L.gen()

        p = PuiseuxXSeries(1 + t, e=1)
        q = PuiseuxXSeries(1 + t + t**2, e=1)
        r = PuiseuxXSeries(t**2, e=1)
        self.assertEqual(q - p, r)

        p = PuiseuxXSeries(1 + t, e=2)
        q = PuiseuxXSeries(1 + t + t**2, e=2)
        r = PuiseuxXSeries(t**2, e=2)
        self.assertEqual(q - p, r)

        p = PuiseuxXSeries(1 + t, e=3)
        q = PuiseuxXSeries(1 + t + t**2, e=3)
        r = PuiseuxXSeries(t**2, e=3)
        self.assertEqual(q - p, r)

        # mixed ramification indices
        p = PuiseuxXSeries(t, e=1)
        q = PuiseuxXSeries(t**2, e=1)
        r = PuiseuxXSeries(t**2 - t**4, e=2)
        self.assertEqual(p - q, r)

    def test_mul(self):
        L = LaurentSeriesRing(QQ, 't')
        t = L.gen()

        p = PuiseuxXSeries(L(1), e=1)
        q = PuiseuxXSeries(t, e=1)
        r = PuiseuxXSeries(t, e=1)
        self.assertEqual(p*q, r)

        p = PuiseuxXSeries(t, e=1)
        q = PuiseuxXSeries(t, e=1)
        r = PuiseuxXSeries(t**2, e=1)
        self.assertEqual(p*q, r)

        p = PuiseuxXSeries(t, e=2)
        q = PuiseuxXSeries(t, e=2)
        r = PuiseuxXSeries(t**2, e=2)
        self.assertEqual(p*q, r)

        p = PuiseuxXSeries(t, e=1)
        q = PuiseuxXSeries(t, e=2)
        r = PuiseuxXSeries(t**3, e=2)
        self.assertEqual(p*q, r)

        p = PuiseuxXSeries(t, e=3)
        q = PuiseuxXSeries(t, e=2)
        r = PuiseuxXSeries(t**5, e=6)
        self.assertEqual(p*q, r)

        p = PuiseuxXSeries(t**(-1), e=3)
        q = PuiseuxXSeries(t, e=3)
        r = PuiseuxXSeries(L(1), e=3)
        self.assertEqual(p*q, r)

        p = PuiseuxXSeries(t**(-1), e=2)
        q = PuiseuxXSeries(t, e=3)
        r = PuiseuxXSeries(t**(-1), e=6)
        self.assertEqual(p*q, r)

    def test_div(self):
        L = LaurentSeriesRing(QQ, 't')
        t = L.gen()

        p = PuiseuxXSeries(L(1), e=1)
        q = PuiseuxXSeries(t, e=1)
        r = PuiseuxXSeries(t, e=1)
        s = PuiseuxXSeries(t**(-1), e=1)
        self.assertEqual(q/p, r)
        self.assertEqual(p/q, s)

        p = PuiseuxXSeries(t, e=1)
        q = PuiseuxXSeries(t**2, e=1)
        r = PuiseuxXSeries(t, e=1)
        s = PuiseuxXSeries(t**(-1), e=1)
        self.assertEqual(q/p, r)
        self.assertEqual(p/q, s)

        p = PuiseuxXSeries(L(1), e=3)
        q = PuiseuxXSeries(t, e=3)
        r = PuiseuxXSeries(t, e=3)
        s = PuiseuxXSeries(t**(-1), e=3)
        self.assertEqual(q/p, r)
        self.assertEqual(p/q, s)

        p = PuiseuxXSeries(t, e=3)
        q = PuiseuxXSeries(t**2, e=3)
        r = PuiseuxXSeries(t, e=3)
        s = PuiseuxXSeries(t**(-1), e=3)
        self.assertEqual(q/p, r)
        self.assertEqual(p/q, s)

        p = PuiseuxXSeries(t, e=3)
        q = PuiseuxXSeries(t, e=2)
        r = PuiseuxXSeries(t**1, e=6)
        s = PuiseuxXSeries(t**(-1), e=6)
        self.assertEqual(q/p, r)
        self.assertEqual(p/q, s)

    def test_valuation(self):
        L = LaurentSeriesRing(QQ, 't')
        t = L.gen()

        p = PuiseuxXSeries(t + t**3, e=1)
        self.assertEqual(p.valuation(), QQ(1))

        p = PuiseuxXSeries(t + t**3, e=2)
        self.assertEqual(p.valuation(), QQ(1)/2)

    def test_prec(self):
        L = LaurentSeriesRing(QQ, 't')
        t = L.gen()

        p = PuiseuxXSeries(t + t**3, e=1)
        self.assertEqual(p.valuation(), QQ(1))

        p = PuiseuxXSeries(t + t**3, e=2)
        self.assertEqual(p.valuation(), QQ(1)/2)

    def test_list(self):
        L = LaurentSeriesRing(QQ, 't')
        t = L.gen()

        p = PuiseuxXSeries(2*t**(-1) + 3 + 5*t**3 + 7*t**8, e=3)
        exponents = p.exponents()
        coefficients = p.coefficients()
        list = p.list()
        self.assertEqual(exponents, [QQ(-1)/3, 0, 1, QQ(8)/3])
        self.assertEqual(coefficients, [2, 3, 5, 7])
        self.assertEqual(list, [2,3,0,0,5,0,0,0,0,7])

    def test_prec(self):
        pass

    def test_symbolic(self):
        from sage.all import SR
        L = LaurentSeriesRing(SR, 't')
        t = L.gen()
        a = SR('a')
        p = PuiseuxXSeries(t**(-1) + a + 5*t + t**3 + 9*t**5, e=3, a=1)
        self.assertTrue(a in p.list())


class TestPuiseux(AbelfunctionsTestCase):

    def test_f2(self):
        R = QQ['x,y']
        x,y = R.gens()
        f = self.f2

        # x=0: one regular and one ramified place
        p = puiseux(f,0)
        self.assertEqual(len(p), 2)

        p[0].extend(8)
        t = p[0].parent().gen()
        xpart = t
        ypart = QQ(1)/2*t**4 - QQ(1)/16*t**9
        self.assertEqual(p[0].xpart, xpart)
        self.assertEqual(p[0].ypart, ypart)

        p[1].extend(8)
        t = p[1].parent().gen()
        xpart = -QQ(1)/2*t**2
        ypart = -QQ(1)/2*t**3 - QQ(1)/64*t**8
        self.assertEqual(p[1].xpart, xpart)
        self.assertEqual(p[1].ypart, ypart)

    def test_f7(self):
        R = QQ['x,y']
        x,y = R.gens()
        f = self.f7

        # x=1: three regular places
        p = puiseux(f,1)
        self.assertEqual(len(p), self.f7.degree(y))

        p[0].extend(4)
        t = p[0].parent().gen()
        xpart = 1 + t
        ypart = -1 + 3*t**2 + 12*t**3
        self.assertEqual(p[0].xpart, xpart)
        self.assertEqual(p[0].ypart, ypart)

        p[1].extend(4)
        t = p[1].parent().gen()
        xpart = 1 + t
        ypart = -3*t - 3*t**2 - QQ(29)/2*t**3
        self.assertEqual(p[1].xpart, xpart)
        self.assertEqual(p[1].ypart, ypart)

        p[2].extend(4)
        t = p[2].parent().gen()
        xpart = 1 + t
        ypart = 2 + 3*t + QQ(5)/2*t**3
        self.assertEqual(p[2].xpart, xpart)
        self.assertEqual(p[2].ypart, ypart)

    def test_issue111(self):
        # the curve -x^5 + x*z^4 + z^2 comes from recentering the curve in
        # issue71 at the singular point (0,1,0)
        f = -x**5 + x*y**4 + y**2

        # two discriminant places
        p = puiseux(f,0)
        self.assertEqual(len(p), 2)

        p[0].extend(16)
        t = p[0].parent().gen()
        xpart = t**2
        ypart = t**5 - QQ(1)/2*t**17
        self.assertEqual(p[0].xpart, xpart)
        self.assertEqual(p[0].ypart, ypart)

        p[1].extend(16)
        t = p[1].parent().gen()
        xpart = -t**2
        ypart = -1/t - QQ(1)/2*t**11 + QQ(5)/8*t**23
        self.assertEqual(p[1].xpart, xpart)
        self.assertEqual(p[1].ypart, ypart)

    def test_prec(self):
        L = LaurentSeriesRing(QQ, 't')
        t = L.gen()

        p = PuiseuxXSeries(L(1), 0, 1, order=5)
        q = PuiseuxXSeries(t**3, 0, 1, order=5)
        r = PuiseuxXSeries(t**(-1) + t**2, 0, 1, order=5)
        s = PuiseuxXSeries(t**(-2) + t**(-1), 0, 1, order=5)

        self.assertEqual((p*p).prec(), 5)
        self.assertEqual((q*q).prec(), 8)
        self.assertEqual((r*r).prec(), 4)
        self.assertEqual((s*s).prec(), 3)

        # ramified
        p = PuiseuxXSeries(L(1), 0, 2, order=5)
        q = PuiseuxXSeries(t**3, 0, 2, order=5)
        r = PuiseuxXSeries(t**(-1) + t**2, 0, 2, order=5)
        s = PuiseuxXSeries(t**(-2) + t**(-1), 0, 2, order=5)

        self.assertEqual((p*p).prec(), QQ(5)/2)
        self.assertEqual((q*q).prec(), QQ(8)/2)
        self.assertEqual((r*r).prec(), QQ(4)/2)
        self.assertEqual((s*s).prec(), QQ(3)/2)

    def test_prec_bigoh(self):
        from sage.rings.big_oh import O
        L = LaurentSeriesRing(QQ, 't')
        t = L.gen()

        # same as test_prec, but using bigoh notation instead
        p = PuiseuxXSeries(L(1) + O(t**5), 0, 2)
        q = PuiseuxXSeries(t**3 + O(t**5), 0, 2)
        r = PuiseuxXSeries(t**(-1) + t**2 + O(t**5), 0, 2)
        s = PuiseuxXSeries(t**(-2) + t**(-1) + O(t**5), 0, 2)

        self.assertEqual((p*p).prec(), QQ(5)/2)
        self.assertEqual((q*q).prec(), QQ(8)/2)
        self.assertEqual((r*r).prec(), QQ(4)/2)
        self.assertEqual((s*s).prec(), QQ(3)/2)
