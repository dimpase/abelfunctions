r"""Puiseux Series Ring Element :mod:`abelfunctions.puiseux_series_ring_element`
============================================================================

Behavior of the elements of a :class:`PuiseuxSeriesRing`. Defines how to
construct an element given a Laurent series, center, and ramficiation
index. Designed to use and be compatible with [Sage's coercion
model](http://doc.sagemath.org/html/en/reference/coercion/index.html).

A Puiseux series is a series of the form .. math::

    p(x) = \sum_{n=N}^\infty a_n (x-a)^{n/e}

where the integer :math:`e` is called the *ramification index* of the series
and the number :math:`a` is the *center*. A Puiseux series is essentially a
Laurent series but with fractional exponents.

Classes
-------

.. autosummary::

    PuiseuxSeries

Functions
---------

.. autosummary::

    is_PuiseuxSeries
    make_element_from_parent

Examples
--------

We begin by constructing the ring of Puiseux series with coefficients in the
rationals.

    sage: from abelfunctions import PuiseuxSeriesRing
    sage: R.<x> = PuiseuxSeriesRing(QQ)

When constructing a Puiseux series the ramification index is automatically
determined from the greatest common divisor of the exponents.

    sage: p = x^(1/2)
    sage: p.ramification_index
    2
    sage: q = x^(1/2) + x*(1/3)
    sage: q.ramficiation_index
    6

Other arithmetic can be performed with Puiseux Series.

    sage: p + q
    x^(1/3) + 2*x^(1/2)
    sage: p - q
    -x^(1/3)
    sage: p * q
    x^(5/6) + x
    sage: (p / q).add_bigoh(4/3)
    x^(1/6) - x^(1/3) + x^(1/2) - x^(2/3) + x^(5/6) - x + x^(7/6) + O(x^(4/3))

Mind the base ring. However, the base ring can be changed.

    sage: I*q
    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
    ...
    TypeError: unsupported operand parent(s) for '*': 'Symbolic Ring' and 'Puiseux Series Ring in x over Rational Field'
    sage: I*q.change_ring(SR)
    I*x^(1/3) + I*x^(1/2)

Other properties of the Puiseux series can be easily obtained.

    sage: r = (3*x^(-1/5) + 7*x^(2/5) + (1/2)*x).add_bigoh(6/5); r
    3*x^(-1/5) + 7*x^(2/5) + 1/2*x + O(x^(6/5))
    sage: r.valuation()
    -1/5
    sage: r.prec()
    6/5
    sage: r.precision_absolute()
    6/5
    sage: r.precision_relative()
    7/5
    sage: r.exponents()
    [-1/5, 2/5, 1]
    sage: r.coefficients()
    [3, 7, 1/2]

Finally, Puiseux series are compatible with other objects in Sage. For example,
you can perform arithmetic with Laurent series.

    sage: L.<x> = LaurentSeriesRing(ZZ)
    sage: l = 3*x^(-2) + x^(-1) + 2 + x**3
    sage: r + l
    3*x^-2 + x^-1 + 3*x^(-1/5) + 2 + 7*x^(2/5) + 1/2*x + O(x^(6/5))


Contents
--------

"""
from sage.rings.arith import gcd
from sage.rings.infinity import infinity
from sage.rings.laurent_series_ring_element import (
    LaurentSeries,
    is_LaurentSeries,
)
from sage.rings.rational_field import QQ
from sage.structure.element import AlgebraElement


def is_PuiseuxSeries(x):
    return isinstance(x, PuiseuxSeries)

def make_element_from_parent(parent, *args):
    return parent(*args)

class PuiseuxSeries(AlgebraElement):
    r"""
    We store a Puiseux series .. math::

        \sum_{n=-N}^\infty a_n x^{n/e}

    as a Laurent series .. math::

        \sum_{n=-N}^\infty a_n t^n

    where `t = x^{1/e}`.
    """
    @property
    def laurent_part(self):
        return self.__l

    @property
    def center(self):
        return self.__a

    @property
    def ramification_index(self):
        return self.__e

    def __init__(self, parent, f, a=0, e=1):
        r"""

        Parameters
        ----------
        parent : Ring
            The target parent.
        f : object
            One of the following types of inputs:

            - `PuiseuxXSeries`
            - `LaurentSeries`

        """
        AlgebraElement.__init__(self, parent)
        self._parent = parent

        if is_PuiseuxSeries(f):
            l = parent.laurent_series_ring()(f.laurent_part)
            a = f.center
            e = f.ramification_index
        else:
            l = parent.laurent_series_ring()(f)

        self.__l = l
        self.__a = a
        self.__e = e

    def __reduce__(self):
        return make_element_from_parent, (self._parent, self.__l, self.__a, self.__e)

    def _repr_(self):
        x = self._parent.variable_name()
        if not self.__a:
            X = x
        else:
            X = '(%s - %s)'%(x,self.__a)

        # extract coefficients and exponets of the laurent part.
        #
        # NOTE: self.__l.coefficients() is bugged when the coefficients are in
        # QQbar but coerced into SR. Therefore, we use self.__l.list() instead
        # (which works) and manually extract the coefficients and exponents
        lst = self.__l.list()
        val = self.valuation()
        coeff = []
        exp = []
        for n in range(len(lst)):
            c = lst[n]
            if not c.is_zero():
                coeff.append(c)
                exp.append(QQ(n)/self.__e + val)

        # print each term
        s = ''
        first = True
        for coeff,exp in zip(coeff,exp):
            # omit ' +' in the first term of the expression
            if first:
                s += str(coeff)
                first = False
            else:
                s += ' + %s'%(coeff)

            # don't print (x-a)^0
            if exp:
                # don't print (x-a)^1
                if exp == 1:
                    s += '*%s'%X
                else:
                    # place parentheses around exponent if rational
                    s += '*%s^'%X
                    if exp.denominator() == 1:
                        s += str(exp)
                    else:
                        s += '(%s)'%exp

        # big oh
        prec = self.prec()
        if prec != infinity:
            prec = QQ(prec)
            if prec == 0:
                bigoh = 'O(1)'
            elif prec == 1:
                bigoh = 'O(%s)'%X
            elif prec.denominator() == 1:
                bigoh = 'O(%s^%s)'%(X,prec)
            else:
                bigoh = 'O(%s^(%s))'%(X,prec)

            if not s:
                return bigoh
            s += ' + %s'%bigoh

        # cleanup
        s = s.replace(' + -', ' - ')
        s = s.replace(' - -', ' + ')
        s = s.replace('1*','')
        s = s.replace('-1*', '-')
        return s

    def __hash__(self):
        return hash(self.__l) ^ self.__e ^ self.__a

    def __call__(self, x):
        r"""Evaluate this Puiseux series."""
        t = (x-self.__a)**(1/self.__e)
        return self.__l(t)

    def _common_ramification_index(self, right):
        r"""Returns a ramification index common to self and right.

        In order to perform arithmetic on Puiseux series it is useful to find a
        common ramification index between two operands. That is, given Puiseux
        series :math:`p` and :math:`q` of ramification indices :math:`e` and
        :math:`f` we write both as series :math:`\tilde{f}` and
        :math:`\tilde{g}` in :math:`(x-a)^(1/g)` such that,

        .. math::

            f = \tilde{f}((x-a)^M), g = \tilde{g}((x-a)^N).

        Parameters
        ----------
        right : PuiseuxXSeries

        Returns
        -------
        g : int
            A ramification index common to self and right.
        M, N : int
            Scaling factors on self and right, respectively.

        """
        m = self.ramification_index
        n = right.ramification_index
        g = gcd(QQ(1)/m,QQ(1)/n).denominator()
        M = g/m
        N = g/n
        return g, M, N

    def _add_(self, right):
        # can only add puiseux series with the same center
        if not self.__a == right.__a:
            raise ValueError('Can only add Puiseux series with same center')

        # # special case when one of the orther is 0
        # if not right:
        #     return self.add_bigoh(right.prec())
        # if not self:
        #     return right.add_bigoh(self.prec())

        # find a common ramification index and transform the two underlying
        # Laurent series
        x = self._parent.laurent_series_ring().gen()
        g, M, N = self._common_ramification_index(right)
        l1 = self.__l(x**M)
        l2 = right.__l(x**N)
        l = l1 + l2
        return PuiseuxSeries(self._parent, l, self.__a, g)

    def _sub_(self, right):
        # can only add puiseux series with the same center
        if not self.__a == right.__a:
            raise ValueError('Can only add Puiseux series with same center')

        # # special case when one of the orther is 0
        # if not right:
        #     return self.add_bigoh(right.prec())
        # if not self:
        #     return right.add_bigoh(self.prec())

        # find a common ramification index
        x = self._parent.laurent_series_ring().gen()
        g, M, N = self._common_ramification_index(right)
        l1 = self.__l(x**M)
        l2 = right.__l(x**N)
        l = l1 - l2
        return PuiseuxSeries(self._parent, l, self.__a, g)

    def _mul_(self, right):
        # find a common ramification index
        x = self._parent.laurent_series_ring().gen()
        g, M, N = self._common_ramification_index(right)
        l1 = self.__l(x**M)
        l2 = right.__l(x**N)
        l = l1 * l2
        return PuiseuxSeries(self._parent, l, self.__a, g)

    def _div_(self, right):
        # find a common ramification index
        x = self._parent.laurent_series_ring().gen()
        g, M, N = self._common_ramification_index(right)
        l1 = self.__l(x**M)
        l2 = right.__l(x**N)
        l = l1 / l2
        return PuiseuxSeries(self._parent, l, self.__a, g)

    def __pow__(self, r):
        r = QQ(r)
        numer = r.numerator()
        denom = r.denominator()

        # if the exponent is integral then do normal exponentiation
        if denom == 1:
            l = self.__l**int(numer)
            e = self.__e

        # otherwise, we only exponentiate by a rational number if there is a
        # single term in the Puiseux series
        #
        # (I suppose we could use Taylor series expansions in the general case)
        else:
            if not self.is_monomial():
                raise ValueError('Can only exponentiate single term by rational')

            x = self._parent.laurent_series_ring().gen()
            l = self.__l(x**numer)
            e = self.__e * denom
        return PuiseuxSeries(self._parent, l, self.__a, e)

    def _cmp_(self, right):
        # scale each laurent series by their ramification indices and compare
        # the laurent series.
        x = self.__l.parent().gen()
        left = self.__l(x**self.__e)
        right = right.laurent_part(x**right.ramification_index)
        return cmp(left, right)

    def __lshift__(self, r):
        return self.shift(r)

    def __rshift__(self, r):
        return self.shift(-r)

    def __nonzero__(self):
        return not not self.__l

    def __hash__(self):
        return hash(self.__l) ^ self.__a ^ self.__e

    def __getitem__(self, r):
        r"""Returns the coefficient with exponent n.

        EXAMPLES::

            sage: R.<x> = PuiseuxSeriesRing(QQ)
            sage: p = x^(-7/2) + 3 + 5*x^(1/2) - 7*x**3
            sage: p[-7/2]
            1
            sage: p[0]
            3
            sage: p[1/2]
            5
            sage: p[3]
            -7
            sage: p[100]
            0
            sage: p[-7/2:1/2]
            x^(-7/2) + 3 + 5*x^(1/2)
        """
        if isinstance(r, slice):
            start, stop, step = r.start, r.stop, r.step
            n = slice(start*self.__e, stop*self.__e, step*self.__e)
            return PuiseuxSeries(self._parent, self.__l[start:stop:step],
                                 self.__a, self.__e)
        else:
            n = int(r*self.__e)
            return self.__l[n]

    def __iter__(self):
        return iter(self.__l)

    def __copy__(self):
        return PuiseuxSeries(self._parent, self.__l.copy(), self.__a, self.__e)

    def valuation(self):
        val = self.__l.valuation() / self.__e
        if val == infinity:
            return 0
        return val

    def add_bigoh(self, prec):
        if prec == infinity or prec >= self.prec():
            return self
        l = self.__l.add_bigoh(prec*self.__e)
        return PuiseuxSeries(self._parent, l, self.__a, self.__e)

    def change_ring(self, R):
        return self._parent.change_ring(R)(self)

    def is_unit(self):
        return self.__l.is_unit() and self.__e == 1

    def is_zero(self):
        return self.__l.is_zero()

    def is_monomial(self):
        return self.__l.is_monomial()

    def __normalize(self):
        r"""Normalize the Puiseux series.

        A Puiseux series `p(x)` is a tuple `(l(x), a, e)` such that .. math::

            p(x) = l((x-a)^{1/e}).

        Normalization makes use that the ramificaiton index is integral.
        """
        if self.is_zero():
            return

        if self.__e.denominator() == 1:
            return

        n = self.__e.numerator()
        d = self.__e.denominator()
        self.__e = d

        x = self.__l.parent().gen()
        self.__l = self.__l(x**n)

    def list(self):
        return self.__l.list()

    def coefficients(self):
        # extract coefficients and exponets of the laurent part.
        #
        # NOTE: self.__l.coefficients() is bugged when the coefficients are in
        # QQbar but coerced into SR. Therefore, we use self.__l.list() instead
        # (which works) and manually extract the coefficients and exponents
        lst = self.__l.list()
        val = self.valuation()
        coeff = []
        for n in range(len(lst)):
            c = lst[n]
            if not c.is_zero():
                coeff.append(c)
        return coeff

    def exponents(self):
        # extract coefficients and exponets of the laurent part.
        #
        # NOTE: self.__l.coefficients() is bugged when the coefficients are in
        # QQbar but coerced into SR. Therefore, we use self.__l.list() instead
        # (which works) and manually extract the coefficients and exponents
        lst = self.__l.list()
        val = self.valuation()
        exp = []
        for n in range(len(lst)):
            c = lst[n]
            if not c.is_zero():
                exp.append(QQ(n)/self.__e + val)
        return exp

    def __setitem__(self, n, value):
        raise IndexError, 'Puiseux series are immutable'

    def degree(self):
        return self.__l.degree()/self.__e

    def shift(self, r):
        r"""Returns this Puiseux series multiplied by `x^r`.

        """
        l = self.__l.shift(r*self.__e)
        return PuiseuxSeries(self._parent, l, self.__a, self.__e)

    def truncate(self, r):
        r"""Returns the Puiseux series of degree ` < r`.

        This is equivalent to self modulo `x^r`.
        """
        l = self.__l.truncate(r*self.__e)
        return PuiseuxSeries(self._parent, l, self.__a, self.__e)

    def prec(self):
        if self.__l.prec() == infinity:
            return infinity
        return self.__l.prec() / self.__e

    def precision_absolute(self):
        return self.prec()

    def precision_relative(self):
        r"""Return the relative precision of the series.

        The relative precision of the Puiseux series is the difference between
        its absolute precision and its valuation.
        """
        if self.is_zero():
            return 0
        return self.prec() - self.valuation()

    def common_prec(self, p):
        r"""returns the minimum precision of `p` and self.
        """

        if self.prec() is infinity:
            return f.prec()
        elif f.prec() is infinity:
            return self.prec()
        return min(self.prec(), f.prec())

    def variable(self):
        return self._parent.variable_name()

    def laurent_series(self):
        r"""If self is a Laurent series, return it as a Laurent series."""
        if self.__e != 1:
            raise ArithmeticError, 'self is not a Laurent series'

        x = self.__l.parent().gen()
        l = self.__l(x-self.__a)
        return l

    def power_series(self):
        try:
            l = self.laurent_series()
            return l.power_series()
        except:
            raise ArithmeticError, 'self is not a power series'

    def inverse(self):
        r"""Returns the inverse of self"""
        return self.__invert__()
