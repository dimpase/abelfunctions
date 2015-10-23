r"""Puiseux Series :mod:`abelfunctions.puiseux`
===========================================

Tools for computing Puiseux series. A necessary component for computing
integral bases and with Riemann surfaces.

Classes
-------

.. autosummary::

    PuiseuxTSeries
    PuiseuxXSeries

Functions
---------

.. autosummary::

    puiseux
    newton_iteration
    newton_iteration_step

References
----------

.. [Duval] D. Duval, "Rational puiseux expansions", Compositio
   Mathematica, vol. 70, no. 2, pp. 119-154, 1989.

.. [Poteaux] A. Poteaux, M. Rybowicz, "Towards a Symbolic-Numeric Method
   to Compute Puiseux Series: The Modular Part", preprint

Examples
--------

Contents
--------

"""

import numpy
import sympy

from operator import itemgetter
from sympy import ( degree, Point, Segment, Poly, poly, Rational, Dummy,
                    RootOf, gcd, gcdex, LC, expand, cancel, simplify, ratsimp)

rootofsimp = lambda x: x

from sage.all import I, pi
from sage.functions.log import log, exp
from sage.functions.other import ceil
from sage.rings.arith import gcd, xgcd
from sage.rings.big_oh import O
from sage.rings.infinity import infinity
from sage.rings.integer_ring import ZZ
from sage.rings.laurent_series_ring import LaurentSeriesRing
from sage.rings.laurent_series_ring_element import LaurentSeries
from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
from sage.rings.qqbar import QQbar
from sage.rings.rational_field import QQ
from sage.structure.element import AlgebraElement


def newton_polygon_exceptional(H):
    r"""Computes the exceptional Newton polygon of `H`."""
    R = H.parent()
    x,y = R.gens()
    d = H(x=0).degree(y)
    return [[(0,0),(d,0)]]

def newton_polygon(H, additional_points=[]):
    r"""Computes the Newton polygon of `H`.

    It's assumed that the first generator of `H` here is the "dependent
    variable". For example, if `H = H(x,y)` and we are aiming to compute
    a `y`-covering of the complex `x`-sphere then each monomial of `H`
    is of the form

    .. math::

        a_{ij} x^j y^i.

    Parameters
    ----------
    H : bivariate polynomial

    Returns
    -------
    list
        Returns a list where each element is a list, representing a side
        of the polygon, which in turn contains tuples representing the
        points on the side.

    Note
    ----
    This is written using Sympy's convex hull algorithm for legacy purposes. It
    can certainly be rewritten to use Sage's Polytope but do so *very
    carefully*! There are a number of subtle things going on here due to the
    fact that boundary points are ignored.

    """
    # because of the way sympy.convex_hull computes the convex hull we
    # need to remove all points of the form (0,j) and (i,0) where j > j0
    # and i > i0, the points on the axes closest to the origin
    R = H.parent()
    x,y = R.gens()
    monomials = H.monomials()
    points = map(lambda monom: (monom.degree(y), monom.degree(x)), monomials)
    support = map(Point, points) + additional_points
    i0 = min(P.x for P in support if P.y == 0)
    j0 = min(P.y for P in support if P.x == 0)
    support = filter(lambda P: (P.x <= i0) and (P.y <= j0), support)
    convex_hull = sympy.convex_hull(*support)

    # special treatment when the hull is just a point or a segment
    if isinstance(convex_hull, Point):
        P = (convex_hull.x,convex_hull.y)
        return [[P]]
    elif isinstance(convex_hull, Segment):
        P = convex_hull.p1
        convex_hull = generalized_polygon_side(convex_hull)
        support.remove(P)
        support.append(convex_hull.p1)
        sides = [convex_hull]
    else:
        # recursive call with generalized point if a generalized newton
        # polygon is needed.
        sides = convex_hull.sides
        first_side = generalized_polygon_side(sides[0])
        if first_side != sides[0]:
            P = first_side.p1
            return newton_polygon(H,additional_points=[P])

    # convert the sides to lists of points
    polygon = []
    for side in sides:
        polygon_side = [P for P in support if P in side]
        polygon_side = sorted(map(lambda P: (int(P.x),int(P.y)), polygon_side))
        polygon.append(polygon_side)

        # stop the moment we hit the i-axis. despite the filtration at
        # the start of this function we need this condition to prevent
        # returning to the starting point of the newton polygon.
        #
        # (See test_puiseux.TestNewtonPolygon.test_multiple)
        if side.p2.y == 0: break

    return polygon

def generalized_polygon_side(side):
    r"""Returns the generalization of a side on the Newton polygon.

    A generalized Newton polygon is one where every side has slope no
    less than -1.

    Parameters
    ----------
    side : sympy.Segment

    Returns
    -------
    side
    """
    if side.slope < -1:
        p1,p2 = side.points
        p1y = p2.x + p2.y
        side = Segment((0,p1y),p2)
    return side

def bezout(q,m):
    r"""Returns :math:`u,v` such that :math:`uq+mv=1`.

    Parameters
    ----------
    q,m : integer
        Two coprime integers with :math:`q > 0`.

    Returns
    -------
    tuple of integers

    """
    if q == 1:
        return (1,0)
    g,u,v = xgcd(q,-m)
    return (u,v)

def transform_newton_polynomial(H, q, m, l, xi):
    r"""Recenters a Newton polynomial at a given singular term.

    Given the Puiseux data :math:`x=\mu x^q, y=x^m(\beta+y)` this
    function returns the polynomial

    .. math::

        \tilde{H} = H(\xi^v x^q, x^m(\xi^u+y)) / x^l.

    where :math:`uq+mv=1`.

    Parameters
    ----------
    H : polynomial in `x` and `y`
    q, m, l, xi : constants
        See above for the definitions of these parameters.

    Returns
    -------
    polynomial
    """
    R = H.parent()
    x,y = R.gens()

    u,v = bezout(q,m)
    newx = (xi**v)*(x**q)
    newy = (x**m)*(xi**u + y)
    newH = H.subs({x:newx,y:newy})

    # divide by x**l
    R = newH.parent()
    x,y = R.gens()
    exponents, coefficients = zip(*(newH.dict().items()))
    exponents = map(lambda e: (e[0]-l, e[1]), exponents)
    newH = R(dict(zip(exponents, coefficients)))
    return newH

def newton_data(H, exceptional=False):
    r"""Determines the "newton data" associated with each side of the polygon.

    For each side :math:`\Delta` of the Newton polygon of `H` we
    associate the data :math:`(q,m,l,`phi)` where

    .. math::

        \Delta: qj + mi = l \\
        \phi_{\Delta}(t) = \sum_{(i,j) \in \Delta} a_{ij} t^{(i-i_0)/q}

    Here, :math:`a_ij x^j y_i` is a term in the polynomial :math:`H` and
    :math:`i_0` is the smallest value of :math:`i` belonging to the
    polygon side :math:`\Delta`.

    Parameters
    ----------
    H : sympy.Poly
        Polynomial in `x` and `y`.

    Returns
    -------
    list
        A list of the tuples :math:`(q,m,l,\phi)`.
    """
    R = H.parent()
    x,y = R.gens()

    if exceptional:
        newton = newton_polygon_exceptional(H)
    else:
        newton = newton_polygon(H)

    # special case when the newton polygon is a single point
    if len(newton[0]) == 1:
        return []

    # for each side dtermine the corresponding newton data: side slope
    # information and corresponding side characteristic polynomial, phi
    result = []
    for side in newton:
        i0,j0 = side[0]
        i1,j1 = side[1]
        slope = QQ(j1-j0)/QQ(i1-i0)
        q = slope.denom()
        m = -slope.numer()
        l = min(q*j0 + m*i0, q*j1 + m*i1)
        phi = sum(H.coefficient({y:i,x:j})*x**((i-i0)/q) for i,j in side)
        phi = phi.univariate_polynomial()
        result.append((q,m,l,phi))
    return result


def newton_iteration(G, n):
    r"""Returns a truncated series `y = y(x)` satisfying

    .. math::

        G(x,y(x)) \equiv 0 \bmod{x^r}

    where $r = \ceil{\log_2{n}}$. Based on the algorithm in [XXX].

    Parameters
    ----------
    G, x, y : polynomial
        A polynomial in `x` and `y`.
    n : int
        Requested degree of the series expansion.

    Notes
    -----
    This algorithm returns the series up to order :math:`2^r > n`. Any
    choice of order below :math:`2^r` will return the same series.

    """
    R = G.parent()
    x,y = R.gens()
    if n < 0:
        raise ValueError('Number of terms must be positive. (n=%d'%n)
    elif n == 0:
        return R(0)

    phi = G
    phiprime = phi.derivative(y)
    try:
        pi = R(x).polynomial(x)
        gi = R(0)
        si = R(phiprime(y=gi)).polynomial(x).inverse_mod(pi)
    except NotImplementedError:
        raise ValueError('Newton iteration for computing regular part of '
                         'Puiseux expansion failed. Curve is most likely '
                         'not regular at center.')

    r = ceil(log(n,2))
    for i in range(r):
        gi,si,pi = newton_iteration_step(phi,phiprime,gi,si,pi)
    return R(gi)


def newton_iteration_step(phi, phiprime, g, s, p):
    r"""Perform a single step of the newton iteration algorithm.

    Parameters
    ----------
    phi, phiprime : sympy.Poly
        Equation and its `y`-derivative.
    g, s : sympy.Poly
        Current solution and inverse (conjugate) modulo `p`.
    p : sympy.Poly
        The current modulus. That is, `g` is the Taylor series solution
        to `phi(t,g) = 0` modulo `p`.
    x,y : sympy.Symbol
        Dependent and independent variables, respectively.

    Returns
    -------
    gnext,snext,pnext

    """
    R = phi.parent()
    x,y = R.gens()
    g = R(g).univariate_polynomial()
    s = R(s).univariate_polynomial()
    p = R(p).univariate_polynomial()

    pnext = p**2
    gnext = g - phi(y=g).univariate_polynomial()*s
    gnext = gnext % pnext
    snext = 2*s - phiprime(y=gnext).univariate_polynomial()*s**2
    snext = snext % pnext

    gnext = R(gnext)
    snext = R(snext)
    pnext = R(pnext)
    return gnext,snext,pnext


def puiseux_rational(H, recurse=False):
    r"""Puiseux data for the curve :math:`H` above :math:`(x,y)=(0,0)`.

    Given a polynomial :math:`H = H(x,y)` :func:`puiseux_rational`
    returns the singular parts of all of the Puiseux series centered at
    :math:`x=0, y=0`.

    Parameters
    ----------
    H : polynomial
        A plane curve in `x` and `y`.
    recurse : boolean
        (Default: `True`) A flag used internally to keep track of which
        term in the singular expansion is being computed.

    Returns
    -------
    list of `(G,P,Q)`
        List of tuples where `P` and `Q` are the x- and y-parts of the
        Puiseux series, respectively, and `G` is a polynomial used in
        :func:`newton_iteration` to generate additional terms in the
        y-series.
    """
    R = H.parent()
    x,y = R.gens()

    # when recurse is true, return if the leading order of H(0,y) is y
    if recurse:
        IH = H(x=0).polynomial(y).ord()
        if IH == 1:
            return [(H,x,y)]

    # for each newton polygon side branch out a new puiseux series
    data = newton_data(H, exceptional=(not recurse))
    singular_terms = []
    for q,m,l,phi in data:
        u,v = bezout(q,m)
        for psi,k in phi.squarefree_decomposition():
            roots = psi.roots(ring=QQbar, multiplicities=False)
            map(lambda x: x.exactify(), roots)
            for xi in roots:
                Hprime = transform_newton_polynomial(H, q, m, l, xi)
                next_terms = puiseux_rational(Hprime, recurse=True)
                for (G,P,Q) in next_terms:
                    singular_term = (G, xi**v*P**q, P**m*(xi**u + Q))
                    singular_terms.append(singular_term)

    return singular_terms


def almost_monicize(f):
    r"""Transform `f` to an "almost monic" polynomial.

    Perform a sequence of substitutions of the form

    .. math::

        f \mapsto x^d f(x,y/x)

    such that :math:`l(0) \neq 0` where :math:`l=l(x)` is the leading
    order coefficient of :math:`f`.

    Parameters
    ----------
    f,x,y : sympy.Expr
        An algebraic curve in `x` and `y`.

    Returns
    -------
    g, transform
        A new, almost monic polynomial `g` and a polynomial `transform`
        such that `y -> y/transform`.
    """
    R = f.parent()
    x,y = R.gens()
    transform = R(1)
    monic = False
    while f.polynomial(y).leading_coefficient()(x=0) == 0:
        r = f(y=y/x)
        n = r.numerator()
        d = r.denominator()
        f = R(n)
        transform *= x
    return f, transform


def puiseux(f, alpha, beta=None, order=None, parametric=True):
    r"""Singular parts of the Puiseux series above :math:`x=\alpha`.

    Parameters
    ----------
    f : polynomial
        A plane algebraic curve in `x` and `y`.
    alpha : complex
        The x-point over which to compute the Puiseux series of `f`.
    t : variable
        Variable used in the Puiseux series expansions.
    beta : complex
        (Optional) The y-point at which to compute the Puiseux series.
    order : int
        (Default: `None`) If provided, returns Puiseux series expansions
        up the the specified order.

    Returns
    -------
    list of PuiseuxTSeries

    """
    R = f.parent()
    x,y = R.gens()

    # recenter the curve at x=alpha
    if alpha in [infinity,'oo']:
        alpha = infinity
        d = f.degree(x)
        F = f(x=1/x)*x**d
        n,d = F.numerator(), F.denominator()
        falpha,_ = n.polynomial(x).quo_rem(d.univariate_polynomial())
        falpha = R(falpha)
    else:
        falpha = f(x=x+alpha)

    # determine the points on the curve lying above x=alpha
    g, transform = almost_monicize(falpha)
    galpha = g(x=0).univariate_polynomial()
    betas = galpha.roots(ring=QQbar, multiplicities=False)
    map(lambda x: x.exactify(), betas)

    # filter for requested value of beta. raise error if not found
    if not beta is None:
        betas = [b for b in betas if b == beta]
        if not betas:
            raise ValueError('The point ({0}, {1}) is not on the '
                             'curve {2}.'.format(alpha, beta, f))

    # for each (alpha, beta) determine the corresponding singular parts of the
    # Puiseux series expansions. note that there may be multiple, distinct
    # places above the same point.
    singular_parts = []
    for beta in betas:
        H = g(y=y+beta)
        singular_part_ab = puiseux_rational(H)

        # recenter the result back to (alpha, beta) from (0,0)
        for G,P,Q in singular_part_ab:
            Q += beta
            Q = Q/transform.subs({x:P})
            if alpha == infinity:
                P = 1/P
            else:
                P += alpha

            # append to list of singular data
            singular_parts.append((G,P,Q))

    # # instantiate PuiseuxTSeries from the singular data
    series = [PuiseuxTSeries(f, alpha, singular_data, order=order)
              for singular_data in singular_parts]

    # # return x-series representations if requested
    # if not parametric:
    #     series = [px for P in series for px in P.xseries()]
    return series


class PuiseuxTSeries(object):
    r"""A Puiseux t-series about some place :math:`(\alpha, \beta) \in X`.

    A parametric Puiseux series :math:`P(t)` centered at :math:`(x,y) =
    (\alpha, \beta)` is given in terms of a pair of functions

    .. math::

        x(t) = \alpha + \lambda t^e, \\
        y(t) = \sum_{h=0}^\infty \alpha_h t^{n_h},

    where :math:`x(0) = \alpha, y(0) = \beta`.

    The primary reference for the notation and computational method of
    these Puiseux series is D. Duval.


    Attributes
    ----------
    f, x, y : polynomial
    x0 : complex
        The x-center of the Puiseux series expansion.
    ramification_index : rational
        The ramification index :math:`e`.
    terms : list
        A list of exponent-coefficient pairs representing the y-series.
    order : int
        The order of the Puiseux series expansion.

    Methods
    -------
    xseries
    extend
    eval_x
    eval_y

    """
    @property
    def xdata(self):
        return (self.center, self.xcoefficient, self.ramification_index)
    @xdata.setter
    def xdata(self, value):
        self.center, self.xcoefficient, self.ramification_index = value

    @property
    def is_symbolic(self):
        return self._is_symbolic
    @property
    def is_numerical(self):
        return not self._is_symbolic

    @property
    def terms(self):
        return self.ypart.laurent_polynomial().dict().items()
    @property
    def xdatan(self):
        if self.is_numerical:
            return self.xdata
        else:
            return (numpy.complex(self.center),
                    numpy.complex(self.xcoefficient),
                    numpy.int(self.ramification_index))
    @property
    def order(self):
        return self._singular_order + self._regular_order

    def __init__(self, f, x0, singular_data, order=None):
        r"""Initialize a PuiseuxTSeries using a set of :math:`\pi = \{\tau\}`
        data.

        Parameters
        ----------
        f, x, y : polynomial
            A plane algebraic curve.
        x0 : complex
            The x-center of the Puiseux series expansion.
        singular_data : list
            The output of :func:`singular`.
        t : variable
            The variable in which the Puiseux t series is represented.

        """
        R = f.parent()
        x,y = R.gens()
        extension_polynomial, xpart, ypart = singular_data
        L = LaurentSeriesRing(ypart.base_ring(), 't')
        t = L.gen()

        self.f = f
        self.x = x
        self.y = y
        self.t = t
        self._xpart = xpart
        self._ypart = ypart

        # store x-part attributes. handle the centered at infinity case
        if x0 == infinity:
            x0 = QQ(0)
        self.x0 = x0

        # extract and store information about the x-part of the puiseux series
        xpart = xpart(x=t)
        xpartshift = xpart - x0
        ramification_index, xcoefficient = xpartshift.laurent_polynomial().dict().popitem()
        self.center = x0
        self.xcoefficient = xcoefficient
        self.ramification_index = QQ(ramification_index).numerator()
        self.xpart = xpart

        # extract and sotre information about the y-part of the puiseux series
        self.ypart = ypart(x=t,y=0)
        self._initialize_extension(extension_polynomial, x, y)

        # determine the initial order. See the order property
        val = ypart(x=t,y=1).valuation()
        self._singular_order = 0 if val == infinity else val
        self._regular_order = self._p.degree(x)

        # the curve, x-part, and terms output by puiseux make the puiseux
        # series unique. any mutability only adds terms
        self.__parent = self.ypart.parent()
        self._hash = hash((self.f, self.xpart, self.ypart))

    def parent(self):
        return self.__parent

    def _initialize_extension(self, extension_polynomial, x, y):
        r"""Set up regular part extension machinery.

        RootOfs in expressions are not preserved under this
        transformation. (that is, actual algebraic representations are
        calculated.) each RootOf is temporarily replaced by a dummy
        variable

        Parameters
        ----------
        extension_polynomial, x, y : polynomial

        Returns
        -------
        None : None
            Internally sets hidden regular extension attributes.
        """
        # store attributes
        _phi = extension_polynomial
        _p = x.parent()(x)
        _g = y.parent()(0)
        self._phi = _phi
        self._phiprime = _phi.derivative(y)
        self._p = _p
        self._g = _g

        # compute inverse of phi'(g) modulo x and store
        _p = _p.polynomial(x)
        _s = self._phiprime(y=_g).univariate_polynomial().inverse_mod(_p)
        self._s = rootofsimp(_s)

    def __repr__(self):
        """Print the x- and y-parts of the Puiseux series."""
        s = '('
        s += str(self.xpart)
        s += ', '
        s += str(self.ypart)
        s += ' + O(%s^%s))'%(self.t,self.order)
        return s

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        r"""Check equality.

        A `PuiseuxTSeries` is uniquely identified by the curve it's
        defined on, its center, x-part terms, and the singular terms of
        the y-part.

        Parameters
        ----------
        other : PuiseuxTSeries

        Returns
        -------
        boolean

        """
        if is_instance(other, PuiseuxTSeries):
            if self._hash == other._hash:
                return True
        return False

    def xseries(self, all_conjugates=True):
        r"""Returns the corresponding x-series.

        Parameters
        ----------
        all_conjugates : bool
            (default: True) If ``True``, returns all conjugates
            x-representations of this Puiseux t-series. If ``False``,
            only returns one representative.

        Returns
        -------
        list
            List of PuiseuxXSeries representations of this PuiseuxTSeries.

        """
        L = self.ypart.parent()
        t = R.gen()
        S = L.base_ring()['z']
        z = S.gen()

        # given x = alpha + lambda*t^e solve for t. this involves finding an
        # e-th root of either (1/lambda) or of lambda, depending on e's sign
        e = self.ramification_index
        lamb = S(self.xcoefficient)
        order = self.order
        if e > 0:
            phi = lamb*z**e - 1
        else:
            phi = z**abs(e) - lamb
        mu = phi.roots(QQbar, multiplicities=False)[0]

        if all_conjugates:
            conjugates = [mu*exp(2*pi*I*k/abs(e)) for k in range(abs(e))]
        else:
            conjugates = [mu]
        map(lambda x: x.exactify(), conjugates)

        # determine the resulting x-series
        xseries = []
        for c in conjugates:
            t = self.ypart.parent().gen()
            fconj = self.ypart.subs({t:(c*t)})
            p = PuiseuxXSeries(fconj, self.x0, e)
            xseries.append(p)
        return xseries

    def nterms(self):
        """Returns the number of non-zero computed terms.

        Parameters
        ----------
        None

        Returns
        -------
        int

        """
        return len(self.terms)

    def add_term(self, order=None):
        r"""Extend the y-series terms in-place using Newton iteration.

        The modular Newtion iteration algorithm in
        :func:`newton_iteration` efficiently computes the series up to
        order :math:`t^{2^n}` where :math:`2^n` is the smallest power of
        two greater than the current order.

        """
        g,s,p = newton_iteration_step(
            self._phi, self._phiprime, self._g, self._s, self._p)

        self._g = g
        self._s = s
        self._p = p

        # operation below: yseries = ypart(y=g)(y=0)
        t = self.t
        self.ypart = (self._ypart)(y=g)(x=t)
        self._regular_order = self._p.degree()

    def extend(self, order=None):
        r"""Extends the series in place.

        Computes additional terms in the Puiseux series up to the
        specified `order` or with `nterms` number of non-zero terms. If
        neither `degree` nor `nterms` are provided then the next
        non-zero term will be added to this t-series.

        Parameters
        ----------
        order : int, optional
            The desired degree to extend the series to.

        Returns
        -------
        None

        """
        # if no order is given, extend the series as little as possible
        if not order:
            order = self.order + 1

        # add_term() updates self.order in-place. keep adding terms
        # until we exceed the desired order
        while self.order < order:
            self.add_term()

    def extend_to_t(self, t, curve_tol=1e-8, rel_tol=1e-4):
        r"""Extend the series to accurately determine the y-values at `t`.

        Add terms to the t-series until the the regular place
        :math:`(x(t), y(t))` is within a particular tolerance of the
        curve that the Puiseux series is approximating.

        Parameters
        ----------
        t : complex
        eps : double
        curve_tol : double
            The tolerance for the corresponding point to lie on the curve.
        rel_tol : double
            A relative tolerance parameter used to ensure that the point
            :math:`(x(t_0),y(t_0))` is on the same branch as the center
            of the Puiseux series.

        Returns
        -------
        none
            The PuiseuxTSeries is modified in-place.
        """
        # note that we need to keep track of how much the y-value
        # changes with each iteration just in case it is intersecting
        # with a different branch of the curve
        num_iter = 0
        max_iter = 16
        while num_iter < max_iter:
            xt = self.eval_x(t).n()
            yt = self.eval_y(t).n()
            n,a = max(self.terms)
            a = a.n()

            curve_error = abs(self.f(x=xt,y=yt).n())
            rel_error = abs((a*t**n/yt).n())
            if (curve_error < curve_tol) and (rel_error < rel_tol):
                break
            else:
                self.add_term()
                num_iter += 1

    def extend_to_x(self, x, curve_tol=1e-8, rel_tol=1e-2):
        r"""Extend the series to accurately determine the y-values at `x`.

        Add terms to the t-series until the the regular place :math:`(x,
        y)` is within a particular tolerance of the curve that the
        Puiseux series is approximating.

        Parameters
        ----------
        x : complex
        curve_tol : double
            The tolerance for the corresponding point to lie on the curve.
        rel_tol : double
            A relative tolerance parameter used to ensure that the point
            :math:`(x(t_0),y(t_0))` is on the same branch as the center
            of the Puiseux series.

        Returns
        -------
        none
            The PuiseuxTSeries is modified in-place.
        """
        # simply convert to t and pass to extend. choose any conjugate
        # since the convergence rates between each conjugate is equal
        center, xcoefficient, ramification_index = self.xdata
        t = numpy.power((x-center)/xcoefficient, 1.0/ramification_index)
        self.extend_to_t(t, curve_tol=curve_tol, rel_tol=rel_tol)

    def eval_x(self, t):
        r"""Evaluate the x-part of the Puiseux series at `t`.

        Parameters
        ----------
        t : sympy.Expr or complex

        Returns
        -------
        val = complex

        """
        center, xcoefficient, ramification_index = self.xdata
        val = center + xcoefficient*t**ramification_index
        return val

    def eval_dxdt(self, t):
        r"""Evaluate the derivative of the x-part of the Puiseux series at 't'.

        Parameters
        ----------
        t : complex

        Returns
        -------
        val : complex
        """
        center, xcoefficient, ramification_index = self.xdata
        val = xcoefficient*ramification_index*t**(ramification_index-1)
        return val

    def eval_y(self, t, order=None):
        r"""Evaluate of the y-part of the Puiseux series at `t`.

        The y-part can be evaluated up to a certain order or with a
        certain number of terms.

        Parameters
        ----------
        t : complex or complex
        nterms : int, optional
            If provided, only evaluates using `nterms` in the y-part of
            the series.  If set to zero, will evaluate the principal
            part of the series: the terms in the series which
            distinguishes places with the same x-projection.
        order : int, optional
            If provided, only evaluates up to `order`.

        Returns
        -------
        complex

        Notes
        -----

        This can be sped up using a Holder-like fast exponent evaluation
        trick.

        """
        if order:
            self.extend(order=order)

        # set which terms will be used for evaluation
        if order >= 0:
            terms = [(n,alpha) for n,alpha in self.terms if n < order]
        else:
            terms = self.terms
        return sum(alpha*t**n for n,alpha in terms)


class PuiseuxXSeries(AlgebraElement):
    r"""A Puiseux x-series centered at :math:`x = x_0`.

    A Puiseux series :math:`p(x)` centered at :math:`x = x_0` is a
    series of the form

    .. math::

        p(x) = \sum_{h=0}^\infty \alpha_h (x - x_0)^{n_h/e},

    The primary reference for the notation and computational method of
    these Puiseux series is D. Duval.

    Attributes
    ----------
    valuation
    prec
    center

    Methods
    -------
    eval
    evalf
    exponents
    coefficients
    list
    is_zero

    """
    @property
    def ramification_index(self):
        return self.__e

    @property
    def center(self):
        return self.__a

    @property
    def laurent_part(self):
        return self.__f

    def __init__(self, f, a=0, e=1):
        r"""Initialize a Puiseux series.

        The input :math:`f` is a Laurent series

        .. math::

            f = \sum \alpha_n t^n,

        where :math:`a` and :math:`e` are such that

        .. math::

            t = (x-a)^{1/e}.

        Thus, the Puiseux series

        .. math::

            f = \sum \alpha_n (x - a)^{n/e}

        is constructed.

        Parameters
        ----------
        f : Laurent series
            A laurent series in :math:`t = (x - a)^{1/e}`.
        a : complex
            The center of the Puiseux series.
        e : integer
            The ramification index of the Puiseux series.

        """
        # coerce input to laurent series
        if isinstance(f, PuiseuxXSeries):
            a = f.__a
            e = f.__e
            f = f.__f
        elif not isinstance(f, LaurentSeries):
            raise ValueError('%s must be a Laurent series.'%f)

        # handle negative exponent
        if e < 0:
            t = f.parent().gen()
            e = -e
            f = f(t=t**(-1))

        AlgebraElement.__init__(self, f.parent())
        self._parent = f.parent()
        self.__f = f
        self.__e = QQ(e)
        self.__a = self._parent(a)

    def _repr_(self):
        # TODO: the variable 'x' is hard-coded throughout. this should
        # obviously change once a PuiseuxSeriesRing object is created
        if self.is_zero():
            if self.prec() == infinity:
                return "0"
            else:
                return "O(%s^%s)"%(self._parent.variable_name(),self.prec())
        s = " "
        v = self.__f.list()  # coefficient list
        valuation = self.valuation()
        m = len(v)
        if self.__a:
            X = '(x - %s)'%(self.__a)
        else:
            X = 'x'

        # print each term
        atomic_repr = self._parent.base_ring()._repr_option('element_is_atomic')
        first = True
        for n in xrange(m):
            x = v[n]
            e = QQ(n)/self.__e + valuation
            x = str(x)
            if x != '0':
                if not first:
                    s += " + "
                if ((not atomic_repr) and
                    (x[1:].find("+") != -1 or x[1:].find("-") != -1)):
                    x = "(%s)"%x
                if e == 1:
                    var = "*%s"%X
                elif e == 0:
                    var = ""
                else:
                    if e.denominator() == 1:
                        var = "*%s^%s"%(X,e)
                    else:
                        var = "*%s^(%s)"%(X,e)
                s += "%s%s"%(x,var)
                first = False

        # clean up
        s = s.replace(" + -", " - ")
        s = s.replace(" - -", " + ")
        s = s.replace(" 1*"," ")
        s = s.replace(" -1*", " -")

        # bigoh
        if self.prec() == 0:
            bigoh = "O(1)"
        elif self.prec() == 1:
            #            bigoh = "O(%s)"%self._parent.variable_name()
            bigoh = "O(x)"
        else:
            if self.__a:
                bigoh = "O((x - %s)^%s)"%(self.__a,self.prec())
            else:
                bigoh = "O(x^%s)"%self.prec()
        if self.prec() != infinity:
            if s == " ":
                return bigoh
            s += " + %s"%bigoh
        return s[1:]

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
        m = self.__e
        n = right.__e
        g = gcd(QQ(1)/m,QQ(1)/n).denominator()
        M = QQ(g)/m
        N = QQ(g)/n
        return g, M, N

    def __eq__(self, right):
        f1 = self.__f
        f2 = right.__f

        if self.__a != right.__a:
            return False
        if self.__e != right.__e:
            g, M, N = self._common_ramification_index(right)
            t = self.parent().gen()
            f1 = self.__f(t=t**M)
            f2 = right.__f(t=t**N)
        if f1 != f2:
            return False
        return True

    def _add_(self, right):
        # TODO: make it work for objects other than Puiseux series
        if self.__a != right.__a:
            raise ValueError('Can only add puiseux series with the same center.')

        t = self.parent().gen()

        # find a common ramification index
        g, M, N = self._common_ramification_index(right)
        f1 = self.__f(t=t**M)
        f2 = right.__f(t=t**N)

        # add
        f = self._parent(f1 + f2)
        return PuiseuxXSeries(f, self.__a, g)

    def _sub_(self, right):
        # TODO: make it work for objects other than Puiseux series

        # find common ramification index
        t = self.parent().gen()

        # find a common ramification index
        g, M, N = self._common_ramification_index(right)
        f1 = self.__f(t=t**M)
        f2 = right.__f(t=t**N)
        f = self._parent(f1 - f2)
        return PuiseuxXSeries(f, self.__a, g)

    def _mul_(self, right):
        # TODO make it work for objects other than Puiseux series

        # find common ramification index
        t = self.parent().gen()

        # find a common ramification index
        g, M, N = self._common_ramification_index(right)
        f1 = self.__f(t=t**M)
        f2 = right.__f(t=t**N)
        f = self._parent(f1 * f2)
        return PuiseuxXSeries(f, self.__a, g)

    def _div_(self, right):
        # TODO make it work for objects other than Puiseux series

        # find common ramification index
        t = self.parent().gen()

        # find a common ramification index
        g, M, N = self._common_ramification_index(right)
        f1 = self.__f(t=t**M)
        f2 = right.__f(t=t**N)
        f = self._parent(f1 / f2)
        return PuiseuxXSeries(f, self.__a, g)

    def _neg_(self):
        return PuiseuxXSeries(self.parent(-self.__f), self.__a, self.__e)

    def eval(self, x):
        xe = x**self.__e
        return self.__f(xe)

    def evalf(self, x, **kwds):
        return self.eval(x).n(**kwds)

    def ramification_index(self):
        return self.__e

    def valuation(self):
        return self.__f.valuation()/self.__e

    def prec(self):
        return self.__f.prec()/self.__e

    def coefficients(self):
        return self.__f.coefficients()

    def exponents(self):
        return map(lambda e: e/self.__e, self.__f.exponents())

    def is_zero(self):
        return self.__f.is_zero()

    def variable(self):
        return self.__f.variable()

    def list(self):
        return self.__f.list()

    def exponents(self):
        return map(lambda e: QQ(e)/self.__e, self.__f.exponents())


# if __name__=='__main__':
#     R = QQ['x,y']
#     x,y = R.gens()
#     f1 = (x**2 - x + 1)*y**2 - 2*x**2*y + x**4
#     f2 = -x**7 + 2*x**3*y + y**3
#     f3 = (y**2-x**2)*(x-1)*(2*x-3) - 4*(x**2+y**2-2*x)**2
#     f4 = y**2 + x**3 - x**2
#     f5 = (x**2 + y**2)**3 + 3*x**2*y - y**3
#     f6 = y**4 - y**2*x + x**2
#     f7 = y**3 - (x**3 + y)**2 + 1
#     f8 = x**2*y**6 + 2*x**3*y**5 - 1
#     f9 = 2*x**7*y + 2*x**7 + y**3 + 3*y**2 + 3*y
#     f10 = (x**3)*y**4 + 4*x**2*y**2 + 2*x**3*y - 1

#     f = f2

#     res = f.resultant(f.derivative(y),y)
#     rts = res.univariate_polynomial().roots(QQbar, multiplicities=False)
#     print 'Discriminant points:'
#     for rt in rts:
#         print rt
#     print rts[0].minpoly()
#     print


#     P = puiseux(f,0)
#     for p in P:
#         p.extend(8)
#         print 'Puiseux:'
#         print 'x-part:', p.xpart
#         print 'y-part:', p.ypart
#         print 'x0:', p.x0
#         Px = p.xseries()
#         for px in Px:
#             print px
#         print
