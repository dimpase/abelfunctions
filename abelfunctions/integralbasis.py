r"""Integral Basis :mod:`abelfunctions.integralbasis`
=================================================

A module for computing integral bases of algebraic function fields of
the form :math:`O(X) = \mathbb{C}[x,y] / (f(x,y))` where :math:`X :
f(x,y) = 0`. The algorithm is based off of the paper "An Algorithm for
Computing an Integral Basis in an Algebraic Function Field" by Mark van
Hoeij [vHoeij]_.

An integral basis for :math:`O(X)` is a set of :math:`\beta_i \in
\mathbb{C}(x,y)` such that

.. math::

    \overline{O(X)} = \beta_1\mathbb{C}[x,y] + \cdots +
    \beta_g\mathbb{C}[x,y].

This data is necessary for computing a basis for the space of
holomorphic differentials :math:`\Omega_X^1` defined on the Riemann
surface :math:`X` which is implemented in ``differentials``.

Functions
---------

.. autosummary::

    integral_basis

References
----------

.. [vHoeij] Mark van Hoeij. "An Algorithm for Computing an Integral
    Basis in an Algebraic Function Field". J. Symbolic
    Computation. (1994) 18, p. 353-363

Examples
--------

We compute an integral basis of the curve :math:`f(x,y) = (x^2 - x +
1)y^2 - 2x^2y + x^4`.

.. code-block:: python

    # from abelfunctions import *
    # import sympy
    # from sympy.abc import x,y
    # f = (x**2 - x + 1)*y**2 - 2*x**2*y + x**4
    # X = RiemannSurface(f,x,y)
    # b = integal_basis(X)
    # sympy.pprint(b, use_unicode=False)

.. code-block:: none

            x*y - y
    [1, y - -------]
                2
               x

Contents
--------

"""

from abelfunctions.puiseux import puiseux, PuiseuxTSeries, PuiseuxXSeries
from abelfunctions.utilities import cached_function

from sage.all import infinity
from sage.rings.rational_field import QQ

def Int(i, px):
    r"""Computes :math:`Int_i = \sum_{k \neq i} v(p_i-p_k)`.

    ``Int`` is used in :func:`compute_expansion_bounds` for determining
    sufficient bounds on Puiseux series expansions.

    Parameters
    ----------
    i : int
        Index of the Puiseux series in the list, `px`, to compute `Int` of.
    p : list, PuiseuxXSeries
        A list of :class:`PuiseuxXSeries`.

    Returns
    -------
    val : rational
       The `Int` of the `i`th element of `px`.

    """
    n = len(px)
    pxi = px[i]
    val = QQ(0)
    for k in xrange(n):
        if k != i:
            val += (pxi-px[k]).valuation()
    return val


def compute_expansion_bounds(px):
    r"""Returns a list of necessary bounds on each Puiseux series in ``px``.

    Computes the expansion bounds :math:`N_1, \ldots, N_n` such that for
    all polynomials :math:`G \in L[x,y]` the truncation :math:`r_i` of
    the Puiseux series :math:`p_i` satisfying :math:`v(r_i - p_i) > N_i`
    satisfies the relation

    .. math::

        \forall M,i, v(G(r_i)) > M

    if and only if

    .. math::

        \forall M,i, v(G(p_i)) > M.

    That is, the truncations :math:`r_i` are sufficiently long so that
    polynomial evaluation of :math:`r_i` and :math:`p_i` has the same
    valuation.

    Parameters
    ----------
    px : list, PuiseuxXSeries

    Returns
    -------
    list, int
        A list of degree bounds for each PuiseuxXSeries in ``px``.

    """
    n = len(px)
    N = []
    max_Int = max([Int(k,px) for k in range(n)])
    for i in xrange(n):
        pairwise_diffs = [(px[k]-px[i]).valuation()
                          for i in range(n) if k != i]
        Ni = max(pairwise_diffs) + max_Int - Int(i,px) + 1
        N.append(Ni)
    return N


def compute_series_truncations(f, alpha):
    r"""Computes Puiseux series at :math:`x=\alpha` with necessary terms.

    The Puiseux series expansions of :math:`f = f(x,y)` centered at
    :math:`\alpha` are computed up to the number of terms needed for the
    integral basis algorithm to be successful. The expansion degree
    bounds are determined by :func:`compute_expansion_bounds`.

    Parameters
    ----------
    f : polynomial
    alpha : complex

    Returns
    -------
    list : PuiseuxXSeries
        A list of Puiseux series expansions centered at :math:`x = \alpha` with
        enough terms to compute integral bases as SymPy expressions.

    """
    # compute the parametric Puiseix series with the minimal number of terms
    # needed to distinguish them.
    pt = puiseux(f,alpha)
    px = [p for P in pt for p in P.xseries()]

    # compute the orders necessary for the integral basis algorithm. the orders
    # are on the Puiseux x-series (non-parametric) so scale by the ramification
    # index of each series
    N = compute_expansion_bounds(px)
    for i in range(len(N)):
        e = px[i].ramification_index
        N[i] = ceiling(N[i]*e)

    order = max(N) + 1
    for pti in pt:
        pti.extend(order=order)

    # recompute the corresponding x-series with the extened terms
    px = [p for P in pt for p in P.xseries()]
    return px


def integral_basis(f):
    r"""Returns the integral basis of the algebraic function field of `f`.

    An integral basis for the algebraic function field :math:`O(X)` is a
    set of :math:`\beta_i \in \mathbb{C}(x,y)` such that

    .. math::

        \overline{O(X)} = \beta_1 \mathbb{C}[x,y] + \cdots + \beta_g
        \mathbb{C}[x,y].

    Parameters
    ----------
    f : sympy.Expr
    x : sympy.Symbol
    y : sympy.Symbol

    Returns
    -------
    list, sympy.Expr
        A list of rational functions representing an integral basis.

    """
    R = f.parent()
    x,y = R.gens()

    # The base algorithm assumes f is monic. If this is not the case then
    # monicize by applying the map `y -> y/lc(x), f -> lc^(d-1) f` where lc(x)
    # is the leading coefficient of f.
    d  = f.degree(y)
    lc = f.polynomial(y).leading_coefficient()
    if lc.degree() > 0:
        fmonic = R(f(x,y/lc)*lc**(d-1))
    else:
        fmonic = f/lc
        lc = 1

    # compute the integral basis for the monicized curve
    b = _integral_basis_monic(fmonic)

    # reverse leading coefficient scaling
    for i in xrange(1,len(b)):
        b[i] = b[i](x,lc*y)
    return b


def _integral_basis_monic(f):
    r"""Returns the integral basis of a monic curve.

    Called by :func:`integral_basis` after monicizing its input curve.

    Parameters
    ----------
    f : polynomial

    Returns
    -------
    list : rational functions
        A list of rational functions representing an integral basis of the
        monic curve.

    See Also
    --------
    integral_basis : generic integral basis function

    """
    R = f.parent()
    x,y = R.gens()

    # compute df: the set of monic, irreducible polynomials k such that k**2
    # divides the resultant
    n = f.degree(y)
    res = f.resultant(f.derivative(y),y).univariate_polynomial()
    factor = res.factor()
    df = [k.as_expr() for k,deg in factor
          if (deg > 1) and (k.leading_coefficient() == 1)]

    # compute the Puiseux series expansions at the roots of each element of
    # `df`. Extend these series to the necessary number of terms.
    r = []
    alpha = []
    for k in df:
        roots = k.roots(QQbar, multiplicities=False)
        alpha.extend(roots)

    for alphak in alpha:
        rk = compute_series_truncations(f,alphak)
        r.append(rk)

    # main loop
    b = [R(1)]
    for d in range(1,n):
        bd = compute_bd(f,b,df,r,alpha)
        b.append(bd)
    return b


def compute_bd(f, b, df, r, alpha):
    """Determine the next integral basis element from those already computed.

    Parameters
    ----------
    f : sympy.Expr
    x : sympy.Symbol
    y : sympy.Symbol
    b : list
        The current set of integral basis elements.
    df : list
        The set of irreducible factors.
    r : list of PuiseuxTSeries
        A list of lists of truncated Puiseux series centered each of the
        x-values in the list `alpha`.
    alpha : list of complex
        The roots of each irreducible factor in `k`.
    a : list of sympy.Symbols


    Returns
    -------
    sympy.Expression
        The next integral basis element.

    """
    R = f.parent()
    x,y = R.gens()
    d = len(b)
    a = var(''.join('a%d '%n for n in range(d)))
    b = tuple(b)  # need to make hashable for caching
    bd = y*b[-1]  # guess for next integral basis element

    # loop over each k-factor and, therefore, each puiseux series centered at
    # the root of k.
    for l in range(len(df)):
        k = df[l]
        alphak = alpha[l]
        rk = r[l]
        sufficiently_singular = False
        while not sufficiently_singular:
            # for each puiseux rki series at alphak determine the negative
            # power coefficients of A and add these coeffs to the set of
            # equations we wish to solve for the a0,...,a(d-1)
            equations = []
            for rki in rk:
                A = evaluate_A(a,b,rki)
                A = A + evaluate_integral_basis_element(bd,rki)

                terms = [coeff for exp,coeff in A.terms if exp < 1]
                equations.extend(terms)

            # build a system of equations for the undetermined coefficients
            # from the singular part of the expansion. if a solution exists
            # then the integral basis element is not singular enough at alphak
            sols = solve_coefficient_system(equations, a[:d])
            if sols:
                # build next guess for current basis element
                bdm1 = sum(sols[i]*b[i] for i in range(d))
                bd = (bdm1 + bd)/k
            else:
                # no solution was found. the integral basis element is
                # sufficiently singular at this alphak
                sufficiently_singular = True
    return bd


def solve_coefficient_system(equations, vars, **kwds):
    r"""Solve the linear system of `equations` with resp. to `vars`.

    The systems of equations considered in this problem is always
    linear. This function constructs the linear system and solves
    it. The format is simliar to `sympy.solve`.

    Parameters
    ----------
    equations : list
        A system of equations in `vars`.
    vars : list
        A list of variables to solve for in `equations`.

    Returns
    -------
    list or `None`
        If a unique solution exists, returns as a list. Otherwise,
        returns `None`.

    """
    # form augmented matrix. note that we negate the RHS entries because
    # they originally appear in the LHS equations, themselves
    polys, opt = sympy.parallel_poly_from_expr(equations,vars)
    M = [[p.coeff_monomial(ai) for ai in vars] for p in polys]
    M = sympy.Matrix(M)
    b = [[-p.coeff_monomial(1)] for p in polys]
    b = sympy.Matrix(b)
    system = M.row_join(b)

    # solve the augmented system
    sols = sympy.solve_linear_system(system, *vars, **kwds)

    # the only case when we have a valid "solution" is in the finite
    # case. an infinite family of solutions doesn't count.
    if sols:
        if len(sols.keys()) < len(vars):
            sols = None
        else:
            sols = [sols[ai] for ai in vars]
    else:
        sols = None
    return sols


@cached_function
def evaluate_A(a, b, rki):
    r"""Evaluate the expression:

    .. math::

        A_i := a_1 b_1(x,r_{ki}) + \cdots + a_n b_n(x,r_{ki})

    An intermediate computation in the evaluation of the integral
    basis. Cached for performance purposes.

    Parameters
    ----------
    a : list of sympy.Symbol
    b : sympy.Expr
        An integral basis element.
    rki : PuiseuxXSeries
        The Puiseux series at which to evaluate the "A" expression.

    Returns
    -------
    PuiseuxXSeries
        `A` evaluated at `rki` as a Puiseux series.
    """
    d = len(b)
    f = rki.f
    x = rki.x
    y = rki.y
    alphak = rki.x0
    zero = {sympy.S(0):0}
    order = rki.order
    A = PuiseuxXSeries(f,x,y,alphak,zero,order=order)
    for ai,bi in zip(a[:d],b):
        term = evaluate_integral_basis_element(bi,rki)
        term = term * ai
        A = A + term
    return A

@cached_function
def evaluate_integral_basis_element(b,rki):
    r"""Evaluates the integral basis element ``b`` at ``rki``.

    Cached for performance considerations.

    Parameters
    ----------
    b : sympy.Expr
        An integral basis element: a function which is polynomial in
        `y` but rational in `x`.
    x : sympy.Symbol
    y : sympy.Symbol
    rki : sympy.Expression
        A Puiseux series in `x`.
    f : sympy.Expr
    alphak : complex
        The center of the Puiseux series expansion.

    Returns
    -------
    PuiseuxXSeries

    Notes
    -----
    Actually, there is some discussion to be had about the performance
    benefits of caching. See the discussion in Issue #45 of
    http://github.com/cswiercz/abelfunctions.

    """
    L = rki.parent()
    t = L.gen()

    alphak = rki.center()
    order = rki.order()
    ramification_index = rki.ramification_index()
    val = PuiseuxXSeries(L(0), alphak, ramification_index, order)

    # extract the coefficients and exponents of the numerator as a polynomial
    # in y and evaluate as a PuiseuxXSeries
    b_num = b.numerator()
    b_den = b.denominator()

    R = b_num.parent()
    x,y = R.gens()
    b_num = b_num.polynomial(y)
    b_den = b_den.polynomial(y)
    for yexp, ycoeff in zip(b_num.exponents(), b_num.coefficients()):
        # convert each coefficient into a puiseux series centered at alphak.
        ycoeff = ycoeff(x+alphak)
        coeff = L(0)
        for xexp, xcoeff in zip(ycoeff.exponents(), ycoeff.coefficents()):
            coeff += xcoeff * t**(xexp*ramificaiton_index)
        val = val + coeff * rki**yexp

    # now write the denominator as a puiseux series and divide
    val = val/b_den
    return val


def evaluate_polynomial_at_puiseux_series(q, p):
    r"""Evaluates the bivariate polynomial `q` at the Puiseux series `p`.

    Given a polynomial `q = q(x,y)` and a :class:`PuiseuxXSeries` `p = p(x)`
    return the expression

    .. math::

        q(x, p(x))

    as a `PuiseuxXSeries`.

    Parameters
    ----------
    q : bivariate polynomial
    p : PuiseuxXSeries

    Returns
    -------
    val : PuiseuxXSeries

    Notes
    -----
    This function should become unnecessary once a proper PuiseuxSeriesRing is
    constructed. I just don't have the time right now to learn about coercion
    models and am instead wasting time on writing this function.

    """
    L = p.parent()
    t = L.gen()
    alphak = p.center
    order = p.order
    ramification_index = p.ramification_index
    val = PuiseuxXSeries(L(0), alphak, ramification_index, order)

    # the strategy is to treat q as a polynomial in y with coefficients in x
    R = q.parent()
    x,y = R.gens()
    q = q.polynomial(y)
    for yexp, ycoeff in zip(q.exponents(), q.coefficients()):
        # convert each coefficient into a puiseux series centered at alphak.
        ycoeff = ycoeff(x+alphak).univariate_polynomial()
        coeff = L(0)

        # t = (x - alpha)**(1/e) so (x-alpha) = t**e
        for xexp, xcoeff in zip(ycoeff.exponents(), ycoeff.coefficients()):
            coeff += xcoeff*t**(ramification_index*xexp)

        coeff = PuiseuxXSeries(coeff, alphak, ramification_index, order)
        val = val + coeff*p**yexp

    return val
