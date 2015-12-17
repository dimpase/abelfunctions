r"""Differentials :mod:`abelfunctions.differentials`
================================================

This module contains functions for computing a basis of holomorphic
differentials of a Riemann surface given by a complex plane algebraic curve
:math:`f \in \mathbb{C}[x,y]`. A differential :math:`\omega = h(x,y)dx` defined
on a Riemann surface :math:`X` is holomorphic on :math:`X` if it is holomorphic
at every point on :math:`X`.

The function :func:`differentials` computes the basis of holomorphic
differentials from an input algebraic curve :math:`f = f(x,y)`. The
differentials themselves are encapsulated in a :class:`Differential` Cython
class.

Classes
-------

.. autosummary::

    Differential

Functions
---------

.. autosummary::

    differentials
    mnuk_conditions

References
----------

.. [Mnuk] M. Mnuk, "An algebraic approach to computing adjoint curves", Journal
   of Symbolic Computation, vol. 23 (2-3), pp. 229-40, 1997.

Examples
--------

Contents
--------

"""
#from abelfunctions.divisor import Divisor, Place
from abelfunctions.integralbasis import integral_basis
from abelfunctions.puiseux import puiseux
from abelfunctions.singularities import singularities, _transform, genus

from sage.all import solve
from sage.rings.polynomial.all import PolynomialRing
from sage.rings.rational_field import QQ
from sage.rings.qqbar import QQbar

def mnuk_conditions(g, b, generic_adjoint, c):
    """Determine the Mnuk conditions on the coefficients of :math:`P`.

    Determine the conditions on the coefficients `c` of `P` at the integral
    basis element `b` modulo the curve `g = g(u,v)`. See [Mnuk] for details.

    Parameters
    ----------
    g : curve
        An algebraic curve.
    b : integral basis function
        An an element of the basis of the integral closure of the coordinate
        ring of `g`. See :func:`abelfunctions.integralbasis.integral_basis`.
    generic_adjoint : polynomial
        A generic adjoint polynomial as provided by :func:`differentials`. Only
        one instance is created for caching and performance purposes.

    Returns
    -------
    conditions : list
        A list of expressions from which a system of equations is build to
        determine the differentials.

    """
    # extract rings. the generic adjoint should be a member of R[*c][u,v] where
    # *c is a vector of the indeterminants. we will need to convert it to a
    # polynomial in R[u,v,*c] and then back (see below)
    R = g.parent()
    S = generic_adjoint.parent()
    B = S.base_ring()
    c = B.gens()
    T = R.base_ring()[R.variable_names() + B.variable_names()]

    # reduce b*P modulo g. we need to cast b*P and g into the "full" polynomial
    # ring R[u,v,*c]
    #import pdb; pdb.set_trace()
    numer = b.numerator()
    denom = b.denominator()
    u,v = map(T, R.gens())
    expr = numer(u,v) * generic_adjoint(u,v)
    ideal = T.ideal(g(u,v))
    r = expr.reduce(ideal)

    # divide by the largest power of x appearing in the denominator. this is
    # sufficient since we've shifted the curve and its singularity to appear at
    u,v = R.gens()
    mult = R(denom).degree(u)

    # return the coefficients of the terms in the remainder where the u-degree
    # is less than the valuation. (these correspond to terms in the rational
    # expression that lead to negative powers in u, which we want to eliminate)
    u,v = map(S, R.gens())
    args = [u,v] + map(S, c)
    r = r(*args)
    coeffs = r.coefficients()
    monoms = r.monomials()
    conditions = [coeff for coeff,monom in zip(coeffs,monoms)
                  if monom.degree(u) < mult]
    #import pdb; pdb.set_trace()
    return conditions

def recenter_curve(g, singular_point):
    r"""Returns a curve centered at a given singular point.

    Given a singular point :math:`(x : y : z) = (\alpha : \beta : \gamma)`on a
    Riemann surface :func:`recenter_curve` returns an affine curve :math:`h =
    h(u,v)` such that the singularity occurs at :math:`u = 0` where

    * :math:`u,v = x,y` if :math:`\gamma = 1`
    * :math:`u,v = x,z` if :math:`\gamma = 0`

    :func:`recenter_curve` is written in such a way to preserve the base ring
    of the original curve in the case when it's a polynomial ring. For example,
    if :math:`g \in R[c][x,y]` then `h \in R[c][u,v]`.

    See Also
    --------
    abelfunctions.singularities._transform : recenters a given curve at the
    singular point such that the singularity occurs at :math:`u = u0`

    """
    # recenter the curve and adjoint polynomial at the singular point: find
    # the affine plane u,v such that the singularity occurs at u=0
    gsing,u0,v0 = _transform(g,singular_point)
    R = gsing.parent()
    u,v = R.gens()
    h = gsing(u+u0,v)
    return h


def differentials_numerators(f):
    """Return the numerators of a basis of holomorphic differentials on a Riemann
    surface.

    Parameters
    ----------
    f : plane algebraic curve

    Returns
    -------
    differentials : list
        A list of :class:`Differential`s representing *a* basis of Abelian
        differentials of the first kind.

    """
    # homogenize and compute total degree
    R = f.parent().change_ring(QQbar)
    x,y = R.gens()
    d = f.total_degree()

    # construct the generalized adjoint polynomial. we want to think of it as
    # an element of B[*c][x,y] where B is the base ring of f and *c are the
    # indeterminates
    cvars = ['c_%d_%d'%(i,j) for i in range(d-2) for j in range(d-2)]
    vars = cvars + list(R.variable_names())
    C = PolynomialRing(QQbar, cvars)
    S = PolynomialRing(C, [x,y])
    T = PolynomialRing(QQbar, vars)
    c = S.base_ring().gens()
    x,y = S(x),S(y)
    P = sum(c[i+(d-2)*j] * x**i * y**j
            for i in range(d-2) for j in range(d-2)
            if i+j <= d-3)

    # for each singular point [x:y:z] = [alpha:beta:gamma], map f onto the
    # "most convenient and appropriate" affine subspace, (u,v), and center at
    # u=0. determine the conditions on P
    singular_points = singularities(f)
    conditions = []
    for singular_point, _ in singular_points:
        # recenter the curve and adjoint polynomial at the singular point: find
        # the affine plane u,v such that the singularity occurs at u=0
        g = recenter_curve(f, singular_point)

        # the following explicit coercion into the parent ring is necessary to
        # keep Ptilde in the ring R[*c][u,v]. otherwise, it will be coerced
        # into R[*c,u,v]
        #import pdb; pdb.set_trace()
        Ptilde = recenter_curve(P,singular_point)

        # compute the intergral basis at the recentered singular point
        # and determine the Mnuk conditions of the adjoint polynomial
        b = integral_basis(g)
        for bi in b:
            conditions_bi = mnuk_conditions(g,bi,Ptilde,c)
            conditions.extend(conditions_bi)

    # reduce the general adjoint modulo the ideal generated by the integral
    # basis conditions. the coefficients of the remaining c_ij's form the
    # numerators of a basis of abelial differentials of the first kind
    T = R.base_ring()[R.variable_names() + S.base_ring().variable_names()]
    ideal = T.ideal(conditions)
    P_reduced = P(T(x),T(y)).reduce(ideal)

    U = R.base_ring()[R.variable_names()][S.base_ring().variable_names()]
    Pc = U(P_reduced)
    numerators = Pc.coefficients()

    #import pdb; pdb.set_trace()
    return numerators


# def fast_expand(numer,denom,t):
#     r"""Quickly compute the Taylor expansion of `numer/denom`.

#     Parameters
#     ----------
#     numer, denom : sympy.Expr
#         Polynomials in t.
#     t : sympy.Symbol
#         The dependent variable of `numer` and `denom`.
#     order : int
#         The desired order of the expansion.

#     Returns
#     -------
#     sympy.Expr
#     """
#     numer = numer.expand()
#     denom = denom.expand()

#     # if denom is a single term then simply divide through
#     if denom.is_Mul:
#         return (numer/denom).expand()

#     # factor out t. denom should be of the form 't**denom_order(c + O(t))'
#     # where c is nonzero. this is necessary for the fast expansion to work
#     denom_order = denom.leadterm(t)[1]
#     denom = (denom/t**denom_order).expand()

#     # convert numerator and denominator to lists of coefficients.  it's
#     # faster to do it "manually" than to coerce to polynomial
#     order = max(sympy.degree(numer,t), sympy.degree(denom,t)) + 1
#     def get_terms_list(expr):
#         l = [0]*order
#         for term,coeff in expr.collect(t,evaluate=False).items():
#             _,n = term.as_coeff_exponent(t)
#             if n < order:
#                 l[n] = coeff
#         return l

#     q = get_terms_list(numer)
#     r = get_terms_list(denom)

#     # forward solve the coefficient system. note that r[0] (constant coeff of
#     # denom) is nonzero by construction
#     s = [0]*order
#     for n in range(order):
#         known_terms = sum(r[n-k]*s[k] for k in range(n))
#         s[n] = (q[n] - known_terms)/r[0]
#     taylor = sum(s[n]*t**(n-denom_order) for n in range(order))
#     return taylor



# def Differential(*args):
#     pass

# class Differential_generic:
#     """A differential one-form which can be defined on a Riemann surface.

#     Attributes
#     ----------
#     numer, denom : MultivariatePolynomial
#         Fast multivariate polynomial objects representing the numerator
#         and denominator of the differential.

#     Methods
#     -------
#     eval
#     as_numer_denom
#     as_sympy_expr

#     Notes
#     -----
#     To reduce the number of discriminant points to check for computing
#     the valuation divisor we keep separate the numerator and denominator
#     of the Differential. This behavior may change after implementing
#     different types of differentials.

#     """
#     def __cinit__(self, RS, *args):
#         """Create a differential on the Riemann surface `RS`.

#         Parameters
#         ----------
#         RS : RiemannSurface
#             The Riemann surface on which the differential is defined.
#         *args : list
#             A differential can be instantiated by a single Sympy
#             expression or by a numerator and denominator. (Also as Sympy
#             expressions.) The latter distinction makes analysis of
#             Abelian differentials of the first kind more computationally
#             efficient.

#         """
#         if (len(args) < 1) or (len(args) > 2):
#             raise ValueError('Instantiate Differential with Sympy expression '
#                              'or numerator/denominator pair.')

#         # determine the numerator and denominator of the differentials
#         if len(args) == 1:
#             numer, denom = args[0].as_numer_denom()
#             self.numer = numer
#             self.denom = denom
#         elif len(args) == 2:
#             self.numer = args[0]
#             self.denom = args[1]

#         self.RS = RS
#         self.x = RS.x
#         self.y = RS.y
#         self.numer_n = MultivariatePolynomial(self.numer,self.x,self.y)
#         self.denom_n = MultivariatePolynomial(self.denom,self.x,self.y)

#     def __repr__(self):
#         s = ''
#         if self.numer.is_Add:
#             s += '(' + str(self.numer) + ')'
#         else:
#             s += str(self.numer)

#         s += '/'

#         if self.denom.is_Add:
#             s += '(' + str(self.denom) + ')'
#         else:
#             s += str(self.denom)
#         return s

#     cpdef complex eval(self, complex x, complex y):
#         r"""Evaluate the differential at the complex point :math:`(x,y)`.

#         Parameters
#         ----------
#         x,y : complex

#         Returns
#         -------
#         complex
#             Returns the value :math:`\omega(x,y)`.

#         """
#         return self.numer_n.eval(x,y) / self.denom_n.eval(x,y)

#     def centered_at_place(self, P, order=None):
#         r"""Rewrite the differential in terms of the local coordinates at `P`.

#         If `P` is a regular place, then returns `self` as a sympy
#         expression. Otherwise, if `P` is a discriminant place
#         :math:`P(t) = \{x(t), y(t)\}` then returns

#         .. math::

#             \omega |_P = q(x(t),y(t)) x'(t) / \partial_y f(x(t),y(t)).

#         Parameters
#         ----------
#         P : Place
#         order : int, optional
#             Passed to :meth:`PuiseuxTSeries.eval_y`.

#         Returns
#         -------
#         sympy.Expr
#         """
#         x = self.x
#         y = self.y

#         # by default, non-discriminant places do not store Pusieux series
#         # expansions. this might change in the future
#         if P.is_discriminant():
#             p = P.puiseux_series
#             t = P.t
#         else:
#             t = sympy.Symbol('t')
#             p = puiseux(self.RS.f,self.x,self.y,P.x,P.y,t)[0]

#         # symbolically evaluate the x, y, and dx parts of the series
#         xt = p.eval_x(t)
#         yt = p.eval_y(t,order=order)
#         dxdt = p.eval_dxdt(t)

#         # substitute Puiseux series expansion into the differrential and expand
#         # as a Laurent series. Sympy's nseries is slow in some rational exprs
#         expr = self.as_sympy_expr()
#         expr = expr.subs({x:xt,y:yt})*dxdt
#         numer,denom = sympy.cancel(expr).as_numer_denom()
#         omega = fast_expand(numer,denom,t)
#         return omega

#     def localize(self, *args, **kwds):
#         r"""Same as :meth:`centered_at_place`."""
#         return self.centered_at_place(*args, **kwds)

#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     cpdef complex[:] evaluate(self, RiemannSurfacePathPrimitive gamma,
#                               double[:] t):
#         r"""Evaluates `omega` along the path at `N` uniform points.

#         .. note::

#             Note: right now it doesn't matter what the values in `t`
#             are. This function will simply turn `t` into a bunch of
#             uniformly distributed points between 0 and 1.

#         Parameters
#         ----------
#         omega : Differential
#         t : double[:]
#             An array of `t` between 0 and 1.

#         Returns
#         -------
#         complex[:]
#             The differential omega evaluated along the path at `N` points.
#         """
#         return gamma.evaluate(self,t)

#     def _find_necessary_xvalues(self):
#         r"""Returns a list of x-points over which the places appearing in the
#         valuation divisor are found.

#         :py:meth:`valuation_divisor` requires a necessary list of x-values from
#         which to compute places which may appear in the valuation divisor of
#         the differential.

#         In the case when `self.denom` is equal to :math:`\partial_y f` we
#         simply use the discriminant points of the curve.

#         Parameters
#         ----------
#         none

#         Returns
#         -------
#         list
#         """
#         f = self.RS.f
#         x = self.x
#         y = self.y

#         # get possible x-values from the numerator by computing the roots of
#         # the resolvent with the curve f
#         numer = self.numer
#         res = sympy.resultant(numer,f,y).as_poly(x)
#         numer_roots = res.all_roots(multiple=False, radicals=False)
#         if numer_roots:
#             numer_roots,_ = zip(*numer_roots)

#         # get possible x-values from the denominator. in the case when the
#         # denominator is dfdy these are simply the discriminant points
#         denom = self.denom
#         if (denom.expand() == f.diff(y).expand()):
#             denom_roots = self.RS.discriminant_points()
#         else:
#             res = sympy.resultant(denom,f,y).as_poly(x)
#             denom_roots = res.all_roots(multiple=False, radicals=False)
#             if denom_roots:
#                 denom_roots,_ = zip(*denom_roots)

#         # finally, the possible x-points contributed by dx are the discriminant
#         # points of the curve
#         discriminant_points = self.RS.discriminant_points()

#         # form the set of x-values over which to compute places. reorder
#         # entries such that x=0 and x=oo appear first because differential
#         # numerators tend to be monomial, resulting in better performance.
#         xvalues = []
#         roots = set([]).union(numer_roots)
#         roots = roots.union(denom_roots)
#         roots = roots.union(discriminant_points)
#         if 0 in roots:
#             xvalues.append(0)
#             roots.discard(0)
#         xvalues.append(sympy.oo)  # account for all places at infinity
#         xvalues.extend(roots)
#         return xvalues

#     def valuation_divisor(self, **kwds):
#         r"""Returns the valuation divisor of the differential.

#         This is a generic algorithm for computing valuation divisors and should
#         only be used if nothing is known about the differential in question.

#         If the differential is Abelian of the first kind (holomorphic) then
#         create an instance of :class:`AbelianDifferentialFirstKind`. Similarly,
#         if the differential is Abelian of the second kind then create an
#         instance of :class:`AbelianDifferentialSecondKind`. These implement
#         versions of :meth:`valuation_divisor` that use properties of the
#         differential to save on time.

#         Parameters
#         ----------
#         none

#         Returns
#         -------
#         Divisor

#         """
#         xvalues = self._find_necessary_xvalues()

#         # for each xvalue, compute the places above it and determine the
#         # valuation of the differential over each of these places
#         D = Divisor(self.RS,0)
#         genus = self.RS.genus()
#         for alpha in xvalues:
#             places_above_alpha = self.RS(alpha)
#             for P in places_above_alpha:
#                 n = P.valuation(self)
#                 D += n*P

#         # the valuation divisor of a generic meromorphic differential is still
#         # canonical. check the degree condition
#         target_genus = 2*genus - 2
#         if D.degree != target_genus:
#             raise ValueError(
#                 'Could not compute valuation divisor of %s: '
#                 'did not reach genus requirement.'%self)
#         return D

#     def plot(self, RiemannSurfacePathPrimitive gamma, N=256, grid=False,
#              **kwds):
#         r"""Plot the differential along the RiemannSurfacePath `gamma`.

#         Parameters
#         ----------
#         gamma : RiemannSurfacePath
#             A path along which to evaluate the differential.
#         N : int
#             Number of interpolating points to use when plotting the
#             value of the differential along the path `gamma`
#         grid : boolean
#             (Default: `False`) If true, draw gridlines at each "segment"
#             of the parameterized RiemannSurfacePath. See the
#             `RiemannSurfacePath` documentation for more information.

#         Returns
#         -------
#         matplotlib.Figure

#         """
#         fig = plt.figure()
#         ax = fig.add_subplot(1,1,1)
#         ax.hold(True)

#         nseg = len(gamma.segments)
#         t = numpy.linspace(0,1,N/nseg)
#         for k in range(nseg):
#             segment = gamma.segments[k]
#             osegment = numpy.array(self.evaluate(segment,t),dtype=complex)
#             tsegment = (t+k)/nseg;
#             ax.plot(tsegment, osegment.real, 'b')
#             ax.plot(tsegment, osegment.imag, 'r--')

#         # plot gridlines at the interface between each set of segments
#         if grid:
#             ticks = numpy.linspace(0,1,len(gamma.segments)+1)
#             ax.xaxis.set_ticks(ticks)
#             ax.grid(True, which='major')
#         return fig

#     def as_numer_denom(self):
#         """Returns the differential as a numerator, denominator pair.

#         Returns
#         -------
#         list, sympy.Expr
#         """
#         return self.numer, self.denom

#     def as_sympy_expr(self):
#         """Returns the differential as a Sympy expression.

#         Returns
#         -------
#         sympy.Expr
#         """
#         return self.numer / self.denom


# cdef class AbelianDifferentialFirstKind(Differential):
#     def valuation_divisor(self, proof=False, **kwds):
#         r"""Returns the valuation divisor of the Abelian differential of the
#         first kind.

#         Because Abelian differentials of the first kind are holomorphic on the
#         Riemann surface, the valuation divisor is of the form

#         .. math::

#             (\omega)_{val} = p_1 P_1 + \cdots + p_m P_m

#         where :math:`\omega` has a zero of multiplicity :math:`p_k` at the
#         place :math:`P_k`.

#         Parameters
#         ----------
#         proof : bool
#             If set to `True`, will provably return the valuation divisor by
#             computing the valuation at every necessary place on `X`. Slow.

#         Notes
#         -----
#         This valuation divisor overload takes advantage of the fact that the
#         differential admits no poles. Therefore, as places on the Riemann
#         surface are checked, the degree of the valuation divisor is
#         non-decreasing. We can terminate the search the moment the degree
#         reaches :math:`2g-2`. If `proof=True` then ignore this trick and
#         compute over every possible place.

#         """
#         xvalues = self._find_necessary_xvalues()

#         # for each xvalue, compute the places above it and determine valuation
#         # of the differential over each of these places.
#         D = Divisor(self.RS,0)
#         genus = self.RS.genus()
#         target_genus = 2*genus - 2
#         for alpha in xvalues:
#             places_above_alpha = self.RS(alpha)
#             for P in places_above_alpha:
#                 n = P.valuation(self)
#                 D += n*P

#                 # abelian differentials of the first kind should have no poles
#                 if n < 0:
#                     raise ValueError(
#                         'Could not compute valuation divisor of %s: '
#                         'found a pole of differential of first kind.'%self)

#                 # break out if the target degree is met
#                 if (D.degree == target_genus) and (not proof):
#                     return D

#         if D.degree != target_genus:
#             raise ValueError('Could not compute valuation divisor of %s: '
#                              'did not reach genus requirement.'%self)
#         return D


# cdef class AbelianDifferentialSecondKind(Differential):
#     r"""Defines an Abelian Differential of the second kind.

#     An Abelian differential of the second kind is one constructed in the
#     following way: given a place :math:`P \in X` and a positive integer
#     :math:`m` an Abelian differential of second kind is a meromorphic
#     differential with a pole only at :math:`P` of order :math:`m+1`.
#     """
#     def valuation_divisor(self, **kwds):
#         r"""Returns the valuation divisor of the Abelian differential of the
#         second kind.

#         Parameters
#         ----------
#         none

#         Returns
#         -------
#         Divisor

#         """
#         xvalues = self._find_necessary_xvalues()

#         # for each xvalue, compute the places above it and determine valuation
#         # of the differential over each of these places.
#         D = Divisor(self.RS,0)
#         genus = self.RS.genus()
#         target_genus = 2*genus - 2
#         num_poles = 0
#         for alpha in xvalues:
#             places_above_alpha = self.RS(alpha)
#             for P in places_above_alpha:
#                 n = P.valuation(self)
#                 D += n*P

#                 # differentials of the second kind should have a single
#                 # pole. raise an error if more are found
#                 if n < 0: num_poles += 1
#                 if num_poles > 1:
#                     raise ValueError(
#                         'Could not compute valuation divisor of %s: '
#                         'found more than one pole.'%self)

#                 # break out if (a) the degree requirement is met and (b) the
#                 # pole was found.
#                 if (D.degree == target_genus) and (num_poles):
#                     return D

#         if D.degree != target_genus:
#             raise ValueError('Could not compute valuation divisor of %s: '
#                              'did not reach genus requirement.'%self)
#         return D
