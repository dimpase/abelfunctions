"""
RiemannSurfaces
===============

Authors
-------

* Chris Swierczewski (January 2014)
"""

import numpy
import scipy
import scipy.linalg

from abelfunctions.differentials import differentials
from abelfunctions.differentials import Differential
from abelfunctions.differentials cimport Differential
from abelfunctions.divisor import Place, DiscriminantPlace, RegularPlace, Divisor
from abelfunctions.puiseux import puiseux
from abelfunctions.riemann_surface_path import RiemannSurfacePathPrimitive
from abelfunctions.riemann_surface_path cimport RiemannSurfacePathPrimitive
from abelfunctions.riemann_surface_path_factory import RiemannSurfacePathFactory
from abelfunctions.singularities import genus

from sage.all import QQbar, infinity


cdef class RiemannSurface:
    """A Riemann surface defined by a complex plane algebraic curve.

    Attributes
    ----------
    f : sympy.Expression
        The algebraic curve representing the Riemann surface.
    """
    property f:
        def __get__(self):
            return self._f
    property deg:
        def __get__(self):
            return self._deg
    property PF:
        def __get__(self):
            return self.PathFactory

    def __init__(self, f, base_point=None, base_sheets=None, kappa=2./5):
        """Construct a Riemann surface.

        Parameters
        ----------
        f : curve
            The algebraic curve representing the Riemann surface.
        base_point : complex, optional
            A custom base point for the Monodromy group.
        base_sheets : complex list, optional
            A custom ordering of the sheets at the base point.
        kappa : double
            A scaling parameter greater than 0 but less than 1 used to
            define the radii of the x-path circles around the curve's
            branch points.
        """
        f = f.change_ring(QQbar)
        R = f.parent()
        x,y = R.gens()
        self._f = f
        self._deg = f.degree(y)

        # set custom base point, if provided. otherwise, base_point is
        # set by self.discriminant_points()
        self._base_point = base_point
        self._discriminant_points = None
        self._discriminant_points_exact = None
        self.discriminant_points()  # sets the base point of the surface

        # set the base sheets
        if base_sheets:
            self._base_sheets = base_sheets
        else:
            self._base_sheets = self.base_sheets()

        # cache for key calculations
        self._base_place = self(self._base_point)[0]
        self._period_matrix = None
        self._riemann_matrix = None
        self._genus = None
        self._holomorphic_differentials = None
        self.PathFactory = RiemannSurfacePathFactory(self)
        
    def __repr__(self):
        s = 'Riemann surface defined by f = %s'%(self.f)
        return s

    def __call__(self, alpha, beta=None):
        r"""Returns a place or places on the Riemann surface.

        Parameters
        ----------
        alpha : complex or sympy.Expr
            The x-projection of the place.
        beta : complex or sympy.Expr (optional)
            If provided, will only return places with the given
            y-projection. There may be multiple places on the surface
            with the same x- and y-projections.

        Returns
        -------
        Place or list of Places
            If multiple places

        """
        # alpha = infinity case
        infinities = [infinity, 'oo', numpy.Inf]
        if alpha in infinities:
            alpha = infinity
            p = puiseux(self.f, alpha)
            return [DiscriminantPlace(self, pi) for pi in p]

        # if alpha is epsilon close to a discriminant point then set it exactly
        # equal to that discriminant point. there is usually no reason to
        # compute a puiseux series so close to a discriminant point
        try:
            alpha = QQbar(alpha)
            exact = True
        except TypeError:
            alpha = numpy.complex(alpha)
            exact = False
        b = self.closest_discriminant_point(alpha,exact=exact)

        # if alpha is equal to or close to a discriminant point then return a
        # discriminant place
        if abs(alpha - b) < 1e-12:
            p = puiseux(self.f, b, beta)
            return [DiscriminantPlace(self, pi) for pi in p]

        # otherwise, return a regular place if far enough away
        if not beta is None:
            curve_eval = self.f(alpha, beta)
            if abs(curve_eval) > 1e-8:
                raise ValueError('The place (%s, %s) does not lie on the curve '
                                 '/ surface.')
            return RegularPlace(self, alpha, beta)

        # if a beta (y-coordinate) is not specified then return all places
        # lying above x=alpha
        R = self.f.parent()
        x,y = R.gens()
        falpha = self.f(alpha,y).univariate_polynomial()
        yroots = falpha.roots(ring=falpha.base_ring(), multiplicities=False)
        return [RegularPlace(self, alpha, beta) for beta in yroots]

    def show_paths(self, ax=None, *args, **kwds):
        """Plots all of the monodromy paths of the curve.

        Parameters
        ----------
        ax : matplotlib.Axes
            The figure axes on which to plot the paths.

        Returns
        -------
        None
        """
        self.PathFactory.show_paths(ax=ax, *args, **kwds)

    def discriminant_points(self, exact=True):
        r"""Returns the discriminant points of the underlying curve.

        A discriminant point :math:`x=b` is an x-point where at least
        one y-root lying above has multiplicity greater than one. A
        :class:`PuiseuxTSeries` is required to represent a place on the
        Riemann surface whose x-projection is a discriminant
        point. These kinds of places are of type
        :class:`DiscriminantPlace`.

        .. note::

            The ordering of the discriminant points is important for the
            purposes of computing the monodromy group, which is done in
            the :class:`RiemannSurfacePathFactory` attribute,
            `PathFactory`.

        Parameters
        ----------
        exact : boolean
            If `True`, returns symbolic discriminant points. Otherwise,
            returns a numerical approximation. Both are cached for
            performance.

        Returns
        -------
        list
            A list of the discriminant points of the underlying curve.

        """
        # use cached discriminant points if available
        if not self._discriminant_points is None:
            if exact:
                return self._discriminant_points_exact
            return self._discriminant_points

        # compute the symbolic and numerical discriminant points
        f = self.f
        x,y = f.parent().gens()
        res = f.resultant(f.derivative(y),y).univariate_polynomial()
        rts = res.roots(ring=QQbar, multiplicities=False)
        discriminant_points_exact = numpy.array(rts)
        discriminant_points = discriminant_points_exact.astype(numpy.complex)

        # determine a base_point, if not specified
        if self._base_point is None:
            a = min(bi.real - 1 for bi in discriminant_points)
            aint = numpy.complex(numpy.floor(a))
            self._base_point = aint

        # sort the discriminant points first by argument with the base
        # point and then by distance from the base point.
        centered_points = discriminant_points - a
        distances = numpy.abs(centered_points)
        arguments = numpy.angle(centered_points)
        sort_index = numpy.lexsort((distances, arguments))

        # cache and return
        self._discriminant_points_exact = discriminant_points_exact[sort_index]
        self._discriminant_points = discriminant_points[sort_index]
        if exact:
            return self._discriminant_points_exact
        return self._discriminant_points

    def closest_discriminant_point(self, x, exact=True):
        r"""Returns the closest discriminant point to a point x.

        An often-used helper function by several components of
        :class:`RiemannSurface`.

        Parameters
        ----------
        x : complex
            A complex x-point.
        exact : boolean
            If `True`, returns a `sympy.Expr` representing the
            discriminant point exactly. Otherwise, returns a numerical
            approximation.

        Returns
        -------
        complex or sympy.Expr
            The discriminant point, either exact or numerical.
        """
        b = self.discriminant_points(exact=exact)
        bf = self.discriminant_points(exact=False)

        # for performance, coerce everything to floating point
        # approximations. if the discriminant points are less than 1e-16
        # apart then we're screwed, anyway.
        x = numpy.complex(x)
        idx = numpy.argmin(numpy.abs(bf - x))
        if exact:
            return b[idx]
        return bf[idx]

    # Monodromy: expose some methods / properties of self.Monodromy
    # without subclassing (since it doesn't make sense that a Riemann
    # surface is a type of Monodromy group.)
    def monodromy_group(self):
        r"""Returns the monodromy group of the underlying curve.

	    The monodromy group is represented by a list of four items:

        * `base_point` - a point in the complex x-plane where every monodromy
          path begins and ends,
        * `base_sheets` - the y-roots of the curve lying above `base_point`,
        * `branch_points` - the branch points of the curve,
        * `permutations` - the permutations of he base sheets corresponding
          to each branch point.

        """
        return self.PathFactory.monodromy_group()

    def base_point(self):
        r"""Returns the base x-point of the Riemann surface.
        """
        return self._base_point

    def base_place(self):
        r"""Returns the base place of the Riemann surface.

        The base place is the place from which all paths on the Riemann
        surface are constructed. The AbelMap begins integrating from the
        base place.

        Parameters
        ----------
        None

        Returns
        -------
        Place

        """
        return self._base_place

    def base_sheets(self):
        r"""Returns the base sheets of the Riemann surface.

        The base sheets are the y-roots lying above the base point of
        the surface.  The base place of the Riemann surface is given by
        the base x-point and the first element of the base sheets.

        Parameters
        ----------
        None

        Returns
        -------
        list, complex
            An ordered list of roots lying above the base point of the
            curve.

        """
        # returned cached base sheets if availabe
        if not self._base_sheets is None:
            return self._base_sheets
        self._base_sheets = self.lift(self._base_point)
        return self._base_sheets

    def lift(self, x0):
        r"""List the x-point `x` to the fibre of y-roots.

        Basically, computes the y-roots of :math:`f(x,y) = 0` for the
        given `x`.

        .. note::

            The y-roots are given in no particular order. Be careful
            when using these to construct :class:`RiemannSurfacePath`
            objects.

        Parameters
        ----------
        x : complex

        Returns
        -------
        list, complex
        """
        # compute the base sheets
        R = self.f.parent()
        x,y = R.gens()
        p = self.f(x0,y).univariate_polynomial()
        lift = p.roots(ring=p.base_ring(), multiplicities=False)
        return lift

    def base_lift(self):
        r"""Same as :meth:`base_sheets`."""
        return self.base_sheets()

    def branch_points(self):
        return self.PathFactory.branch_points()

    def holomorphic_differentials(self):
        r"""Returns a basis of holomorphic differentials on the surface.

        Parameters
        ----------
        None

        Returns
        -------
        list, HolomorphicDifferential

        """
        if not self._holomorphic_differentials:
            self._holomorphic_differentials = differentials(self)
        return self._holomorphic_differentials

    def holomorphic_oneforms(self):
        r"""Alias for :meth:`holomorphic_differentials`."""
        return self.holomorphic_differentials()

    def genus(self):
        if not self._genus:
            self._genus = genus(self.f)
        return self._genus

    def a_cycles(self):
        return self.PF.a_cycles()

    def b_cycles(self):
        return self.PF.b_cycles()

    def c_cycles(self):
        return self.PF.c_cycles()

    def path(self, P, P0=None):
        r"""Constructs a path to the place `P`.

        Parameters
        ----------
        P : Place
            The place

        Returns
        -------
        RiemannSurfacePath

        """
        return self.PathFactory.path_to_place(P)

    cpdef complex integrate(self, Differential omega,
                            RiemannSurfacePathPrimitive gamma):
        r"""Integrate the differential `omega` over the path `gamma`.

        Parameters
        ----------
        omega : Differenial
            A differential defined on the Riemann surface.
        gamma : RiemannSurfacePathPrimitive
            A continuous path on the Riemann surface.

        Returns
        -------
        complex
            The integral of `omega` on `gamma`.

        """
        cdef complex value = gamma.integrate(omega)
        return value

    def period_matrix(self):
        r"""Returns the period matrix of the Riemann surface.

        The period matrix is obtained by integrating a basis of
        holomorphic one-forms over a first homology group basis.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.array
            A :math:`g \times 2g` complex matrix of periods.

        """
        if not (self._period_matrix is None):
            return self._period_matrix

        print 'Computing c_cycles:'
        c_cycles, linear_combinations = self.c_cycles()
        oneforms = self.holomorphic_oneforms()
        c_periods = []
        g = self.genus()
        m = len(c_cycles)
        print c_cycles
        print

        for omega in oneforms:
            omega_periods = []
            for gamma in c_cycles:
                print 'Computing c-period:'
                omega_periods.append(self.integrate(omega, gamma))
                print omega_periods
            c_periods.append(omega_periods)

        # take appropriate linear combinations of the c-periods to
        # obtain the a- and b-periods
        #
        # tau[i,j] = \int_{a_j} \omega_i,  j < g
        # tau[i,j] = \int_{b_j} \omega_i,  j >= g
        #
        tau = numpy.zeros((g,2*g), dtype=numpy.complex)
        for i in range(g):
            for j in range(2*g):
                tau[i,j] = sum(linear_combinations[j,k] * c_periods[i][k]
                               for k in range(m))

        self._period_matrix = tau
        return self._period_matrix

    def riemann_matrix(self):
        r"""Returns the Riemann matrix of the Riemann surface.

        A Riemann matrix of the surface is obtained by normalizing the
        chosen basis of holomorphic differentials.

        .. math::

            \tau = [A \; B] = [I \; \Omega]

        Parameters
        ----------
        None

        Returns
        -------
        numpy.array
            A :math:`g \times g` Riemann matrix corresponding to the
            Riemann surface.

        """
        if not self._riemann_matrix is None:
            return self._riemann_matrix

        g = self.genus()
        tau = self.period_matrix()
        A = tau[:,:g]
        B = tau[:,g:]
        self._riemann_matrix = numpy.dot(scipy.linalg.inv(A), B)
        return self._riemann_matrix
