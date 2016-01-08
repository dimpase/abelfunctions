r"""Analytic Continutation :mod:`abelfunctions.analytic_continuation`
=================================================================

Objects for performing analytic continuation along a
:class:`RiemannSurfacePath` object and, in fact, is a member object of
class:`RiemannSurfacePath`.

This module contains an abstract class :class`:AnalyticContinuator`

Classes
-------

.. autosummary::

  AnalyticContinuator
  AnalyticContinuatorPuiseux

Functions
---------

Examples
--------

Contents
--------

"""

import numpy
import scipy

from abelfunctions.divisor import Place, DiscriminantPlace
from abelfunctions.puiseux import puiseux
from abelfunctions.utilities import matching_permutation

from sage.all import CC, QQbar, infinity

class AnalyticContinuator:
    r"""Abstract class for analytically continuing along a curve.

    An analytic continutator object dictates how to continue a y-fibre from one
    x-point :math:`x_i` to another point :math:`x_{i+1}` along a
    :class:`RiemannSurfacePath` object. (Technically, a
    :class:`RiemannSurfacePathPrimitive` object which serves as the base class
    for all path types.)

    Attributes
    ----------
    RS : RiemannSurface
        The Riemann surface on which analytic continuation takes place
    gamma : RiemannSurfacePathPrimitive
        The path along which the analytic continuation is performed.

    Methods
    -------
    analytically_continue

    """
    def __init__(self, RS, gamma):
        r"""AnalyticContinuators are initialized with a `RiemannSurface`."""
        self.RS = RS
        self.gamma = gamma
        self.deg = self.RS.deg

    def analytically_continue(self, xi, yi, xip1):
        r"""Analytically continues the fibre `yi` from `xi` to `xip1`.

        Parameters
        ----------
        xi : complex
            The starting complex x-value.
        yi: complex[:]
            The starting complex y-fibre lying above `xi`.
        xip1: complex
            The target complex x-value.

        Returns
        -------
        complex[:]
            The y-fibre lying above `xip1`.
        """
        raise NotImplementedError('Must override AnalyticContinuator.'
                                  'analytically_continue() method in '
                                  'subclass.')

    def parameterize(self, omega):
        r"""Returns the differential omega parameterized on the path.

        Given a differential math:`\omega = \omega(x,y)dx`, `parameterize`
        returns the differential

        .. math::

            \omega_\gamma(s) = \omega(\gamma_x(s),\gamma_y(s)) \gamma_x'(s)

        where :math:`s \in [0,1]` and :math:`\gamma_x,\gamma_y` and the x- and
        y-components of the path `\gamma` using this analytic continuator.

        Parameters
        ----------
        omega : Differential

        Returns
        -------
        function
        """
        raise NotImplementedError('Must implement AnalyticContinuator.'
                                  'integrate in subclass.')

    def integrate(self, omega):
        r"""Integrate `omega` on the path using this analytic continuator.

        Parameters
        ----------
        omega : Differential

        Returns
        -------
        complex
        """
        raise NotImplementedError('Must implement AnalyticContinuator.'
                                  'integrate in subclass.')


class AnalyticContinuatorPuiseux(AnalyticContinuator):
    r"""Riemann surface path analytic continuation using Puiseux series.

    We must use Puiseux series in order to analytically continue a y-fibre to a
    discriminant point :math:`x=b`. The initial y-fibre establishes the
    ordering of the :class:`PuiseuxXSeries` at :math:`x=b`.

    Attributes
    ----------
    RS : RiemannSurface
        The Riemann surface on which analytic continuation takes place
    gamma : RiemannSurfacePathPrimitive
        The path along which the analytic continuation is performed.
    center : complex or infinity
        The center of the Puiseux series expansions. Usually a discriminant
        point of the underlying curve of the Riemann surface.
    puiseux_series : list, PuiseuxXSeries
        An ordered list of :class:`PuiseuxXSeries` corresponding to the nearby
        branches of the curve.

    Methods
    -------
    analytically_continue
    _compute_puisuex_series

    """
    @property
    def target_place(self):
        return self._target_place

    def __init__(self, RS, gamma, discriminant_point):
        AnalyticContinuator.__init__(self, RS, gamma)
        self._target_place = None
        self.center = discriminant_point
        self.alpha = 0  # overwritten in _self._compute_puisuex_series
        self.puiseux_series = self._compute_puiseux_series(gamma)

    def _compute_puiseux_series(self, gamma, epsilon=1e-8):
        r"""Return the Puiseux series at the center in the correct order.

        In order to analytically continue from the regular places at the
        beginning of the path :math:`x=a` to the discriminant places at the end
        of the path :math:`x=b`we need to compute all of the `PuiseuxXSeries`
        at :math:`x=b`. There are two steps to this calculation:

        * compute enough terms of the Puiseux series centered at :math:`x=b` in
          order to accurately capture the y-roots at :math:`x=a`.

        * permute the series accordingly to match up with the y-roots at
          :math:`x=a`.

        Parameters
        ----------
        gamma : RiemannSurfacePathPrimitive
            The path or path segment starting at a regular point and ending at
            a discriminant point.

        Returns
        -------
        list
            A list of ordered Puiseux series corresponding to each branch at
            :math:`x=a`.

        Notes
        -----
        A more efficient method of extending the :class:`PuiseuxTSeries` could
        probably by designed. In particular, some series may "converge" faster
        than others or may have much higher order terms than the others.

        """

        # obtian the PuiseuxTSeries at x=b (the center)
        f = self.RS.f
        x0 = gamma.x0
        P = puiseux(f, self.center)

        # XXX HACK - need to coerce input to CC
        x0 = CC(x0)

        # compute enouge terms of each Puiseux series to accurately capture the
        # y-roots at x=a
        for Pi in P:
            Pi.extend_to_x(x0, curve_tol=epsilon)

        # now that we have sufficiently many terms we compute the corresponding
        # x-series and reorder them according to the ordering of the incoming
        # regular places roots
        alpha = 0 if self.center == infinity else self.center
        px = [[pxi for pxi in Pi.xseries(all_conjugates=True)] for Pi in P]
        ramification_indices = [Pi.ramification_index for Pi in P]
        p = [pxi for sublist in px for pxi in sublist]
        py = numpy.array([pj(x0-alpha) for pj in p], dtype=numpy.complex)
        sigma = matching_permutation(py, gamma.y0)
        p = sigma.action(p)

        # before returning, obtain store the target place (t-series)
        px_idx = sigma[0] # index of target x-series in unsorted list
        place_idx = -1    # index of place corresponding to this x-series
        while px_idx >= 0:
            place_idx += 1
            px_idx -= numpy.abs(ramification_indices[place_idx])
        self._target_place = DiscriminantPlace(self.RS,P[place_idx])
        return p

    def analytically_continue(self, xi, yi, xip1):
        # XXX HACK - need to coerce input to CC for puiseux series to evaluate
        xi = CC(xi)
        xip1 = CC(xip1)

        # return the current fibre if the step size is too small
        if numpy.abs(xip1-xi) < 1e-15:
            return yi

        # simply evaluate the ordered puiseux series at xip1
        alpha = 0 if self.center == infinity else self.center
        yip1 = numpy.array([pj(xip1-alpha) for pj in self.puiseux_series],
                           dtype=numpy.complex)
        return yip1


    def parameterize(self, omega):
        r"""Returns the differential omega parameterized on the path.

        Given a differential math:`\omega = \omega(x,y)dx`, `parameterize`
        returns the differential

        .. math::

            \omega_\gamma(s) = \omega(\gamma_x(s),\gamma_y(s)) \gamma_x'(s)

        where :math:`s \in [0,1]` and :math:`\gamma_x,\gamma_y` and the x- and
        y-components of the path `\gamma` using this analytic continuator.

        Parameters
        ----------
        omega : Differential

        Returns
        -------
        function
        """
        # localize the differential at the discriminant place
        P = self._target_place
        omega_local = omega.localize(P)
        omega_local = omega_local.laurent_polynomial().change_ring(CC)

        # extract relevant information about the Puiseux series
        p = P.puiseux_series
        x0 = numpy.complex(self.gamma.x0)
        y0 = numpy.complex(self.gamma.y0[0])
        center = numpy.complex(p.center)
        xcoefficient = numpy.complex(p.xcoefficient)
        e = numpy.int(p.ramification_index)

        # the parameter of the path s \in [0,1] does not necessarily match with
        # the local coordinate t of the place. perform the appropriate scaling
        # on the integral.
        tprim = numpy.complex((x0-center)/xcoefficient)**(1./e)
        unity = [numpy.exp(2.j*numpy.pi*k/abs(e)) for k in range(abs(e))]
        tall = [unity[k]*tprim for k in range(abs(e))]
        ytprim = numpy.array([p.eval_y(tk) for tk in tall], dtype=numpy.complex)
        k = numpy.argmin(numpy.abs(ytprim - y0))
        tcoefficient = tall[k]

        # XXX HACK - CC coercion
        tcoefficient = CC(tcoefficient)
        def omega_gamma(s):
            s = CC(s)
            dtds = -tcoefficient
            val = omega_local(tcoefficient*(1-s)) * dtds
            return numpy.complex(val)
        return omega_gamma


    def integrate(self, omega):
        r"""Integrate the differential along the underlying path.

        When integrating a holomorphic differential to place
        corresponding to a discriminant point of the curve care needs to
        be taken since the denominator of the differential vanishes at
        this point.

        Parameters
        ----------
        omega : Differential

        """
        omega_gamma = self.parameterize(omega)
        integral = scipy.integrate.romberg(omega_gamma,0,1)
        return integral
