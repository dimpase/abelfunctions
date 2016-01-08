"""
Analytic Continuation: Smale's Alpha Theory
===========================================

This module implements subclass of :class:AnalyticContinuator which uses
Smale's alpha theory for analytically continuing y-roots. This
AnalyticContinuator is only effective away from the branch points and
singular points of the curve since its primary mechanism is Newton
iteration. A different AnalyticContinuator, such as
:class:AnalyticContinuatorPuiseux, is required in order to analytically
continue to such points.

Functions
---------

  factorial
  newton
  smale_alpha
  smale_beta
  smale_gamma

Classes
-------

  UnivariatePolynomial
  MultivariatePolynomial

Globals::

  ABELFUNCTIONS_SMALE_ALPHA0

"""

import numpy
import scipy
import scipy.integrate

from abelfunctions.analytic_continuation import AnalyticContinuator
from sage.all import CC, factorial, fast_callable

ABELFUNCTIONS_SMALE_ALPHA0 = 1.1884471871911697 # = (13-2*sqrt(17))/4


def newton(df, xip1, yij):
    """Newton iterate a y-root yij of a polynomial :math:`f = f(x,y)`, lying above
    some x-point xi, to the x-point xip1.

    Given :math:`f(x_i,y_{i,j}) = 0` and some complex number :math:`x_{i+1}`,
    this function returns a complex number :math:`y_{i+1,j}` such that
    :math:`f(x_{i+1},y_{i+1,j}) = 0`.

    Parameters
    ----------
    df : MultivariatePolynomial[:]
        A list of all of the y-derivatives of f, including the function f
        itself.
    xip1 : complex
        The x-point to analytically continue to.
    yij: complex
        A y-root at xi. The root that we'll analytically continue.

    Returns
    -------
    A y-root of f lying above `xip1`.

    """
    df0 = df[0]
    df1 = df[1]
    step = numpy.complex(1.0)
    while numpy.abs(step) > 1e-12:
        # if df is not invertible then we are at a critical point.
        df1y = df1(xip1,yij)
        if numpy.abs(df1y) < 1e-12:
            return yij
        step = df0(xip1,yij) / df1y
        yij = yij - step
    return yij

def smale_beta(df, xip1, yij):
    """Compute the Smale beta function at this y-root.

    The Smale beta function is simply the size of a Newton iteration

    Parameters
    ---------
    df : MultivariatePolynomial[:]
        A list of all of the y-derivatives of f, including the function
        f itself.
    xip1 : complex
        The x-point to analytically continue to.
    yij: complex
        A y-root at xi. The root that we'll analytically continue.

    Returns
    -------
    :math:`\beta(f,x_{i+1},y_{i,j})`.
    """
    df0 = df[0]
    df1 = df[1]
    val = numpy.abs(df0(xip1,yij) / df1(xip1,yij))
    return val

def smale_gamma(df, xip1, yij):
    """Compute the Smale gamma function.

    Parameters
    ----------
    df : MultivariatePolynomial
        a list of all of the y-derivatives of f (up to the y-degree)
    xip1 : complex
        the x-point to analytically continue to
    yij : complex
        a y-root at xi. The root that we'll analytically continue.

    Returns
    -------
    double
        The Smale gamma function.
    """
    df0 = df[0]
    df1 = df[1]
    deg = len(df) - 1
    df1y = df1(xip1,yij)
    gamma = numpy.double(0)

    for n in range(2,deg+1):
        dfn = df[n]
        gamman = numpy.abs(dfn(xip1,yij) / (factorial(n)*df1y))
        gamman = gamman**(1.0/(n-1.0))
        if gamman > gamma:
            gamma = gamman
    return gamma


def smale_alpha(df, xip1, yij):
    """Compute Smale alpha.

    Parameters
    ----------
    df : MultivariatePolynomial
        a list of all of the y-derivatives of f (up to the y-degree)
    xip1 : complex
        the x-point to analytically continue to
    yij : complex
        a y-root at xi. The root that we'll analytically continue.

    Returns
    -------
    double
        The Smale alpha function.
    """
    return smale_beta(df,xip1,yij) * smale_gamma(df,xip1,yij)


class AnalyticContinuatorSmale(AnalyticContinuator):
    """Riemann surface path analytic continuation using Smale's alpha theory.

    When sufficiently far away from branch points and singular point of the
    curve we can use Newton iteration to analytically continue the y-roots of
    the curve along paths in :math:`\mathbb{C}_x`. Smale's alpha theory is used
    to determine an optimal step size in :math:`\mathbb{C}_x` to ensure that
    Newton iteration will not only succeed with each y-root but the y-roots
    will not "collide" or swap places with each other. See [XXX REFERENCE XXX]
    for more information.

    .. note::

        This class uses the functions :func:`newton`,
        :func:`smale_alpha`, :func:`smale_beta`, and
        :func:`smale_gamma`, defined in this module.

    Attributes
    ----------
    RS : RiemannSurface
        The Riemann surface on which analytic continuation takes place.
    gamma : RiemannSurfacePathPrimitive
        The path along which the analytic continuation is performed.
    df : MultivariatePolynomial[:]
        A list of all of the y-derivatives of the curve, `f = f(x,y)`.
        These are used by Smale's alpha theory.

    Methods
    -------
    analytically_continue

    """
    def __init__(self, RS, gamma):
        deg = RS.deg
        f = RS.f.change_ring(CC)
        x,y = f.parent().gens()
        df = [fast_callable(f.derivative(y,k), vars=[x,y], domain=numpy.complex)
              for k in range(deg+1)]
        self.df = df
        AnalyticContinuator.__init__(self, RS, gamma)

    def analytically_continue(self, xi, yi, xip1):
        """Analytically continues the fibre `yi` from `xi` to `xip1` using Smale's
        alpha theory.

        Parameters
        ----------
        gamma : RiemannSurfacePathPrimitive
            A Riemann surface path-type object.
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
        # return the current fibre if the step size is too small
        if numpy.abs(xip1-xi) < 1e-15:
            return yi

        # first determine if the y-fibre guesses are 'approximate
        # solutions'. if any of them are not then refine the step by
        # analytically continuing to an intermediate "time"
        for j in range(self.deg):
            yij = yi[j]
            if smale_alpha(self.df, xip1, yij) > ABELFUNCTIONS_SMALE_ALPHA0:
                xiphalf = (xi + xip1)/2.0
                yiphalf = self.analytically_continue(xi, yi, xiphalf)
                yip1 = self.analytically_continue(xiphalf, yiphalf, xip1)
                return yip1

        # next, determine if the approximate solutions will converge to
        # different associated solutions
        for j in range(self.deg):
            yij = yi[j]
            betaij = smale_beta(self.df, xip1, yij)
            for k in range(j+1, self.deg):
                yik = yi[k]
                betaik = smale_beta(self.df, xip1, yik)
                distancejk = numpy.abs(yij-yik)
                if distancejk < 2*(betaij + betaik):
                    # approximate solutions don't lead to distinct
                    # roots. refine the step by analytically continuing
                    # to an intermedite time
                    xiphalf = (xi + xip1)/2.0
                    yiphalf = self.analytically_continue(xi, yi, xiphalf)
                    yip1 = self.analytically_continue(xiphalf, yiphalf, xip1)
                    return yip1

        # finally, since we know that we have approximate solutions that
        # will converge to difference associated solutions we will
        # Netwon iterate
        yip1 = numpy.zeros(self.deg, dtype=numpy.complex)
        for j in range(self.deg):
            yip1[j] = newton(self.df, xip1, yi[j])
        return yip1

    def parameterize(self, omega):
        r"""Returns the differential omega parameterized on the path.

        Given a differential math:`\omega = \omega(x,y)dx`,
        `parameterize` returns the differential

        .. math::

            \omega_\gamma(s) = \omega(\gamma_x(s),\gamma_y(s)) \gamma_x'(s)

        where :math:`s \in [0,1]` and :math:`\gamma_x,\gamma_y` and the
        x- and y-components of the path `\gamma` using this analytic
        continuator.

        Parameters
        ----------
        omega : Differential

        Returns
        -------
        function
        """
        def omega_gamma(t):
            xt = self.gamma.get_x(t)
            yt = self.gamma.get_y(t)[0]
            dxdt = self.gamma.get_dxdt(t)
            return omega(xt,yt) * dxdt
        return numpy.vectorize(omega_gamma, otypes=[numpy.complex])

    def integrate(self, omega):
        r"""Integrate `omega` on the path using this analytic continuator.

        Parameters
        ----------
        omega : Differential

        Returns
        -------
        complex
        """
        integral = numpy.complex(0.0)
        omega_gamma = self.parameterize(omega)
        integral = scipy.integrate.romberg(omega_gamma,0.0,1.0)
        return integral

