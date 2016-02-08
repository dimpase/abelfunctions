r"""Complex Paths :mod:`abelfunctions.complex_path`

Data structures for paths in the complex plane.

Classes
-------

.. autosummary::

  ComplexPathPrimitive
  ComplexPath
  ComplexPathLine
  ComplexPathArc
  ComplexPathRay

Contents
--------
"""

import numpy

from numpy import Infinity, complex
from sage.functions.other import real_part, imag_part, floor
from sage.plot.line import line


class ComplexPathPrimitive(object):
    r"""
    Base class for paths in the complex plane.
    """
    @property
    def segments(self):
        return [self]

    ##############################
    # overload these in subclass #
    ##############################
    def __init__(self, *args):
        raise NotImplementedError()

    def __repr__(self, type='path'):
        if not self._segments:
            return 'Null Path on %s'%self.riemann_surface

        start_point = self[0](0.0)
        end_point = self[-1](1.0)
        s = 'Complex path from %s to %s'%(start_point, end_point)
        return s

    def eval(self, s):
        raise NotImplementedError()

    def derivative(self, s):
        raise NotImplementedError()

    ######################
    # additional methods #
    ######################
    def __add__(self, other):
        # assert that the path is continuous
        self_end = self(1.0)
        other_start = other(0.0)
        eps = 1e-14
        if abs(self_end - other_start) > eps:
            raise ValueError('Cannot form sum of complex paths: ending point '
                             'of left does not match start point of right.')

        # form the complex path
        segments = self.segments + other.segments
        gamma = ComplexPath(*segments)
        return gamma

    def __call__(self, s):
        return self.eval(s)

    def eval(self, s):
        raise NotImplementedError('Implement in subclass.')

    def derivative(self, s):
        raise NotImplementedError('Implement in subclass.')

    def reverse(self):
        raise NotImplementedError('Implement in subclass.')

    def plot(self, plot_points=128, **kwds):
        r"""Return a plot of the path.

        Parameters
        ----------
        plot_points : int or list
            The number or plot points or a list of parameter values lying in
            the interval [0,1]. (Default: 128)
        **kwds : dict
            Additional keywords passed to `sage.plot.line.line`.

        Returns
        -------
        plt : Sage plot
            A plot of the complex path.
        """
        # if the plot_points are given as a list then use the list of
        # parameters. otherwise, create a linspace
        s = plot_points
        if not (isinstance(s, list) or isinstance(s, numpy.ndarray)):
            s = numpy.linspace(0,1,s)

        # s is now a list of points. compute the path points and draw
        vals = [self(si) for si in s]
        pts = [(real_part(x), imag_part(x)) for x in vals]
        plt = line(pts, **kwds)
        return plt


class ComplexPath(ComplexPathPrimitive):
    r"""
    A composite path in the complex plane.

    Every `ComplexPath` is composed of individual primitive paths, called
    "segments". Every path is parameterized from `s=0` to `s=1`. `ComplexPath`
    follows the composite design pattern.

    Attributes
    ----------
    segments : list
        A list of the constituent segments of the path.

    """
    @property
    def segments(self):
        return self._segments

    def __init__(self, *args):
        r"""Directly instantiate an ComplexPath composite from a list of
        ComplexPathPrimitives.

        Parameters
        ----------
        *args : list
            A list of :class:`ComplexPathPrimitive`s.
        """
        # assert that the segments form a continuous path
        n = len(args)
        eps = 1e-14
        for k in range(n-1):
            gamma0 = args[0]
            gamma1 = args[1]
            if abs(gamma1(0.0) - gamma0(1.0)) > eps:
                raise ValueError('Segments must form continuous path.')

        self._segments = list(args)
        self._nsegments = n

    def __getitem__(self, index):
        r"""Return the segment at index `index`"""
        return self._segments[index]

    def segment_index_at_parameter(self, s):
        r"""Returns the index of the complex path segment located at the given
        parameter :math:`s \in [0,1]`.

        Parameters
        ----------
        s : float
            Path parameter in the interval [0,1].

        Returns
        -------
        index : int
            The index `k` of the path segment :math:`\gamma_k`.
        """
        # the following is a fast way to divide the interval [0,1] into n
        # partitions and determine which partition s lies in. since this is
        # done often it needs to be fast
        k = floor(s*self._nsegments)
        diff = (self._nsegments - 1) - k
        dsgn = diff >> 31
        return k + (diff & dsgn)

    def eval(self, s):
        r"""Return the complex point along the path at the parameter `s`.

        .. note::

            Directly called by :meth:`__call__`.

        Parameters
        ----------
        s : float
            Path parameter in the interval [0,1].

        Returns
        -------
        val : complex
            The point
        """
        k = self.segment_index_at_parameter(s)
        s_seg = s*self._nsegments - k
        seg = self._segments[k]
        val = seg.eval(s_seg)
        return val

    def derivative(self, s):
        r"""Return the derivative of the complex path with respect to the
        parameter.

        Parameters
        ----------
        s : float
            Path parameter in the interval [0,1].

        Returns
        -------
        index : int
            The index `k` of the path segment :math:`\gamma_k`.
        """
        k = self.segment_index_at_parameter(s)
        s_seg = s*self._nsegments - k
        seg = self._segments[k]
        val = seg.derivative(s_seg)
        return val

    def reverse(self):
        r"""Return the path reversed.

        Parameters
        ----------
        None

        Returns
        -------
        gamma : ComplexPath
        """
        reversed_segments = [segment.reverse() for segment in self[::-1]]
        gamma = ComplexPath(*reversed_segments)
        return gamma

    def plot(self, plot_points=128, **kwds):
        r"""Return a plot of the path.

        Parameters
        ----------
        plot_points : int or list
            The number or plot points or a list of parameter values lying in
            the interval [0,1]. (Default: 128)
        **kwds : dict
            Additional keywords passed to `sage.plot.line.line`.

        Returns
        -------
        plt : Sage plot
            A plot of the complex path.
        """
        # if explicit points are given then plot as usual
        if (isinstance(plot_points, list) or
            isinstance(plot_points, numpy.ndarray)):
            return ComplexPathPrimitive.plot(self, s_seg, )

        # otherwise, plot one segment at a time so as to include the endpoints
        # of each segment (otherwise, it looks fragmented)
        s_seg = floor(plot_points / self._nsegments)
        plt = sum(seg.plot(s_seg, **kwds) for seg in self._segments)
        return plt

class ComplexLine(ComplexPathPrimitive):
    r"""A line segment in the complex plane.

    Attributes
    ----------
    x0 : complex
        The starting point of the line.
    x1 : complex
        The ending point of the line.
    """
    def __init__(self, x0, x1):
        self.x0 = numpy.complex(x0)
        self.x1 = numpy.complex(x1)

    def __repr__(self):
        s = 'Line(%s,%s)'%(self.x0, self.x1)
        return s

    def __eq__(self, other):
        if not isinstance(other, ComplexLine):
            return False
        if (self.x0 == other.x0) and (self.x1 == other.x1):
            return True
        return False

    def eval(self, s):
        val = self.x0 + (self.x1-self.x0)*s
        return val

    def derivative(self, s):
        val = self.x1 - self.x0
        return val

    def reverse(self):
        return ComplexLine(self.x1, self.x0)


class ComplexArc(ComplexPathPrimitive):
    r"""A complex arc. (Part of a circle in the complex plane.)

    Attributes
    ----------
    R : complex
        The radius of the arc.
    w : complex
        The center of the arc.
    theta : complex
        The starting angle (in radians) on the arc. Usually 0 or :math:`\pi`.
    dtheta : complex
        The number of radians to travel where the sign of `dtheta`
        indicates direction. The absolute value of `dtheta` is equal to
        the arc length.
    """
    def __init__(self, R, w, theta, dtheta):
        self.R = numpy.complex(R)
        self.w = numpy.complex(w)
        self.theta = numpy.complex(theta)
        self.dtheta = numpy.complex(dtheta)

    def __repr__(self):
        s = 'Arc(%s,%s,%s,%s)'%(self.R, self.w, self.theta, self.dtheta)
        return s

    def __eq__(self, other):
        if not isinstance(other, ComplexArc):
            return False
        if ((self.R == other.R) and (self.w == other.w) and
            (self.theta == other.theta) and (self.dtheta == other.dtheta)):
            return True
        return False

    def eval(self, s):
        val = self.R*numpy.exp(1.0j*(self.theta + s*self.dtheta)) + self.w
        return val

    def derivative(self, s):
        val = (self.R*1.0j*self.dtheta) * \
              numpy.exp(1.0j*(self.theta + s*self.dtheta))
        return val

    def reverse(self):
        return ComplexArc(self.R, self.w, self.theta+self.dtheta, -self.dtheta)


class ComplexRay(ComplexPathPrimitive):
    r"""A complex ray: a path with a finite starting point going to infinity.

    Attributes
    ----------
    x0 : complex
        The starting point of the ray.
    """
    def __init__(self, x0):
        if abs(x0) < 1e-12:
            raise ValueError('Complex rays must start away from the origin.')
        self.x0 = numpy.complex(x0)

    def __repr__(self):
        s = 'Arc(%s)'%self.x0
        return s

    def __eq__(self, other):
        if not isinstance(other, ComplexRay):
            return False
        if self.x0 != other.x0:
            return False
        return True

    def eval(self, s):
        if s == 1.0: return Infinity
        val = self.x0/(1.-s)
        return val

    def derivative(self, s):
        if s == 1.0: return Infinity
        val = -self.x0/(1.-s)**2
        return val

    def reverse(self):
        raise ValueError('Cannot reverse paths to infinity.')
