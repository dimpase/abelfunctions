r"""X-Path Factory :mod:`abelfunctions.complex_path_factory`
=================================================

Module for computing the monodromy group of the set of discriminant points
of a complex plane algebraic curve.

"""

import numpy
import scipy

from numpy import double, complex, floor, angle
from sage.all import infinity, QQbar, Graphics, scatter_plot
from sage.functions.other import real_part, imag_part

from abelfunctions.complex_path import (
    ComplexLine,
    ComplexArc,
    ComplexPath,
)

class ComplexPathFactory(object):
    r"""Factory for computing complex paths on the x-projection of a Riemann surface
    determined by an algebraic curve :math:`C : f(x,y) = 0`.

    Since paths on a Riemann surface are computed via analytic continuation
    care needs to be taken when the x-part of the path gets close to a
    discriminant point of the algebraic curve from which the Riemann surface is
    derived. This is because some of the y-sheets of the curve, when considered
    as a covering of the complex x-plane, coalesce at the discriminant points.
    Therefore, "bounding circles" need to be computed at each discriminant
    point.

    Attributes
    ----------
    riemann_surface : RiemannSurface
        The Riemann surface on which to construct the x-paths.
    base_point : complex
        If a base point isn't provided, one will be chosen.
    kappa : double (default: 3/5)
        A scaling factor between 0.5 and 1.0 used to modify the radius
        of the bounding circles.

    Methods
    -------
    .. autosummary::

      discriminant_points
      complex_path_to_discriminant_point
      complex_path_circle_discriminant_point
      complex_path_monodromy_path
      complex_path_to_point
      complex_path_reverse
      show_paths

    """
    @property
    def base_point(self):
        return self._base_point

    @property
    def discriminant_points(self):
        return self._discriminant_points

    @property
    def radii(self):
        return self._radii

    def __init__(self, f, base_point=None, kappa=3./5.):
        """Initialize a complex path factory.

        Complex path factories require a base point from which most complex
        paths begin on a Riemann surface. In particular, this base point is
        used as the base point in constructing the monodromy group of the
        Riemann surface.

        Parameters
        ----------
        f : polynomial
            The plane algebraic curve defining the Riemann surface.
        base_point : complex
            The base point of the factory and of the monodromy group of the
            Riemann surface. If not provided one will be chosen based on the
            discriminant point placement.
        kappa : double
            A scaling factor used to determine the radii of the "bounding
            circles" around each discriminant point. `kappa = 1.0` means the
            bounding circles are made as large as possible, resulting in
            possibly touching circles between two or more discriminant points.

        """
        self.f = f

        # compute the discriminant points and determine a base point if none
        # was provided
        b,d = self._compute_discriminant_points(base_point)
        self._base_point = b
        self._discriminant_points = d

        # compute the bounding circle radii from the discriminant points
        r = self._compute_radii(kappa)
        self._radii = r

    def _compute_discriminant_points(self, base_point):
        r"""Computes and stores the discriminant points of the underlying curve.

        A discriminant point :math:`x=b` is an x-point where at least one
        y-root lying above has multiplicity greater than one. A
        :class:`PuiseuxTSeries` is required to represent a place on the Riemann
        surface whose x-projection is a discriminant point. These kinds of
        places are of type :class:`DiscriminantPlace`.

        .. note::

            The ordering of the discriminant points is important for the
            purposes of computing the monodromy group, which is done in the
            :class:`RiemannSurfacePathFactory` attribute, `PathFactory`.

        Parameters
        ----------
        None

        Returns
        -------
        list : complex
            Return a list of ordered discriminant points from the base point.

        """
        # compute the symbolic and numerical discriminant points
        f = self.f
        x,y = f.parent().gens()
        res = f.resultant(f.derivative(y), y).univariate_polynomial()
        rts = res.roots(ring=QQbar, multiplicities=False)
        discriminant_points = rts

        # determine a base_point, if not specified
        if not base_point:
            a = min(real_part(bi) for bi in discriminant_points)
            a = a - 1
            aint = complex(floor(a))
            base_point = aint

        # sort the discriminant points first by argument with the base point
        # and then by distance from the base point.
        discriminant_points = numpy.array(discriminant_points, dtype=complex)
        centered_points = discriminant_points - base_point
        distances = abs(centered_points)
        arguments = angle(centered_points)
        sort_index = numpy.lexsort((distances, arguments))

        # sort and return
        discriminant_points = discriminant_points[sort_index]
        return base_point, discriminant_points

    def closest_discriminant_point(self, x):
        r"""Returns the closest discriminant point to a point x.

        An often-used helper function by several components of
        :class:`RiemannSurface`.

        Parameters
        ----------
        x : complex
            A complex x-point.
        exact : boolean
            If `True`, returns a `sympy.Expr` representing the discriminant
            point exactly. Otherwise, returns a numerical approximation.

        Returns
        -------
        complex or sympy.Expr
            The discriminant point, either exact or numerical.
        """
        # for performance, coerce everything to floating point approximations.
        # if the discriminant points are less than 1e-16 apart then we're
        # screwed, anyway.
        b = numpy.array(self.discriminant_points)
        x = complex(x)
        idx = numpy.argmin(abs(b - x))
        return b[idx]

    def _compute_radii(self, kappa):
        """Returns the radii of the bounding circles.

        Parameters
        ----------
        kappa : double
            A scaling factor between 0.5 and 1.0. `kappa = 1.0` means that the
            bounding circles are taken to be as large as possible without
            overlapping.

        Returns
        -------
        radii : array
            An ordered list of radii. The radius at index `k` is associated
            with the discriminant point at index `k` in
            `self.discriminant_points`.
        """
        # special case when there is only one finite discriminant point: take
        # the distance from the base point to the discriminant point (scaled by
        # kappa, of course)
        if len(self.discriminant_points) == 1:
            b = self.discriminant_points[0]
            radius = numpy.abs(self.base_point - b)
            radius *= kappa/2.0
            radii = numpy.array([radius], dtype=double)
            return radii

        # when there is more than one discriminant point we scale disctances
        # accordingly
        radii = []
        for bi in self.discriminant_points:
            dists = [abs(bi - bj) for bj in self.discriminant_points
                     if bi != bj]
            rho = min(dists)
            radius = rho*kappa/2.0
            radii.append(radius)
        radii = numpy.array(radii, dtype=double)

        # final check: assert that the base point is sufficiently far away from
        # the discriminant points
        dists = [abs(bi - self.base_point) for bi in self.discriminant_points]
        dists = numpy.array(dists, dtype=double) - radii
        if any(dists < 0):
            raise ValueError('Base point lies in the bounding circles of the '
                             'discriminant points. Use different base point or '
                             'circle scaling factor kappa.')

        return radii

    def radius(self, bi):
        """Returns the raidus of the bounding circle around `bi`.

        Parameters
        ----------
        bi : complex
            A discriminant point of the algebraic curve.

        Returns
        -------
        radius : double
            The radius of the bounding circle.
        """
        # find the index where bi appears in the list of discriminant points.
        # it's done numerically in case a numerical approximation bi is given
        index = 0
        for z in self.discriminant_points:
            if abs(z-bi) < 1e-15:
                break
            index += 1

        # raise an error if not found
        if index == len(self.discriminant_points):
            raise ValueError('%s is not a discriminant point of %s'%(bi,f))

        radius = self.radii[index]
        return radius

    def complex_path_to_discriminant_point(self, bi):
        """Returns the complex path leading to the discriminant point `bi`.

        This path is from the base point :attribute:`base_point` to a point `z`
        on the bounding circle around `bi` which lies on the straight line path
        from the base point to the discriminant point.

        Parameters
        ----------
        bi : complex
            A discriminant point of the curve.

        Returns
        -------
        path : ComplexPath
            A path from the base point to a point on the bounding circle around
            `bi`.

        """
        # determine the point `z` on the bounding circle lying on the straight
        # line path from the base point to the discriminant point
        #
        # l(s) = base_point  + (b_i - base_point)*s, s \in [0,1]
        #
        # Solve: |l(s) - b_i| = Ri
        #
        # Solution:
        #             1
        # s = 1 - ---------
        #         |a - b_i|
        #
        Ri = self.radius(bi)
        l = lambda s: self.base_point + (bi - self.base_point)*s
        s = 1.0 - Ri/numpy.abs(self.base_point - bi)
        z = l(s)
        path = self.complex_path_build_avoiding_path(self.base_point, z)
        return path

    def complex_path_around_discriminant_point(self, bi, nrots=1):
        """Returns the complex path consisting of `nrots` rotations around the bounding
        circle of discriminant point `bi`.

        The sign of `nrots` indicates the sign of the direction.

        Parameters
        ----------
        bi : complex
            A discriminant point of the curve.
        nrots : integer (default `1`)
            A number of rotations around this discriminant point.
        starting_point : complex
            A point on the bounding circle 

        Returns
        -------
        path : ComplexPath
            A path representing circles starting to the left of the
            discriminant point `bi` and encircling it `nrots` times.

        """
        Ri = self.radius(bi)
        z = self.complex_path_to_discriminant_point(bi)(1.0)
        theta = numpy.angle(z - bi)
        dtheta = numpy.pi if nrots > 0 else -numpy.pi
        circle = ComplexArc(Ri, bi, theta, dtheta) + \
                 ComplexArc(Ri, bi, theta + dtheta, dtheta)

        # rotate |nrots| times
        path = circle
        for _ in range(abs(nrots)-1):
            path += circle
        return path

    def complex_path_to_point(self, x):
        """Returns a complex path from the base point to `x`.
 
        Parameters
        ----------
        x : complex
            A point on the complex x-sphere.

        Returns
        -------
        path : ComplexPath
            A path

        """
        return self.complex_path_build_avoiding_path(self.base_point, x)

    def complex_path_build_avoiding_path(self, z0, z1):
        """Returns a complex path to `z1` from `z0` avoiding discriminant points as
        necessary.

        Parameters
        ----------
        z0 : complex
            The starting point of the path.
        z1 : complex
            The ending point of the path.

        Returns
        -------
        path : ComplexPath
            A :class:`ComplexPath` from `z0` to `z1` bounded away from the
            discriminant points.

        """
        # compute the set of discriminant points whose bounding circle
        # intersects the line from a to z
        z0 = complex(z0)
        z1 = complex(z1)
        segments = []
        b = numpy.array([bi for bi in self.discriminant_points
                         if self._intersects_discriminant_point(z0,z1,bi)],
                        dtype=complex)

        # sort by increasing distance from z0
        b = b.tolist()
        b.sort(key=lambda bi: numpy.abs(bi-z0))
        for bi in b:
            # compute the intersection points of the segment from z0 to
            # z1 with the circle around bi.
            w0, w1 = self._intersection_points(z0,z1,bi)

            # compute the arc going from w0 to w1 avoiding the bounding
            # circle around bi.
            arc = self._avoiding_arc(w0,w1,bi)

            # add to the path and update the loop
            segments.append(ComplexLine(z0,w0))
            segments.append(arc)
            z0 = w1

        # add the final line segment and return
        segments.append(ComplexLine(z0,z1))
        if len(segments) == 1:
            path = segments[0]
        else:
            path = ComplexPath(*segments)
        return path


    def complex_path_monodromy_path(self, bi, nrots=1):
        """Returns the complex path starting from the base point, going around the
        discriminant point `bi` `nrots` times, and returning to the base
        x-point.

        The sign of `nrots` indicates the sign of the direction.

        Parameters
        ----------
        bi : complex
            A discriminant point.
        nrots : integer (default `1`)
            A number of rotations around this discriminant point.

        Returns
        -------
        path : ComplexPath
            A complex path representing the monodromy path with `nrots`
            rotations about the discriminant point `bi`.

        """
        # special case when going around infinity.
        if bi == infinity:
            return self.complex_path_around_infinity(nrots=nrots)

        path_to_bi = self.complex_path_to_discriminant_point(bi)
        path_around_bi = self.complex_path_around_discriminant_point(
            bi, nrots=nrots)
        path_from_bi = path_to_bi.reverse()
        path = path_to_bi + path_around_bi + path_from_bi
        return path

    def complex_path_around_infinity(self, nrots=1):
        """Returns the complex path starting at the base point, going around
        infinity `nrots` times, and returning to the base point.

        This path is sure to not only encircle all of the discriminant
        points but also stay sufficiently outside the bounding circles
        of the points.

        Parameters
        ----------
        nrots : integer, (default `1`)
            The number of rotations around infinity.

        Returns
        -------
        RiemannSurfacePath
            The complex path encircling infinity.

        """
        path = []

        # determine the radius R of the circle, centered at the origin,
        # encircling all of the discriminant points and the bounding circles
        b = self.discriminant_points
        R = numpy.abs(self.base_point)
        for bi in b:
            radius = self.radius(bi)
            Ri = numpy.abs(bi) + 2*radius  # to be safely away
            R = Ri if Ri > R else R

        # the path begins with a line starting the base point and ending
        # at -R.
        path = ComplexLine(self.base_point, -R)

        # the positive direction around infinity is equal to the
        # negative direction around the origin
        dtheta = -numpy.pi if nrots > 0 else numpy.pi
        for _ in range(abs(nrots)):
            path += ComplexArc(R, 0, numpy.pi, dtheta)
            path += ComplexArc(R, 0, 0, dtheta)

        # return to the base point
        path += ComplexLine(-R, self.base_point)
        return path

    def complex_path_to_point(self, x):
        """Returns a complex path to an arbitrary point `x`.

        Parameters
        ----------
        x : complex
            A point on the complex x-sphere.

        """
        raise NotImplementedError('Implement in subclass.')

    def show_paths(self, *args, **kwds):
        """Plots all of the monodromy paths of the curve.

        Parameters
        ----------
        ax : matplotlib.Axes
            The figure axes on which to plot the paths.

        Returns
        -------
        None
        """
        # fill the bounding circles around each discriminant point
        a = self.base_point
        b = self.discriminant_points

        # plot the base point and the discriminant points
        pts = [(real_part(a), imag_part(a))]
        plt = scatter_plot(pts, facecolor='red', **kwds)
        pts = zip(b.real, b.imag)
        plt += scatter_plot(pts, facecolor='black', **kwds)

        # plot the monodromy paths
        for bi in b:
            path = self.complex_path_monodromy_path(bi)
            plt += path.plot(**kwds)
        return plt

    def _intersection_points(self, z0, z1, bi):
        """Returns the complex points `w0,w1` where the line from `z0` to `z1`
        intersects the bounding circle around `bi`.

        Parameters
        ----------
        z0 : complex
            Line starting point.
        z1 : complex
            Line ending point.
        bi : complex
            A discriminant point.

        Returns
        -------
        w0, w1 : complex
            Points on the bounding circle of `bi` where the line z0-z1
            intersects.

        """
        # construct the polynomial giving the distance from the line l(t),
        # parameterized by t in [0,1], to bi.
        z0 = complex(z0)
        z1 = complex(z1)
        bi = complex(bi)
        Ri = self.radius(bi)
        v = z1 - z0
        w = z0 - bi
        p2 = v.real**2 + v.imag**2
        p1 = 2*(v.real*w.real + v.imag*w.imag)
        p0 = w.real**2 + w.imag**2 - Ri**2  # solving |l(t) - bi| = Ri

        # find the roots of this polynomial and sort by increasing t
        p = numpy.poly1d([p2, p1, p0])
        t = numpy.roots(p)
        t.sort()

        # compute ordered intersection points
        w0 = v*t[0] + z0   # first intersection point
        w1 = v*t[1] + z0   # second intersection point
        return w0, w1

    def _avoiding_arc(self, w0, w1, bi):
        """Returns the arc `(radius, center, starting_theta, dtheta)`, from the points
        `w0` and `w1` on the bounding circle around `bi`.

        The arc is constructed in such a way so that the monodromy properties
        of the path are conserved.

        Parameters
        ----------
        w0 : complex
            The starting point of the arc on the bounding circle of `bi`.
        w1 : complex
            The ending point of the arc on the bounding circle of `bi`.
        bi : complex
            The discriminant point to avoid.

        Returns
        -------
        arc : ComplexArc
            An arc from `w0` to `w1` around `bi`.

        """
        # the angles of w0 and w1 around the circle tells us the length of the
        # arc connecting the points
        theta0 = numpy.angle(w0 - bi)
        theta1 = numpy.angle(w1 - bi)

        # the angles that w1 and bi make with w0 tell us whether the path will
        # go above bi or below bi
        phi1 = numpy.angle(w1 - w0)
        phii = numpy.angle(bi - w0)

        # go above (counterclockwise) if w1 lies over bi
        direction = numpy.sign(phii - phi1)
        if (theta0 < 0) and (theta1 > 0):
            theta0 += 2*numpy.pi
        if (theta0 > 0) and (theta1 < 0):
            theta1 += 2*numpy.pi
        dtheta = min(abs(theta0 - theta1), abs(theta1 - theta0))
        dtheta = direction*dtheta

        # degenerate case when the path is colinear with bi: ALWAYS GO ABOVE
        # bi!!! this implies that the furthest colinear discriminant points are
        # "above" those that are closer.
        if dtheta == 0:
            dtheta = -numpy.pi

        # add the path from z0 to w1 going around bi
        Ri = self.radius(bi)
        arc = ComplexArc(Ri, bi, theta0, dtheta)
        return arc

    def _intersects_discriminant_point(self, z0, z1, bi):
        """Returns `True` if the line from `z0` to `z1` intersects the bounding circle
        around the discriminant point `bi`.

        Parameters
        ----------
        z0 : complex
            Line starting point.
        z1 : complex
            Line ending point.
        bi : complex
            A discriminant point.

        Returns
        -------
        is_intersecting : bool
            `True` if the line from `z0` to `z1` gets too close to `bi`.
        """
        # first check the perpendicular distance from bi to the line
        # passing through z0 and z1
        z0 = complex(z0)
        z1 = complex(z1)
        bi = complex(bi)
        direction = numpy.sign(angle(z1-z0) - angle(bi-z0))
        normv = numpy.abs(z1-z0)
        v = 1.0j*direction*(z1 - z0)
        r = z0 - bi

        # degenerate case: the line through z0 and z1 crosses bi. in this case
        # just check if the branch point lies in between
        if direction == 0:
            if (abs(bi - z0) <= normv) and (abs(bi - z1) <= normv):
                return True
            else:
                return False

        # return False if the distance from the _line_ passing through
        # z0 and z1 to bi is greater than the radius fo teh bounding
        # circle.
        distance = (v.real*r.real + v.imag*r.imag)
        distance = distance / normv
        if distance > self.radius(bi):
            return False

        # also need to check if bi "lies between" the _line segment_
        # between z0 and z1. use the distance vector w = d*v/|v|. the
        # distance from vtilde to z0 and z1 should be less that the
        # distance between z0 and z1
        w = distance*v/normv + bi
        if (abs(w - z0) <= normv) and (abs(w - z1) <= normv):
            return True
        return False
