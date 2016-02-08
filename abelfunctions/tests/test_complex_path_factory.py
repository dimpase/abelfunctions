import unittest

import numpy
from numpy import pi, Infinity, exp, sqrt, complex
from abelfunctions.complex_path import (
    ComplexPathPrimitive,
    ComplexPath,
    ComplexLine,
    ComplexArc,
    ComplexRay,
)
from abelfunctions.complex_path_factory import ComplexPathFactory
from abelfunctions.riemann_surface import RiemannSurface
from sage.all import QQ, QQbar, I

class TestConstruction(unittest.TestCase):
    def setUp(self):
        R = QQ['x,y']
        x,y = R.gens()

        f1 = y**2 - x
        self.f1 = f1

        f2 = y**2 - (x**2 + 1)
        self.f2 = f2

        f3 = y**2 - (x**4 - 1)
        self.f3 = f3

    def test_discriminant_points(self):
        # these should be in the proper order
        f = self.f1
        CPF = ComplexPathFactory(f, -1)
        discriminant_points = CPF.discriminant_points
        self.assertItemsEqual(discriminant_points, [QQbar(0)])

        f = self.f2
        CPF = ComplexPathFactory(f, -2)
        discriminant_points = CPF.discriminant_points
        self.assertItemsEqual(discriminant_points, map(QQbar, [-I, I]))

        f = self.f3
        CPF = ComplexPathFactory(f, -2)
        discriminant_points = CPF.discriminant_points
        self.assertItemsEqual(discriminant_points,
                              map(QQbar,[-I, -1, 1, I]))

    def test_base_point(self):
        f = self.f1
        CPF = ComplexPathFactory(f, -1)
        self.assertAlmostEqual(CPF.base_point, -1)

        CPF = ComplexPathFactory(f, -2)
        self.assertAlmostEqual(CPF.base_point, -2)

    def test_kappa(self):
        f = self.f1
        CPF = ComplexPathFactory(f, -1, kappa=1.0)
        self.assertAlmostEqual(CPF.radius(0), 0.5)

        CPF = ComplexPathFactory(f, -1, kappa=0.5)
        self.assertAlmostEqual(CPF.radius(0), 0.25)

        # two disc points \pm I: distance between them = 2
        f = self.f2
        CPF = ComplexPathFactory(f, -1, kappa=1.0)
        self.assertAlmostEqual(CPF.radius(-1.j), 1.0)
        self.assertAlmostEqual(CPF.radius(1.j), 1.0)

        CPF = ComplexPathFactory(f, -1, kappa=0.5)
        self.assertAlmostEqual(CPF.radius(-1.j), 0.5)
        self.assertAlmostEqual(CPF.radius(1.j), 0.5)

        # four disc points at 4th roots of unity: distance between them =
        # sqrt(2)
        f = self.f3
        CPF = ComplexPathFactory(f, -2, kappa=1.0)
        self.assertAlmostEqual(CPF.radius(-1.j), sqrt(2.0)/2)
        self.assertAlmostEqual(CPF.radius(-1), sqrt(2.0)/2)
        self.assertAlmostEqual(CPF.radius(1), sqrt(2.0)/2)
        self.assertAlmostEqual(CPF.radius(1.j), sqrt(2.0)/2)

        CPF = ComplexPathFactory(f, -2, kappa=0.5)
        self.assertAlmostEqual(CPF.radius(-1.j), sqrt(2.0)/4)
        self.assertAlmostEqual(CPF.radius(-1), sqrt(2.0)/4)
        self.assertAlmostEqual(CPF.radius(1), sqrt(2.0)/4)
        self.assertAlmostEqual(CPF.radius(1.j), sqrt(2.0)/4)

    def test_kappa_error(self):
        # tests that an error is raised if the base point lies in a bounding
        # circle of a discriminant poin
        f = self.f2
        with self.assertRaises(ValueError):
            CPF = ComplexPathFactory(f, 0.5j, kappa=1.0)

        f = self.f3
        with self.assertRaises(ValueError):
            CPF = ComplexPathFactory(f, -1, kappa=1.0)
        with self.assertRaises(ValueError):
            CPF = ComplexPathFactory(f, -1.5, kappa=1.0)
        CPF = ComplexPathFactory(f, -2, kappa=1.0)

        with self.assertRaises(ValueError):
            CPF = ComplexPathFactory(f, -1, kappa=0.5)
        with self.assertRaises(ValueError):
            CPF = ComplexPathFactory(f, -1.25, kappa=0.5)
        CPF = ComplexPathFactory(f, -1.5, kappa=0.5)

    def test_complex_path_to_discriminant_point(self):
        #
        f = self.f1
        CPF = ComplexPathFactory(f, -2, kappa=1.0)
        path = CPF.complex_path_to_discriminant_point(0)
        self.assertEqual(path, ComplexLine(-2,-1))

        #
        f = self.f2
        CPF = ComplexPathFactory(f, -1, kappa=1.0)
        path = CPF.complex_path_to_discriminant_point(-1.j)

        z = -sqrt(2.0)/2.0 - (1-sqrt(2.0)/2.0)*1.j
        w = path(1.0)
        self.assertAlmostEqual(z,w)

        path = CPF.complex_path_to_discriminant_point(1.j)
        z = z.conjugate()
        w = path(1.0)
        self.assertAlmostEqual(z,w)

        # distance between 4th roots of unity = sqrt(2)
        # kappa = 0.5 ==> radius = sqrt(2)/4.0
        f = self.f3
        CPF = ComplexPathFactory(f, -2, kappa=0.5)
        path = CPF.complex_path_to_discriminant_point(-1.j)

        alpha = 1.0-sqrt(10.0)/20.0
        z = (2*alpha-2) - alpha*1.0j
        w = path(1.0)
        self.assertAlmostEqual(z,w)

        path = CPF.complex_path_to_discriminant_point(1.j)
        z = z.conjugate()
        w = path(1.0)
        self.assertAlmostEqual(z,w)

    def test_complex_path_to_discriminant_point_abel_ordering(self):
        # the way abel complex paths are constructed ensure that all points
        # along the path to bi lie above all discriminant points below bi and
        # above all discriminant points below bi. this will test that scenario
        #
        # note that this particular test doesn't really work for paths to
        # co-linear points
        f = self.f3
        CPF = ComplexPathFactory(f, -2, kappa=0.5)

        # 1.j lies above all other discriminant points. make sure the points
        # along the path all lie above the other discriminant point
        #
        # ignore the base point in these tests
        path = CPF.complex_path_to_discriminant_point(1.j)
        s = numpy.linspace(0,1,32)
        path_points = path(s)
        centered_path_points = path_points - CPF.base_point

        centered_b0 = complex(CPF.discriminant_points[0] - CPF.base_point)
        b0_angles = numpy.angle(centered_path_points - centered_b0)
        self.assertTrue(all(b0_angles[1:] >= 0))

        centered_b1 = complex(CPF.discriminant_points[1] - CPF.base_point)
        b1_angles = numpy.angle(centered_path_points - centered_b1)
        self.assertTrue(all(b1_angles[1:] >= 0))

        centered_b2 = complex(CPF.discriminant_points[2] - CPF.base_point)
        b2_angles = numpy.angle(centered_path_points - centered_b2)
        self.assertTrue(all(b2_angles[1:] >= 0))

        # -1.j lies below all other discriminant points. make sure the points
        # along the path all lie below the other discriminant point
        path = CPF.complex_path_to_discriminant_point(-1.j)
        s = numpy.linspace(0,1,32)
        path_points = path(s)
        centered_path_points = path_points - CPF.base_point

        b1_angles = numpy.angle(centered_path_points - centered_b1)
        self.assertTrue(all(b1_angles[1:] <= 0))

        b2_angles = numpy.angle(centered_path_points - centered_b2)
        self.assertTrue(all(b2_angles[1:] <= 0))

        centered_b3 = complex(CPF.discriminant_points[3] - CPF.base_point)
        b3_angles = numpy.angle(centered_path_points - centered_b3)
        self.assertTrue(all(b3_angles[1:] <= 0))


    def test_ordering(self):
        # issue fff: we need to make sure that paths preserve discriminant
        # point ordering. in an earlier version of the code paths were
        # constructed to hit a point to the left of a discriminant point.
        # however, this can mess up the monodromy ordering as indicated in the
        # following picture: the dotted line is the straight line path to
        # discriminant point b, the dashed line is the straight line path to a
        # point to the left of b. note that these two lines imply conjugation
        # by discriminant point a.
        #
        #
        #               * b-R          *  b
        #              /             .
        #             /            .
        #            /           .
        #           /          .
        #          /    *    .
        #         /     a  .
        #        /       .
        #       /      .
        #      /     .
        #     /    .
        #    /   .
        #   /  .
        #  / .
        # .
        #
        pass


    def test_closest_discriminant_point(self):
        f = self.f3
        CPF = ComplexPathFactory(f, -2, kappa=1.0)

        # [-I, -1, 1, I]
        discriminant_points = CPF.discriminant_points
        b = CPF.closest_discriminant_point(-2)
        self.assertEqual(b, discriminant_points[1])

        b = CPF.closest_discriminant_point(-2j)
        self.assertEqual(b, discriminant_points[0])

        b = CPF.closest_discriminant_point(-2 + 0.5j)
        self.assertEqual(b, discriminant_points[1])

        b = CPF.closest_discriminant_point(-2 + 0.5j)
        self.assertEqual(b, discriminant_points[1])

        b = CPF.closest_discriminant_point(-2 + 0.5j)
        self.assertEqual(b, discriminant_points[1])

        b = CPF.closest_discriminant_point(-2 + 1.9j)
        self.assertEqual(b, discriminant_points[1])

        b = CPF.closest_discriminant_point(-2 + 2.1j)
        self.assertEqual(b, discriminant_points[3])

    def test_complex_path_around_infinity(self):
        f = self.f3
        CPF = ComplexPathFactory(f, -3, kappa=0.5)
        gamma = CPF.complex_path_around_infinity()
        self.assertAlmostEqual(gamma(0), CPF.base_point)
        self.assertAlmostEqual(gamma(1.0), CPF.base_point)

