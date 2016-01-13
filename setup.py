#!/usr/bin/env python
"""
Distutils based setup script for abelfunctions

This uses Distutils (http://python.org/sigs/distutils-sig/) the standard
python mechanism for installing packages. For the easiest installation
just type the command (you'll probably need root privileges for that):

    python setup.py install

This will install the library in the default location. To install in a
custom directory <dir>, use:

    python setup.py install --prefix=<dir>

To install for your user account (recommended) use:

    python setup.py install --user

"""
import numpy

import os
import sys
import shutil
import unittest
import build_system

from distutils.core import Command

SAGE_ROOT = os.environ['SAGE_ROOT']
SAGE_LOCAL = os.environ['SAGE_LOCAL']

INCLUDES = [os.path.join(SAGE_ROOT,'devel','sage','sage','ext'),
            os.path.join(SAGE_ROOT,'devel','sage'),
            os.path.join(SAGE_ROOT,'devel','sage','sage','gsl'),
            os.path.join(SAGE_LOCAL,'include','csage'),
            os.path.join(SAGE_LOCAL,'include'),
            os.path.join(SAGE_LOCAL,'include','python')]
INCLUDES_NUMPY = [os.path.join(SAGE_LOCAL,'lib','python','site-packages',
                               'numpy','core','include')]


class clean(Command):
    """Cleans files so you should get the same copy as in git."""
    description = 'remove build files'
    user_options = [('all', 'a', 'the same')]

    def initialize_options(self):
        self.all = None

    def finalize_options(self):
        pass

    def run(self):
        # delete all files ending with certain extensions
        # currently: '.pyc', '~'
        dir_setup = os.path.dirname(os.path.realpath(__file__))
        curr_dir = os.getcwd()
        for root, dirs, files in os.walk(dir_setup):
            for file in files:
                file = os.path.join(root, file)
                if file.endswith('.pyc') and os.path.isfile(file):
                    os.remove(file)
                if file.endswith('~') and os.path.isfile(file):
                    os.remove(file)

        os.chdir(dir_setup)

        # explicity remove files and directories from 'blacklist'
        blacklist = ['build', 'dist', 'doc/_build']
        for file in blacklist:
            if os.path.isfile(file):
                os.remove(file)
            elif os.path.isdir(file):
                shutil.rmtree(file)

        os.chdir(dir_setup)

        # delete temporary cython .c files. be careful to only delete the .c
        # files corresponding to .pyx files. (keep other .c files)
        ext_sources = [f for ext in ext_modules for f in ext.sources]
        for file in ext_sources:
            file = os.path.join(dir_setup, file)
            if file.endswith('.pyx') and os.path.isfile(file):
                (root, ext) = os.path.splitext(file)
                file_c = root + '.c'
                if os.path.isfile(file_c):
                    os.remove(file_c)

        os.chdir(dir_setup)

        # delete cython .so modules
        ext_module_names = [ext.name for ext in ext_modules]
        for mod in ext_module_names:
            file = mod.replace('.', os.path.sep) + '.so'
            file = os.path.join(dir_setup, file)
            if os.path.isfile(file):
                os.remove(file)

        os.chdir(curr_dir)


class test_abelfunctions(Command):
    """Runs all tests under every abelfunctions/ directory.

    All Cython modules must be built in-place for testing to work.
    """
    description = "run all tests and doctests"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        loader = unittest.TestLoader()
        suite = loader.discover('abelfunctions')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        errno = not result.wasSuccessful()
        sys.exit(errno)


if '-ba' in sys.argv:
    print "Rebuilding all Cython extensions."
    sys.argv.remove('-ba')
    FORCE = True
else:
    FORCE = False

def Extension(*args, **kwds):
    if not kwds.has_key('include_dirs'):
        kwds['include_dirs'] = INCLUDES
    else:
        kwds['include_dirs'] += INCLUDES
    if not kwds.has_key('force'):
        kwds['force'] = FORCE

    # Disable warnings when running GCC step -- cython has already parsed the code and
    # generated any warnings; the GCC ones are noise.
    if not kwds.has_key('extra_compile_args'):
        kwds['extra_compile_args'] = ['-w']
    else:
        kwds['extra_compile_args'].append('-w')

    E = build_system.Extension(*args, **kwds)
    E.libraries = ['csage'] + E.libraries
    return E



packages = [
#    'abelfunctions.riemann_theta',
#    'abelfunctions.utilities',
    ]

ext_modules = [
    # Extension('abelfunctions.riemann_surface',
    #           sources=[os.path.join('abelfunctions','riemann_surface.pyx')]
    #       ),
    # Extension('abelfunctions.riemann_surface_path',
    #           sources=[os.path.join('abelfunctions','riemann_surface_path.pyx')]
    #       ),
    # Extension('abelfunctions.analytic_continuation',
    #           sources=[os.path.join('abelfunctions',
    #                                 'analytic_continuation.pyx')]
    #       ),
    # Extension('abelfunctions.analytic_continuation_smale',
    #           sources=[os.path.join('abelfunctions',
    #                                 'analytic_continuation_smale.pyx')]
    #       ),
    # Extension('abelfunctions.polynomials',
    #           sources=[os.path.join('abelfunctions','polynomials.pyx')]
    #       ),
    # Extension('abelfunctions.differentials',
    #           sources=[os.path.join('abelfunctions','differentials.pyx')]
    #       ),
    Extension('abelfunctions.puiseux_series_ring_element',
              sources=[os.path.join('abelfunctions','puiseux_series_ring_element.pyx')],
              language='c++',
              extra_compile_args=['-std=c99'],
          ),
    # Extension('abelfunctions.riemann_theta.radius',
    #           sources=[os.path.join('abelfunctions','riemann_theta',
    #                                 'lll_reduce.c'),
    #                    os.path.join('abelfunctions','riemann_theta',
    #                                 'radius.pyx')]
    #       ),
    # Extension('abelfunctions.riemann_theta.integer_points',
    #           sources=[os.path.join('abelfunctions','riemann_theta',
    #                                 'integer_points.pyx')]
    #       ),
    # Extension('abelfunctions.riemann_theta.riemann_theta',
    #           sources=[os.path.join('abelfunctions','riemann_theta',
    #                                 'finite_sum.c'),
    #                    os.path.join('abelfunctions','riemann_theta',
    #                                 'riemann_theta.pyx')]
    #       ),
    ]

# parameters for all extension modules:
#
# * use all include directories in INCLUDES
# * disable warnings in gcc step
for mod in ext_modules:
    mod.include_dirs.extend(INCLUDES)
    mod.include_dirs.extend(INCLUDES_NUMPY)
    mod.extra_compile_args.append('-w')

tests = [
    'abelfunctions.tests',
#    'abelfunctions.riemanntheta.tests',
    ]

exec(open('abelfunctions/version.py').read())

build_system.cythonize(ext_modules)
build_system.setup(
    name = 'abelfunctions',
    version = __version__,
    description = 'A library for computing with Abelian functions, Riemann '
                  'surfaces, and algebraic curves.',
    author = 'Chris Swierczewski',
    author_email = 'cswiercz@gmail.com',
    url = 'https://github.com/cswiercz/abelfunctions',
    license = 'GPL v2+',
    packages = ['abelfunctions'],
    ext_modules = ext_modules,
    platforms = ['Linux', 'Unix', 'Mac OS-X'],
)
