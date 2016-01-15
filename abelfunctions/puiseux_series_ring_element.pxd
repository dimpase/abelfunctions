from sage.structure.element cimport AlgebraElement
from sage.rings.laurent_series_ring_element cimport LaurentSeries

cdef class PuiseuxSeries(AlgebraElement):
     cpdef LaurentSeries __l
     cdef long __e
