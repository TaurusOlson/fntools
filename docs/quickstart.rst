.. _quickstart:

Quickstart
==========

.. toctree::
   :maxdepth: 3


use_with
--------

::

    >>> from collections import namedtuple
    >>> from functools import partial
    >>> cols = ('sepal_width', 'sepal_length', 'petal_length', 'petal_width', 'species')
    >>> Table = namedtuple('Table', cols)
    >>> iris = Table(sepal_length=(5.1, 4.9, 4.7),
                     sepal_width=(3.5, 3.0, 3.2),
                     petal_length=(1.4, 1.4, 1.3),
                     petal_width=(0.2, 0.2, 0.2),
                     species=("setosa", "setosa", "setosa"))
    >>> mean = lambda xs: sum(xs) / len(xs)
    >>> calc_mean = partial(use_with, iris, mean)
    >>> map(calc_mean, ("sepal_width", "petal_width"))
    
