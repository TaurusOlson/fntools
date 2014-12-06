=======
fntools
=======


.. image:: https://readthedocs.org/projects/fntools/badge/?version=master
:target: https://readthedocs.org/projects/fntools/?badge=master
:alt: Documentation Status

**fntools** provides functional programming tools for data processing. This
module is a set of functions that I needed in my work and found useful.


Installation
------------

::

    pip install fntools


Examples
--------

* Split a list of elements with factors with `split`::

    songs = ('Black', 'Even Flow', 'Amongst the waves', 'Sirens')
    albums = ('Ten', 'Ten', 'Backspacer', 'Lightning Bolt')
    print split(songs, albums)
    {'Lightning Bolt': ['Sirens'], 'Ten': ['Black', 'Even Flow'], 'Backspacer': ['Amongst the waves']}


* Determine whether any element of a list is included in another list with `any_in`::

    print any_in(['Oceans', 'Big Wave'], ['Once', 'Alive', 'Oceans', 'Release'])
    True

    print any_in(['Better Man'], ['Man of the Hour', 'Thumbing my way'])
    False


* Apply many functions on the data with `dispatch`::

    # Suppose we want to know the mean, the standard deviation and the median of
    # a distribution (here we use the standard normal distribution)

    import numpy as np
    np.random.seed(10)
    x = np.random.randn(10000)

    print dispatch(x, (np.mean, np.std, np.median))
    [0.0051020560019149385, 0.98966401277169491, 0.013111308495186252]


Many more useful functions are available. For more details, go to the
documentation_.


Inspirations
------------

* The excellent toolz_ by `Matthew Rocklin`_
* `A pratical introduction to functional programming`_ by `Mary Rose Cook`_
* A bit of `R`_ (multimap, use, use_with)


.. _documentation: http://fntools.readthedocs.org/en/latest
.. _toolz: https://github.com/mrocklin/toolz
.. _`A pratical introduction to functional programming`: http://maryrosecook.com/blog/post/a-practical-introduction-to-functional-programming
.. _`Matthew Rocklin`: https://github.com/mrocklin
.. _`Mary Rose Cook`: https://github.com/maryrosecook
.. _R: http://www.r-project.org

