"""
fntools: functional programming tools for data processing
=========================================================

"""

from .fntools import use_with, zip_with, unzip, concat, mapcat, dmap, rmap, replace,\
        compose, groupby, reductions, split, assoc, dispatch, multimap,\
        multistarmap, pipe, pipe_each, shift, repeatedly, update, duplicates, pluck, use, get_in,\
        valueof, take, drop, find, remove, isiterable, are_in, any_in,\
        all_in, monotony, attributes, find_each, dfilter, occurrences,\
        indexof, indexesof, count, isdistinct, nrow, ncol, names


__version__ = '1.1.1'
__title__ = 'fntools'
__author__ = 'Taurus Olson'
__license__ = 'MIT'
