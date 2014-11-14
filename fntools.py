"""Functional programming tools for data processing

`fntools` is a simple library providing the user with functional programming
functions to transform, filter, create and inspect Python data structures.

"""


from copy import deepcopy
import itertools
import operator
import collections
from functools import wraps


# TRANSFORMATION {{{1

def use_with(data, fn, *attrs):
    """
    # Let's create some data first
    >>> from collections import namedtuple
    >>> Person = namedtuple('Person', ('name', 'age', 'gender'))
    >>> alice = Person('Alice', 30, 'F')

    # Usage
    >>> make_csv_row = lambda n, a, g: '%s,%d,%s' % (n, a, g)
    >>> use_with(alice, make_csv_row, 'name', 'age', 'gender')
    'Alice,30,F'

    """
    args = [getattr(data, x) for x in attrs]
    return fn(*args)


def zip_with(fn, *colls):
    """Return the result of the function applied on the zip of the collections

    :param fn: a function
    :param colls: collections

    >>> print list(zip_with(lambda x, y: x-y, [10, 20, 30], [42, 19, 43]))
    [-32, 1, -13]

    """
    return itertools.starmap(fn, itertools.izip(*colls))


def unzip(colls):
    """Unzip collections"""
    return zip(*colls)


def concat(colls):
    """Concatenate a list of collections

    :param colls: a list of collections
    :returns: the concatenation of the collections

    >>> print concat(([1, 2], [3, 4]))
    [1, 2, 3, 4]

    """
    return list(itertools.chain.from_iterable(colls))


def mapcat(fn, colls):
    """Concatenate the result of a map

    :param fn: a function
    :param colls: a list of collections
    """
    return map(fn, concat(colls))


# TODO Fix and test dmap
def dmap(fn, record):
    """map for a directory

    :param fn: a function
    :param record: a dictionary
    :returns: a dictionary

    """
    values = (fn(v) for k, v in record.items())
    return dict(itertools.izip(record, values))


def compose(*fns):
    """Return the function composed with the given functions

    >>> add2 = lambda x: x+2
    >>> mult3 = lambda x: x*3
    >>> new_fn = compose(add2, mult3)
    >>> print new_fn(2)
    8

    .. note:: compose(fn1, fn2, fn3) is the same as fn1(fn2(fn3))
       which means that the last function provided is the first to be applied.

    """
    def compose2(f, g):
        return lambda x: f(g(x))
    return reduce(compose2, fns)


def groupby(f, sample):
    """Group elements in sub-samples by f

    >>> print groupby(len, ['John', 'Terry', 'Eric', 'Graham', 'Mickael'])
    {4: ['John', 'Eric'], 5: ['Terry'], 6: ['Graham'], 7: ['Mickael']}

    """
    d = collections.defaultdict(list)
    for item in sample:
        key = f(item)
        d[key].append(item)
    return dict(d)


def reductions(fn, seq, acc=None):
    """Return the intermediate values of a reduction

    >>> print reductions(lambda x, y: x + y, [1, 2, 3])
    [1, 3, 6]

    >>> print reductions(lambda x, y: x + y, [1, 2, 3], 10)
    [11, 13, 16]

    """
    indexes = xrange(len(seq))
    if acc:
        return map(lambda i: reduce(lambda x, y: fn(x, y), seq[:i+1], acc), indexes)
    else:
        return map(lambda i: reduce(lambda x, y: fn(x, y), seq[:i+1]), indexes)


def split(coll, factor):
    """Split a collection by using a factor

    >>> bands = ('Led Zeppelin', 'Debussy', 'Metallica', 'Iron Maiden', 'Bach')
    >>> styles = ('rock', 'classic', 'rock', 'rock', 'classic')
    >>> print split(bands, styles)
    {'classic': ['Debussy', 'Bach'], 'rock': ['Led Zeppelin', 'Metallica', 'Iron Maiden']}

    """
    groups = groupby(lambda x: x[0], itertools.izip(factor, coll))
    return dmap(lambda x: [y[1] for y in x], groups)


# FILTERING  {{{1

def duplicates(coll):
    """Return the duplicated items in the given collection"""
    return list(set(x for x in coll if coll.count(x) > 1))


def pluck(record, *keys, **kwargs):
    """

    >>> d = {'name': 'Lancelot', 'actor': 'John Cleese', 'color': 'blue'}
    >>> print pluck(d, 'name', 'color')
    {'color': 'blue', 'name': 'Lancelot'}

    # the keyword 'default' allows to replace a None value
    >>> d = {'year': 2014, 'movie': 'Bilbo'}
    >>> print pluck(d, 'year', 'movie', 'nb_aliens', default=0)
    {'movie': 'Bilbo', 'nb_aliens': 0, 'year': 2014}

    """
    default = kwargs.get('default', None)
    return reduce(lambda a, x: assoc(a, x, record.get(x, default)), keys, {})


def use(data, *attrs):
    """
    # Let's create some data first
    >>> from collections import namedtuple
    >>> Person = namedtuple('Person', ('name', 'age', 'gender'))
    >>> alice = Person('Alice', 30, 'F')

    # Usage
    >>> use(alice, 'name', 'gender')
    ['Alice', 'F']

    """
    return map(lambda x: getattr(data, x), attrs)


def get_in(record, *keys, **kwargs):
    """Return the value corresponding to the keys in a nested record

    >>> d = {'id': {'name': 'Lancelot', 'actor': 'John Cleese', 'color': 'blue'}}
    >>> print get_in(d, 'id', 'name')
    Lancelot

    >>> print get_in(d, 'id', 'age', default='?')
    ?

    """
    default = kwargs.get('default', None)
    return reduce(lambda a, x: a.get(x, default), keys, record)


def valuesof(record, keys):
    """Return the values corresponding to the given keys

    >>> band = {'name': 'Metallica', 'singer': 'James Hetfield', 'guitarist': 'Kirk Hammet'}
    >>> print valuesof(band, ('name', 'date', 'singer'))
    ['Metallica', None, 'James Hetfield']

    """
    if not isiterable(keys):
        keys = [keys]
    return map(record.get, keys)


def valueof(records, key):
    """Extract the value corresponding to the given key in all the dictionaries

    # >>> bands = [{'name': 'Led Zeppelin', 'singer': 'Robert Plant', 'guitarist': 'Jimmy Page'},
    # ....: {'name': 'Metallica', 'singer': 'James Hetfield', 'guitarist': 'Kirk Hammet'}]
    # >>> print valueof(bands, 'singer')
    # ['Robert Plant', 'James Hetfield']

    """
    if isinstance(records, dict):
        records = [records]
    return map(operator.itemgetter(key), records)


def take(n, seq):
    """Return the n first items in the sequence

    >>> take(3, xrange(10000))
    [0, 1, 2]

    """
    return list(itertools.islice(seq, 0, n))


def drop(n, seq):
    """Return the n last items in the sequence

    >>> drop(9997, xrange(10000))
    [9997, 9998, 9999]

    """
    return list(itertools.islice(seq, n, None))


# CREATION {{{1

def assoc(_d, key, value):
    """Associate a key with a value in a dictionary

    >>> movie = assoc({}, 'name', 'Holy Grail')
    >>> print movie
    {'name': 'Holy Grail'}

    """
    d = deepcopy(_d)
    d[key] = value
    return d


def dispatch(data, fns):
    """Apply the functions on the data

    :param data: the data
    :param fns: a list of functions

    >>> x = (1, 42, 5, 79)
    >>> print dispatch(x, (min, max))
    [1, 79]

    """
    return map(lambda fn: fn(data), fns)


def multimap(fn, colls):
    """Apply a function on multiple collections

    >>> print multimap(operator.add, ((1, 2, 3), (4, 5, 6)))
    [5, 7, 9]

    >>> f = lambda x, y, z: 2*x + 3*y - z
    >>> result = multimap(f, ((1, 2), (4, 1), (1, 1)))
    >>> result[0] == f(1, 4, 1)
    True
    >>> result[1] == f(2, 1, 1)
    True

    """
    return list(itertools.starmap(fn, zip(*colls)))


def pipe(data, *fns):
    """Apply functions recursively on your data

    >>> inc = lambda x: x + 1
    >>> pipe(42, inc, str) 
    '43'
    """
    return reduce(lambda acc, f: f(acc), fns, data)


def pipe_each(coll, *fns):
    """Apply functions recursively on your collection of data

    """
    return map(lambda x: pipe(x, *fns), coll)


# INSPECTION {{{1

def isiterable(coll):
    """Return True if the collection is any iterable except a string"""
    return hasattr(coll, "__iter__")


def are_in(items, collection):
    """Return True for each item in the collection

    >>> print are_in(['Terry', 'James'], ['Terry', 'John', 'Eric'])
    [True, False]

    """
    if not isinstance(items, (list, tuple)):
        items = (items, )
    return map(lambda x: x in collection, items)


def any_in(items, collection):
    """Return True if any of the items are in the collection

    :param items: items that may be in the collection
    :param collection: a collection

    >>> print any_in(2, [1, 3, 2])
    True
    >>> print any_in([1, 2], [1, 3, 2])
    True
    >>> print any_in([1, 2], [1, 3])
    True

    """
    return any(are_in(items, collection))


def all_in(items, collection):
    """Return True if all of the items are in the collection

    :param items: items that may be in the collection
    :param collection: a collection

    >>> print all_in(2, [1, 3, 2])
    True
    >>> print all_in([1, 2], [1, 3, 2])
    True
    >>> print all_in([1, 2], [1, 3])
    False

    """
    return all(are_in(items, collection))


def monotony(seq):
    """Determine the monotony of a sequence

    :param seq: a sequence

    :returns: 1 if the sequence is sorted (increasing)
    :returns: 0 if it is not sorted
    :returns: -1 if it is sorted in reverse order (decreasing)

    >>> monotony([1, 2, 3])
    1
    >>> monotony([1, 3, 2])
    0
    >>> monotony([3, 2, 1])
    -1

    """
    if seq == sorted(seq):
        return 1
    elif seq == list(reversed(sorted(seq))):
        return -1
    else:
        return 0


def attributes(data):
    """Return all the non callable and non special attributes of the input data"""
    return [x for x in dir(data) if not callable(x) and not x.startswith('__')]


def find(fn, record):
    """'Apply a function on the record and return the corresponding new record

    >>> print find(max, {'Terry': 30, 'Graham': 35, 'John': 27})
    {'Graham': 35}

    """
    values_result = fn(record.values())
    keys_result = [k for k, v in record.items() if v == values_result]
    return {keys_result[0]: values_result}


def dfilter(fn, record):
    """filter for a directory

    :param fn: A predicate function
    :param record: a dict
    :returns: a dict

    >>> odd = lambda x: x % 2 != 0
    >>> print dfilter(odd, {'Terry': 30, 'Graham': 35, 'John': 27})
    {'John': 27, 'Graham': 35}

    """
    return dict([(k, v) for k, v in record.items() if fn(v)])


def occurrences(coll, value=None):
    """Return the occurrences of the elements in the collection

    >>> print occurrences((1, 1, 2, 3))
    {1: 2, 2: 1, 3: 1}
    >>> print occurrences((1, 1, 2, 3), 1)
    2

    """
    count = {}
    for element in coll:
        count[element] = count.get(element, 0) + 1

    if value:
        count = count.get(value, 0)
    return count


def indexof(coll, item, start=0, default=None):
    """Return the index of the item in the collection

    :param coll: iterable
    :param item: scalar
    :param start: (optional) The start index
    :default: The default value of the index if the item is not in the collection

    :returns: idx -- The index of the item in the collection

    """
    if item in coll:
        return list(coll).index(item, start)
    else:
        return default


def indexesof(coll, item):
    """Return all the indexes of the item in the collection"""
    return [indexof(coll, item, i) for i in xrange(len(coll)) if coll[i] == item]


def count(fn, coll):
    """Return the count of True values returned by the function applied to the
    collection

    >>> count(lambda x: x % 2 == 0, [11, 22, 31, 24, 15])
    2

    """
    return len([x for x in coll if fn(x) is True])


def isdistinct(coll):
    """
    >>> isdistinct([1, 2, 3])
    True
    >>> isdistinct([1, 2, 2])
    False

    """
    most_common = collections.Counter(coll).most_common(1)
    return not most_common[0][1] > 1


def shift(func, *args, **kwargs):
    """This function is basically a beefed up lambda x: func(x, *args, **kwargs)
    
    `shift` comes in handy when it is used in a pipeline with a function that
    needs the passed value as its first argument.

    >>> def div(x, y): return float(x) / y

    # This is equivalent to div(42, 2):
    >>> shift(div, 2)(42)
    21.0

    # which is different from div(2, 42):
    >>> partial(div, 2)(42)
    0.047619047619047616

    """
    @wraps(func)
    def wrapped(x):
        return func(x, *args, **kwargs)
    return wrapped


