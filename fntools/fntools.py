"""Functional programming tools for data processing

`fntools` is a simple library providing the user with functional programming
functions to transform, filter and inspect Python data structures.
It introduces no new class or data structures but instead emphasizes the use
of:

* pure
* composable
* lightweight

functions to solve common problems while processing your data.

"""


from copy import deepcopy
import itertools
import operator
import collections
from functools import wraps


# TRANSFORMATION {{{1

def use_with(data, fn, *attrs):
    """Apply a function on the attributes of the data

    :param data: an object
    :param fn: a function
    :param attrs: some attributes of the object
    :returns: an object

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
    :returns: an iterator

    >>> list(zip_with(lambda x, y: x-y, [10, 20, 30], [42, 19, 43]))
    [-32, 1, -13]

    """
    return itertools.starmap(fn, itertools.izip(*colls))


def unzip(colls):
    """Unzip collections

    :param colls: collections 
    :returns: unzipped collections

    >>> unzip([[1, 2, 3], [10, 20, 30], [100, 200, 300]])
    [(1, 10, 100), (2, 20, 200), (3, 30, 300)]

    """
    return zip(*colls)


def concat(colls):
    """Concatenate a list of collections

    :param colls: a list of collections
    :returns: the concatenation of the collections

    >>> concat(([1, 2], [3, 4]))
    [1, 2, 3, 4]

    """
    return list(itertools.chain(*colls))


def mapcat(fn, colls):
    """Concatenate the result of a map

    :param fn: a function
    :param colls: a list of collections
    :returns: a list

    >>> mapcat(reversed,  [[3, 2, 1, 0], [6, 5, 4], [9, 8, 7]])
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    """
    return concat(map(fn, colls))


def dmap(fn, record):
    """map for a directory

    :param fn: a function
    :param record: a dictionary
    :returns: a dictionary

    >>> grades = [{'math': 13, 'biology': 17, 'chemistry': 18},
    ... {'math': 15, 'biology': 12, 'chemistry': 13},
    ... {'math': 16, 'biology': 17, 'chemistry': 11}]

    >>> def is_greater_than(x):
    ...     def func(y):
    ...         return y > x
    ...     return func

    >>> dmap(is_greater_than(15), grades[0])
    {'biology': True, 'chemistry': True, 'math': False}

    """
    values = (fn(v) for k, v in record.items())
    return dict(itertools.izip(record, values))


def rmap(fn, coll, is_iterable=None):
    """A recursive map

    :param fn: a function
    :param coll: a list
    :param isiterable: a predicate function determining whether a value is iterable.
    :returns: a list

    >>> rmap(lambda x: 2*x, [1, 2, [3, 4]])
    [2, 4, [6, 8]]

    """
    result = []
    for x in coll:
        if is_iterable is None:
            is_iterable = isiterable

        if is_iterable(x):
            y = rmap(fn, x)
        else:
            y = fn(x)
        result.append(y)
    return result


# TODO new can be a value or a function applied to x
def replace(x, old, new, fn=operator.eq):
    """
    Replace x with new if fn(x, old) is True.

    :param x: Any value
    :param old: The old value we want to replace
    :param new: The value replacing old
    :param fn: 
        The predicate function determining the relation between x and
        old. By default fn is the equality function.
    :returns: x or new

    >>> map(lambda x: replace(x, None, -1), [None, 1, 2, None])
    [-1, 1, 2, -1]

    """
    return new if fn(x, old) else x


def compose(*fns):
    """Return the function composed with the given functions

    :param fns: functions
    :returns: a function

    >>> add2 = lambda x: x+2
    >>> mult3 = lambda x: x*3
    >>> new_fn = compose(add2, mult3)
    >>> new_fn(2)
    8

    .. note:: compose(fn1, fn2, fn3) is the same as fn1(fn2(fn3))
       which means that the last function provided is the first to be applied.

    """
    def compose2(f, g):
        return lambda x: f(g(x))
    return reduce(compose2, fns)


def groupby(fn, coll):
    """Group elements in sub-collections by fn

    :param fn: a function
    :param coll: a collection
    :returns: a dictionary

    >>> groupby(len, ['John', 'Terry', 'Eric', 'Graham', 'Mickael'])
    {4: ['John', 'Eric'], 5: ['Terry'], 6: ['Graham'], 7: ['Mickael']}

    """
    d = collections.defaultdict(list)
    for item in coll:
        key = fn(item)
        d[key].append(item)
    return dict(d)


def reductions(fn, seq, acc=None):
    """Return the intermediate values of a reduction

    :param fn: a function
    :param seq: a sequence
    :param acc: the accumulator
    :returns: a list

    >>> reductions(lambda x, y: x + y, [1, 2, 3])
    [1, 3, 6]

    >>> reductions(lambda x, y: x + y, [1, 2, 3], 10)
    [11, 13, 16]

    """
    indexes = xrange(len(seq))
    if acc:
        return map(lambda i: reduce(lambda x, y: fn(x, y), seq[:i+1], acc), indexes)
    else:
        return map(lambda i: reduce(lambda x, y: fn(x, y), seq[:i+1]), indexes)


def split(coll, factor):
    """Split a collection by using a factor

    :param coll: a collection
    :param factor: a collection of factors
    :returns: a dictionary

    >>> bands = ('Led Zeppelin', 'Debussy', 'Metallica', 'Iron Maiden', 'Bach')
    >>> styles = ('rock', 'classic', 'rock', 'rock', 'classic')
    >>> split(bands, styles)
    {'classic': ['Debussy', 'Bach'], 'rock': ['Led Zeppelin', 'Metallica', 'Iron Maiden']}

    """
    groups = groupby(lambda x: x[0], itertools.izip(factor, coll))
    return dmap(lambda x: [y[1] for y in x], groups)


def assoc(_d, key, value):
    """Associate a key with a value in a dictionary

    :param _d: a dictionary
    :param key: a key in the dictionary
    :param value: a value for the key
    :returns: a new dictionary

    >>> data = {}
    >>> new_data = assoc(data, 'name', 'Holy Grail')
    >>> new_data
    {'name': 'Holy Grail'}
    >>> data
    {}

    .. note:: the original dictionary is not modified

    """
    d = deepcopy(_d)
    d[key] = value
    return d


def dispatch(data, fns):
    """Apply the functions on the data

    :param data: the data
    :param fns: a list of functions
    :returns: a collection

    >>> x = (1, 42, 5, 79)
    >>> dispatch(x, (min, max))
    [1, 79]

    """
    return map(lambda fn: fn(data), fns)


def multimap(fn, colls):
    """Apply a function on multiple collections

    :param fn: a function
    :param colls: collections
    :returns: a collection

    >>> multimap(operator.add, ((1, 2, 3), (4, 5, 6)))
    [5, 7, 9]

    >>> f = lambda x, y, z: 2*x + 3*y - z
    >>> result = multimap(f, ((1, 2), (4, 1), (1, 1)))
    >>> result[0] == f(1, 4, 1)
    True
    >>> result[1] == f(2, 1, 1)
    True

    """
    return list(itertools.starmap(fn, zip(*colls)))


def multistarmap(fn, *colls):
    """Apply a function on multiple collections

    :param fn: a function
    :param colls: collections
    :returns: a collection

    >>> multistarmap(operator.add, (1, 2, 3), (4, 5, 6))
    [5, 7, 9]

    >>> f = lambda x, y, z: 2*x + 3*y - z
    >>> result = multistarmap(f, (1, 2), (4, 1), (1, 1))
    >>> result[0] == f(1, 4, 1)
    True
    >>> result[1] == f(2, 1, 1)
    True

    """
    return list(itertools.starmap(fn, zip(*colls)))


def pipe(data, *fns):
    """Apply functions recursively on your data

    :param data: the data
    :param fns: functions
    :returns: an object

    >>> inc = lambda x: x + 1
    >>> pipe(42, inc, str)
    '43'

    """
    return reduce(lambda acc, f: f(acc), fns, data)


def pipe_each(coll, *fns):
    """Apply functions recursively on your collection of data

    :param coll: a collection
    :param fns: functions
    :returns: a list

    >>> inc = lambda x: x + 1
    >>> pipe_each([0, 1, 1, 2, 3, 5], inc, str)
    ['1', '2', '2', '3', '4', '6']

    """
    return map(lambda x: pipe(x, *fns), coll)


def shift(func, *args, **kwargs):
    """This function is basically a beefed up lambda x: func(x, *args, **kwargs)

    `shift` comes in handy when it is used in a pipeline with a function that
    needs the passed value as its first argument.

    :param func: a function
    :param args: objects
    :param kwargs: keywords

    >>> def div(x, y): return float(x) / y

    # This is equivalent to div(42, 2):
    >>> shift(div, 2)(42)
    21.0

    # which is different from div(2, 42):
    >>> from functools import partial
    >>> partial(div, 2)(42)
    0.047619047619047616

    """
    @wraps(func)
    def wrapped(x):
        return func(x, *args, **kwargs)
    return wrapped


def repeatedly(func):
    """Repeat a function taking no argument

    :param func: a function
    :returns: a generator

    >>> import random as rd
    >>> rd.seed(123)
    >>> take(3, repeatedly(rd.random))
    [0.052363598850944326, 0.08718667752263232, 0.4072417636703983]

    """
    while True:
        yield func()


def update(records, column, values):
    """Update the column of records

    :param records: a list of dictionaries
    :param column: a string
    :param values: an iterable or a function
    :returns: new records with the columns updated

    >>> movies = [
    ... {'title': 'The Holy Grail', 'year': 1975, 'budget': 4E5, 'total_gross': 5E6},
    ... {'title': 'Life of Brian', 'year': 1979, 'budget': 4E6, 'total_gross': 20E6},
    ... {'title': 'The Meaning of Life', 'year': 1983, 'budget': 9E6, 'total_gross': 14.9E6}
    ... ]
    >>> new_movies = update(movies, 'budget', lambda x: 2*x)
    >>> [new_movies[i]['budget'] for i,_ in enumerate(movies)]
    [800000.0, 8000000.0, 18000000.0]

    >>> new_movies2 = update(movies, 'budget', (40, 400, 900))
    >>> [new_movies2[i]['budget'] for i,_ in enumerate(movies)]
    [40, 400, 900]

    """
    new_records = deepcopy(records)

    if values.__class__.__name__ == 'function':
        for row in new_records:
            row[column] = values(row[column])
    elif isiterable(values):
        for i, row in enumerate(new_records):
            row[column] = values[i]
    else:
        msg = "You must provide a function or an iterable."
        raise ValueError(msg)
    return new_records


# FILTERING  {{{1

def duplicates(coll):
    """Return the duplicated items in the given collection

    :param coll: a collection
    :returns: a list of the duplicated items in the collection

    >>> duplicates([1, 1, 2, 3, 3, 4, 1, 1])
    [1, 3]

    """
    return list(set(x for x in coll if coll.count(x) > 1))


# TODO Second argument should be an iterable
def pluck(record, *keys, **kwargs):
    """Return the record with the selected keys

    :param record: a list of dictionaries
    :param keys: some keys from the record
    :param kwargs: keywords determining how to deal with the keys

    >>> d = {'name': 'Lancelot', 'actor': 'John Cleese', 'color': 'blue'}
    >>> pluck(d, 'name', 'color')
    {'color': 'blue', 'name': 'Lancelot'}

    # the keyword 'default' allows to replace a None value
    >>> d = {'year': 2014, 'movie': 'Bilbo'}
    >>> pluck(d, 'year', 'movie', 'nb_aliens', default=0)
    {'movie': 'Bilbo', 'nb_aliens': 0, 'year': 2014}

    """
    default = kwargs.get('default', None)
    return reduce(lambda a, x: assoc(a, x, record.get(x, default)), keys, {})


def pluck_each(records, columns):
    """Return the records with the selected columns

    :param records: a list of dictionaries
    :param columns: a list or a tuple
    :returns: a list of dictionaries with the selected columns

    >>> movies = [
    ... {'title': 'The Holy Grail', 'year': 1975, 'budget': 4E5, 'total_gross': 5E6},
    ... {'title': 'Life of Brian', 'year': 1979, 'budget': 4E6, 'total_gross': 20E6},
    ... {'title': 'The Meaning of Life', 'year': 1983, 'budget': 9E6, 'total_gross': 14.9E6}
    ... ]
    >>> pluck_each(movies, ('title', 'year'))
    [{'year': 1975, 'title': 'The Holy Grail'}, {'year': 1979, 'title': 'Life of Brian'}, {'year': 1983, 'title': 'The Meaning of Life'}]

    """
    return [pluck(records[i], *columns) for i, _ in enumerate(records)]


def use(data, attrs):
    """Return the values of the attributes for the given data

    :param data: the data
    :param attrs: strings
    :returns: a list

    # With a dict
    >>> band = {'name': 'Metallica', 'singer': 'James Hetfield', 'guitarist': 'Kirk Hammet'}
    >>> use(band, ('name', 'date', 'singer'))
    ['Metallica', None, 'James Hetfield']

    # With a non dict data structure
    >>> from collections import namedtuple
    >>> Person = namedtuple('Person', ('name', 'age', 'gender'))
    >>> alice = Person('Alice', 30, 'F')
    >>> use(alice, ('name', 'gender'))
    ['Alice', 'F']

    """
    if isinstance(data, dict):
        if not isiterable(attrs):
            attrs = [attrs]
        coll = map(data.get, attrs)
    else:
        coll = map(lambda x: getattr(data, x), attrs)
    return coll


def get_in(record, *keys, **kwargs):
    """Return the value corresponding to the keys in a nested record

    :param record: a dictionary
    :param keys: strings
    :param kwargs: keywords
    :returns: the value for the keys

    >>> d = {'id': {'name': 'Lancelot', 'actor': 'John Cleese', 'color': 'blue'}}
    >>> get_in(d, 'id', 'name')
    'Lancelot'

    >>> get_in(d, 'id', 'age', default='?')
    '?'

    """
    default = kwargs.get('default', None)
    return reduce(lambda a, x: a.get(x, default), keys, record)


# TODO Rename valueof
def valueof(records, key):
    """Extract the value corresponding to the given key in all the dictionaries

    >>> bands = [{'name': 'Led Zeppelin', 'singer': 'Robert Plant', 'guitarist': 'Jimmy Page'},
    ... {'name': 'Metallica', 'singer': 'James Hetfield', 'guitarist': 'Kirk Hammet'}]
    >>> valueof(bands, 'singer')
    ['Robert Plant', 'James Hetfield']

    """
    if isinstance(records, dict):
        records = [records]
    return map(operator.itemgetter(key), records)


def take(n, seq):
    """Return the n first items in the sequence

    :param n: an integer
    :param seq: a sequence
    :returns: a list

    >>> take(3, xrange(10000))
    [0, 1, 2]

    """
    return list(itertools.islice(seq, 0, n))


def drop(n, seq):
    """Return the n last items in the sequence

    :param n: an integer
    :param seq: a sequence
    :returns: a list

    >>> drop(9997, xrange(10000))
    [9997, 9998, 9999]

    """
    return list(itertools.islice(seq, n, None))


def find(fn, record):
    """Apply a function on the record and return the corresponding new record

    :param fn: a function
    :param record: a dictionary
    :returns: a dictionary

    >>> find(max, {'Terry': 30, 'Graham': 35, 'John': 27})
    {'Graham': 35}

    """
    values_result = fn(record.values())
    keys_result = [k for k, v in record.items() if v == values_result]
    return {keys_result[0]: values_result}


def remove(coll, value):
    """Remove all the occurrences of a given value

    :param coll: a collection
    :param value: the value to remove
    :returns: a list

    >>> data = ('NA', 0, 1, 'NA', 1, 2, 3, 'NA', 5)
    >>> remove(data, 'NA')
    (0, 1, 1, 2, 3, 5)

    """
    coll_class = coll.__class__
    return coll_class(x for x in coll if x != value)


# INSPECTION {{{1

def isiterable(coll):
    """Return True if the collection is any iterable except a string

    :param coll: a collection
    :returns: a boolean

    >>> isiterable(1)
    False
    >>> isiterable('iterable')
    False
    >>> isiterable([1, 2, 3])
    True

    """
    return hasattr(coll, "__iter__")


def are_in(items, collection):
    """Return True for each item in the collection

    :param items: a sub-collection
    :param collection: a collection
    :returns: a list of booleans

    >>> are_in(['Terry', 'James'], ['Terry', 'John', 'Eric'])
    [True, False]

    """
    if not isinstance(items, (list, tuple)):
        items = (items, )
    return map(lambda x: x in collection, items)


def any_in(items, collection):
    """Return True if any of the items are in the collection

    :param items: items that may be in the collection
    :param collection: a collection
    :returns: a boolean

    >>> any_in(2, [1, 3, 2])
    True
    >>> any_in([1, 2], [1, 3, 2])
    True
    >>> any_in([1, 2], [1, 3])
    True

    """
    return any(are_in(items, collection))


def all_in(items, collection):
    """Return True if all of the items are in the collection

    :param items: items that may be in the collection
    :param collection: a collection
    :returns: a boolean

    >>> all_in(2, [1, 3, 2])
    True
    >>> all_in([1, 2], [1, 3, 2])
    True
    >>> all_in([1, 2], [1, 3])
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
    """Return all the non callable and non special attributes of the input data

    :param data: an object
    :returns: a list

    >>> class table:
    ...     def __init__(self, name, rows, cols):
    ...         self.name = name
    ...         self.rows = rows
    ...         self.cols = cols

    >>> t = table('people', 100, 3)
    >>> attributes(t)
    ['cols', 'name', 'rows']

    """
    return [x for x in dir(data) if not callable(x) and not x.startswith('__')]


def find_each(fn, records):
    """Apply a function on the records and return the corresponding new record

    :param fn: a function
    :param records: a collection of dictionaries
    :returns: new records

    >>> grades = [{'math': 13, 'biology': 17, 'chemistry': 18},
    ... {'math': 15, 'biology': 12, 'chemistry': 13},
    ... {'math': 16, 'biology': 17, 'chemistry': 11}]
    >>> find_each(max, grades)
    [{'chemistry': 18}, {'math': 15}, {'biology': 17}]

    """
    return [find(fn, record) for record in records]


def dfilter(fn, record):
    """filter for a directory

    :param fn: A predicate function
    :param record: a dict
    :returns: a dict

    >>> odd = lambda x: x % 2 != 0
    >>> dfilter(odd, {'Terry': 30, 'Graham': 35, 'John': 27})
    {'John': 27, 'Graham': 35}

    """
    return dict([(k, v) for k, v in record.items() if fn(v)])


# TODO Refactor to avoid duplication
def _filter_occurrences(count, relat_op):
    """Filter the occurrences with respect to the selected relational operators"""
    # Filter the occurrences equal (or not equal) to a given value
    if "eq" in relat_op:
        count = dfilter(lambda x: x == relat_op["eq"], count)
    elif "ne" in relat_op:
        count = dfilter(lambda x: x != relat_op["ne"], count)

    # Filter the occurrences lower (or equal) than a given value
    if "lt" in relat_op:
        count = dfilter(lambda x: x < relat_op["lt"], count)
    elif "le" in relat_op:
        count = dfilter(lambda x: x <= relat_op["le"], count)

    # Filter the occurrences greater (or equal) than a given value
    if "gt" in relat_op:
        count = dfilter(lambda x: x > relat_op["gt"], count)
    elif "ge" in relat_op:
        count = dfilter(lambda x: x >= relat_op["ge"], count)
    return count


def occurrences(coll, value=None, **options):
    """Return the occurrences of the elements in the collection

    :param coll: a collection
    :param value: a value in the collection
    :param options: 
        an optional keyword used as a criterion to filter the
        values in the collection
    :returns: the frequency of the values in the collection as a dictionary

    >>> occurrences((1, 1, 2, 3))
    {1: 2, 2: 1, 3: 1}
    >>> occurrences((1, 1, 2, 3), 1)
    2

    # Filter the values of the occurrences that
    # are <, <=, >, >=, == or != than a given number
    >>> occurrences((1, 1, 2, 3), lt=3)
    {1: 2, 2: 1, 3: 1}
    >>> occurrences((1, 1, 2, 3), gt=1)
    {1: 2}
    >>> occurrences((1, 1, 2, 3), ne=1)
    {1: 2}

    """
    count = {}
    for element in coll:
        count[element] = count.get(element, 0) + 1

    if options:
        count = _filter_occurrences(count, options)

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

    >>> monties = ['Eric', 'John', 'Terry', 'Terry', 'Graham', 'Mickael']
    >>> indexof(monties, 'Terry')
    2

    >>> indexof(monties, 'Terry', start=3)
    3

    >>> indexof(monties, 'Terry', start=4) is None
    True

    """
    if item in coll[start:]:
        return list(coll).index(item, start)
    else:
        return default


def indexesof(coll, item):
    """Return all the indexes of the item in the collection

    :param coll: the collection
    :param item: a value
    :returns: a list of indexes

    >>> monties = ['Eric', 'John', 'Terry', 'Terry', 'Graham', 'Mickael']
    >>> indexesof(monties, 'Terry')
    [2, 3]

    """
    return [indexof(coll, item, i) for i in xrange(len(coll)) if coll[i] == item]


def count(fn, coll):
    """Return the count of True values returned by the predicate function applied to the
    collection

    :param fn: a predicate function
    :param coll: a collection
    :returns: an integer

    >>> count(lambda x: x % 2 == 0, [11, 22, 31, 24, 15])
    2

    """
    return len([x for x in coll if fn(x) is True])


# TODO Check collections.Counter can be imported
# (it is available only in recent versions of Python)
def isdistinct(coll):
    """Return True if all the items in the collections are distinct.

    :param coll: a collection
    :returns: a boolean

    >>> isdistinct([1, 2, 3])
    True
    >>> isdistinct([1, 2, 2])
    False

    """
    most_common = collections.Counter(coll).most_common(1)
    return not most_common[0][1] > 1


def nrow(records):
    """Return the number of rows in the records

    :param records: a list of dictionaries
    :returns: an integer

    >>> movies = [
    ... {'title': 'The Holy Grail', 'year': 1975, 'budget': 4E5, 'total_gross': 5E6},
    ... {'title': 'Life of Brian', 'year': 1979, 'budget': 4E6, 'total_gross': 20E6},
    ... {'title': 'The Meaning of Life', 'year': 1983, 'budget': 9E6, 'total_gross': 14.9E6}
    ... ]
    >>> nrow(movies)
    3

    """
    return len(records)


def ncol(records):
    """Return the number of columns in the records

    :param records: a list of dictionaries
    :returns: an integer

    >>> movies = [
    ... {'title': 'The Holy Grail', 'year': 1975, 'budget': 4E5, 'total_gross': 5E6},
    ... {'title': 'Life of Brian', 'year': 1979, 'budget': 4E6, 'total_gross': 20E6},
    ... {'title': 'The Meaning of Life', 'year': 1983, 'budget': 9E6, 'total_gross': 14.9E6}
    ... ]
    >>> ncol(movies)
    4

    """
    return len(records[0])


def names(records):
    """Return the column names of the records

    :param records: a list of dictionaries
    :returns: a list of strings

    >>> movies = [
    ... {'title': 'The Holy Grail', 'year': 1975, 'budget': 4E5, 'total_gross': 5E6},
    ... {'title': 'Life of Brian', 'year': 1979, 'budget': 4E6, 'total_gross': 20E6},
    ... {'title': 'The Meaning of Life', 'year': 1983, 'budget': 9E6, 'total_gross': 14.9E6}
    ... ]
    >>> names(movies)
    ['total_gross', 'year', 'budget', 'title']

    """
    return records[0].keys()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
