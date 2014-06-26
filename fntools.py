"""Functional programming tools for data processing"""


from copy import deepcopy
import itertools
import operator


# def moving_average(y, i, n):
#     ys = lambda j: y[i-j] + y[i] + y[i+j]
#     return sum(y[i-j] + y[i] + y[i+j] for j in range(1, n+1)) / (2.0*n + 1.0)


# FUNCTIONS FOR LISTS {{{2

def zip_with(f, *colls):
    """Return the result of the function applied on the zip of the collections
    
    >>> print list(zip_with(lambda x, y: x-y, [10, 20, 30], [42, 19, 43]))
    [-32, 1, -13]

    """
    return itertools.starmap(f, zip(*colls))


def duplicates(coll):
    """Return the duplicated items in the given collection"""
    return list(set(x for x in coll if coll.count(x) > 1))
    

def concat(colls):
    """Concatenate a list of collections 
    >>> print concat(([1, 2], [3, 4]))
    [1, 2, 3, 4]

    """
    return list(itertools.chain.from_iterable(colls))


def flatten(colls):
    return list(itertools.chain(*colls))    


def mapcat(fn, colls):
    """Concatenate the result of a map"""
    return concat(map(fn, colls))


def occurrences(coll, value=None):
    """Return the occurrences of the elements in the collection
    
    >>> print occurrences((1, 1, 2, 3))
    {3: 1, 2: 1, 1: 2}

    >>> print occurrences((1, 1, 2, 3), 1)
    2
    """
    count = {}
    for element in coll:
        count.setdefault(element, 0)
        count[element] += 1

    if value:
        count = count[value]

    return count


def compose(*funcs):
    def compose2(f, g):
        return lambda x: f(g(x))
    return reduce(compose2, funcs)


def dispatch(data, funcs):
    """Apply the functions on the data
    
    >>> x = (1, 42, 5, 79)
    >>> print dispatch(x, (min, max))
    (1, 79)
    """
    return map(lambda func: func(data), funcs)
    

def indexesof(coll, element):
	return [i for i in xrange(len(coll)) if coll[i] == element]


# PREDICATES

def issorted(coll):
    """Determine if a collection is sorted
    
    Returns
    -------
    1 if the collection is sorted (increasing)
    0 if it is not sorted 
    -1 if is sorted in reverse order (decreasing)
    """
    if coll == sorted(coll):
        return 1
    elif coll == list(reversed(sorted(coll))):
        return -1
    else:
        return 0


# FUNCTIONS FOR DICTIONARIES {{{2

def assoc(_d, key, value):
    """Associate a key with a value in a dictionary
    >>> movie = assoc({}, 'name', 'Holy Grail')
    >>> print movie
    {'name': 'Holy Grail'}

    """
    d = deepcopy(_d)
    d[key] = value
    return d


def pluck(record, *keys):
    """
    >>> d = {"name": "Lancelot", "actor": "John Cleese", "color": "blue"}
    >>> print pluck(d, ["name", "color"])
    {"name": "Lancelot", "color": "blue"}

    """
    return reduce(lambda a, x: assoc(a, x, record.get(x)), keys, {})


def extract_values(record, keys):
    return map(record.get, keys)


def extract_from_key(records, key):
    """Extract the value corresponding to the given key in all the dictionaries

    >>> bands = [{"name": "Led Zeppelin", "singer": "Robert Plant", "guitarist": "Jimmy Page"},
    ....: {"name": "Metallica", "singer": "James Hetfield", "guitarist": "Kirk Hammet"}]
    >>> print extract_from_key(bands, "singer")
    ["Robert Plant", "James Hetfield"]

    """
    return map(operator.itemgetter(key), records)


def load_dataset():
    """Simple function providing a dumb dataset"""
    return [{"name": "Led Zeppelin", "singer": "Robert Plant", "guitarist": "Jimmy Page"},
            {"name": "Metallica", "singer": "James Hetfield", "guitarist": "Kirk Hammet"}]


def groupby(f, sample):
    """Group elements in sub-samples by f"""
    d = {}
    for item in sample:
        key = f(item)
        if key not in d:
            d[key] = []
        d[key].append(item)
    return d


def dmap(func, record):
    """A map for a directory"""
    values = (func(v) for k, v in record.items())
    return dict(zip(record, values))


# FUNCTIONS FOR INSPECTING THE DATA {{{2

def attributes(data):
    """Return all the non callable and non special attributes of the input data"""
    return [x for x in dir(data) if not callable(x) and not x.startswith("__")]


def find(func, record):
    """"Apply a function on the record and return the corresponding new record
    
    >>> print find(max, {"Terry": 30, "Graham": 35, "John": 27}) 
    {"Graham": 35}

    """
    values_result = func(record.values())
    keys_result = [k for k, v in record.items() if v == values_result]
    return {keys_result[0]: values_result}


def select_value(func, record):
    """Returns a record satisfying the input predicate function on the value

    :func: A predicate function
    :record: a dict
    :returns: a dict

    Usage
    -----
    >>> odd = lambda x: x % 2 != 0
    >>> print select(odd, {"Terry": 30, "Graham": 35, "John": 27})
    {"John": 27, "Graham": 35}

    """
    return dict([(k, v) for k, v in record.items() if func(v)])


def select_key(func, record):
    """Returns a record satisfying the input predicate function on the key

    :func: A predicate function
    :record: a dict
    :returns: a dict

    Usage
    -----
    >>> odd = lambda x: x % 2 != 0
    >>> print select(odd, {"Terry": 30, "Graham": 35, "John": 27})
    {"John": 27, "Graham": 35}

    """
    return dict([(k, v) for k, v in record.items() if func(v)])
