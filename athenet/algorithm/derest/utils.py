def change_order(a):
    """
    Change order so the last will be first if passed element is a tuple
    If not, nothing changes
    """
    try:
        return (a[-1], ) + a[:-1]
    except:
        return a


def make_iterable(a):
    """
    Check if a is iterable and if not, make it a one element tuple

    """
    try:
        iter(a)
        return a
    except TypeError:
        return (a, )


def add_tuples(a, b):
    """
    Add two elements as it were tuples

    :return tuple:
    """
    if not isinstance(a, tuple):
        a = (a, )
    if not isinstance(b, tuple):
        b = (b, )
    return a + b
