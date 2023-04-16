COMP = 0

def comparator(x, i, j):
    """Swap x[i] and x[j] if they are out of order"""
    global COMP
    COMP += 1
    if x[i] > x[j]:
        x[i], x[j] = x[j], x[i]

def oddevenmergesort(x, indexes=None):
    """In-place odd-even mergesort, applied to slice of x defined by indexes. Assumes len(x) is a power of 2. """
    if indexes == None:
        indexes = range(len(x))
    n = len(indexes)
    if n > 1:
        oddevenmergesort(x, indexes[:n//2])
        oddevenmergesort(x, indexes[n//2:])
        oddevenmerge(x, indexes)

def oddevenmerge(x, indexes=None):
    """Assuming the first and second half of x are sorted, in-place merge. Optionally restrict to slice of x defined by indexes."""

    if indexes == None:
        indexes = range(len(x))

    if len(indexes) == 2:
        i, j = indexes
        comparator(x, i, j)
        return

    oddevenmerge(x, indexes[::2])
    oddevenmerge(x, indexes[1::2])

    for r in range(1, len(indexes)-1, 2):
        i, j = indexes[r], indexes[r+1]
        comparator(x, i, j)

unsorted = [3, 9, 2, 7, 1, 5, 8, 5, 2, 7, 1, 0, 2, 7, 5, 2]
copy = list(unsorted)
oddevenmergesort(copy)
print(COMP)
assert copy == sorted(unsorted)