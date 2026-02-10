"""Collection utilities for coders."""


class OrderedSet:
    """A simple ordered set backed by a dict, used for file paths.

    Preserves insertion order and de-duplicates entries, similar to a set.
    The touch() method allows moving items to the end (MRU position),
    which helps keep "hot" (frequently edited) files later in the prompt.
    """

    def __init__(self, iterable=None):
        self._data = {}
        if iterable:
            for item in iterable:
                self._data[item] = None

    def add(self, item):
        """Add an item to the set. If already present, does nothing."""
        self._data[item] = None

    def remove(self, item):
        """Remove an item from the set. Raises KeyError if not present."""
        del self._data[item]

    def discard(self, item):
        """Remove an item from the set if present. No error if not present."""
        self._data.pop(item, None)

    def __contains__(self, item):
        """Check if item is in the set."""
        return item in self._data

    def __iter__(self):
        """Iterate over items in insertion order."""
        return iter(self._data.keys())

    def __len__(self):
        """Return the number of items in the set."""
        return len(self._data)

    def touch(self, item):
        """Move an existing item to the end (most recently used position).

        This is useful for keeping frequently edited ("hot") files
        appearing later in the prompt, which can improve cache efficiency.
        """
        if item in self._data:
            value = self._data.pop(item)
            self._data[item] = value
