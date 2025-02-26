use std::borrow::Borrow;
use std::fmt;
use std::iter::FromIterator;

use crate::base::{self, Bound, SkipList};

/// a map based on skip list.
pub struct SkipMap<K, V> {
    inner: SkipList<K, V>,
}

impl<K, V> SkipMap<K, V> {
    /// Creates an empty `SkipMap`.
    pub fn new() -> Self {
        SkipMap {
            inner: SkipList::new(),
        }
    }

    /// Returns the number of elements in the map.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if the map contains no elements.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<K, V> SkipMap<K, V>
where
    K: Ord,
{
    /// Returns the entry with the smallest key.
    pub fn front(&self) -> Option<Entry<K, V>> {
        self.inner.front().map(Entry::new)
    }

    /// Returns the entry with the largest key.
    pub fn back(&self) -> Option<Entry<K, V>> {
        self.inner.back().map(Entry::new)
    }

    /// Returns `true` if the map contains a value for the specified key.
    pub fn contain_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.contain_key(key)
    }

    /// Returns an entry with the specified `key`.
    pub fn get<Q>(&self, key: &Q) -> Option<Entry<K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.get(key).map(Entry::new)
    }

    /// Returns an `Entry` pointing to the lowest element whose key is above
    /// the given bound. If no such element is found then `None` is
    /// returned.
    pub fn lower_bound<'a, Q>(&'a self, bound: Bound<&Q>) -> Option<Entry<'a, K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.lower_bound(bound).map(Entry::new)
    }

    /// Returns an `Entry` pointing to the highest element whose key is below
    /// the given bound. If no such element is found then `None` is
    /// returned.
    pub fn upper_bound<'a, Q>(&'a self, bound: Bound<&Q>) -> Option<Entry<'a, K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.upper_bound(bound).map(Entry::new)
    }

    /// Finds an entry with the specified key, or inserts a new `key`-`value` pair if none exist.
    pub fn get_or_insert(&mut self, key: K, value: V) -> Entry<K, V> {
        Entry::new(self.inner.get_or_insert(key, value))
    }

    /// Returns an iterator over all entries in the map.
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            inner: self.inner.iter(),
        }
    }
    /// Returns an iterator over a subset of entries in the skip list.
    pub fn range<'a, 'k, Min, Max>(
        &'a self,
        lower_bound: Bound<&'k Min>,
        upper_bound: Bound<&'k Max>,
    ) -> Range<'a, 'k, Min, Max, K, V>
    where
        K: Ord + Borrow<Min> + Borrow<Max>,
        Min: Ord + ?Sized + 'k,
        Max: Ord + ?Sized + 'k,
    {
        Range {
            inner: self.inner.range(lower_bound, upper_bound),
        }
    }
}

impl<K, V> SkipMap<K, V>
where
    K: Ord,
{
    /// Inserts a `key`-`value` pair into the map and returns the new entry.
    ///
    /// If there is an existing entry with this key, it will be removed before inserting the new
    /// one.
    pub fn insert(&mut self, key: K, value: V) -> Entry<K, V> {
        Entry::new(self.inner.insert(key, value))
    }

    /// Removes an entry with the specified `key` from the map and returns it.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.remove(key)
    }

    /// Removes an entry from the front of the map.
    pub fn pop_front(&mut self) -> Option<(K, V)> {
        self.inner.pop_front()
    }

    /// Removes an entry from the back of the map.
    pub fn pop_back(&mut self) -> Option<(K, V)> {
        self.inner.pop_back()
    }

    /// Iterates over the map and removes every entry.
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

impl<K, V> Default for SkipMap<K, V> {
    fn default() -> SkipMap<K, V> {
        SkipMap::new()
    }
}

impl<K, V> fmt::Debug for SkipMap<K, V>
where
    K: Ord + fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut m = f.debug_map();
        for e in self.iter() {
            m.entry(e.key(), e.value());
        }
        m.finish()
    }
}

impl<K, V> IntoIterator for SkipMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> IntoIter<K, V> {
        IntoIter {
            inner: self.inner.into_iter(),
        }
    }
}

impl<'a, K, V> IntoIterator for &'a SkipMap<K, V>
where
    K: Ord,
{
    type Item = Entry<'a, K, V>;
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

impl<K, V> FromIterator<(K, V)> for SkipMap<K, V>
where
    K: Ord,
{
    fn from_iter<I>(iter: I) -> SkipMap<K, V>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let mut s = SkipMap::new();
        for (k, v) in iter {
            s.get_or_insert(k, v);
        }
        s
    }
}

/// An owning iterator over the entries of a `SkipMap`.
pub struct IntoIter<K, V> {
    inner: base::IntoIter<K, V>,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        self.inner.next()
    }
}

impl<K, V> fmt::Debug for IntoIter<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "IntoIter {{ ... }}")
    }
}

/// A reference-counted entry in a map.
pub struct Entry<'a, K: 'a, V: 'a> {
    inner: base::Entry<'a, K, V>,
}

impl<'a, K, V> Entry<'a, K, V> {
    fn new(inner: base::Entry<'a, K, V>) -> Self {
        Entry { inner }
    }

    /// Returns a reference to the key.
    pub fn key(&self) -> &K {
        self.inner.key()
    }

    /// Returns a reference to the value.
    pub fn value(&self) -> &V {
        self.inner.value()
    }
}

impl<'a, K, V> Entry<'a, K, V>
where
    K: Ord,
{
    /// Moves to the next entry in the map.
    pub fn move_next(&mut self) -> bool {
        self.inner.move_next()
    }

    /// Moves to the previous entry in the map.
    pub fn move_prev(&mut self) -> bool {
        self.inner.move_prev()
    }

    /// Returns the next entry in the map.
    pub fn next(&self) -> Option<Entry<'a, K, V>> {
        self.inner.next().map(Entry::new)
    }

    /// Returns the previous entry in the map.
    pub fn prev(&self) -> Option<Entry<'a, K, V>> {
        self.inner.prev().map(Entry::new)
    }
}

impl<'a, K, V> Clone for Entry<'a, K, V> {
    fn clone(&self) -> Entry<'a, K, V> {
        Entry {
            inner: self.inner.clone(),
        }
    }
}

impl<'a, K, V> fmt::Debug for Entry<'a, K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Entry")
            .field(self.key())
            .field(self.value())
            .finish()
    }
}

/// An iterator over the entries of a `SkipMap`.
pub struct Iter<'a, K: 'a, V: 'a> {
    inner: base::Iter<'a, K, V>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V>
where
    K: Ord,
{
    type Item = Entry<'a, K, V>;

    fn next(&mut self) -> Option<Entry<'a, K, V>> {
        self.inner.next().map(Entry::new)
    }
}

impl<'a, K, V> DoubleEndedIterator for Iter<'a, K, V>
where
    K: Ord,
{
    fn next_back(&mut self) -> Option<Entry<'a, K, V>> {
        self.inner.next_back().map(Entry::new)
    }
}

impl<'a, K, V> fmt::Debug for Iter<'a, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Iter {{ ... }}")
    }
}

/// An iterator over the entries of a `SkipMap`.
pub struct Range<'a, 'k, Min, Max, K: 'a, V: 'a>
where
    K: Ord + Borrow<Min> + Borrow<Max>,
    Min: Ord + ?Sized + 'k,
    Max: Ord + ?Sized + 'k,
{
    inner: base::Range<'a, 'k, Min, Max, K, V>,
}
impl<'a, 'k, Min, Max, K, V> Iterator for Range<'a, 'k, Min, Max, K, V>
where
    K: Ord + Borrow<Min> + Borrow<Max>,
    Min: Ord + ?Sized + 'k,
    Max: Ord + ?Sized + 'k,
{
    type Item = Entry<'a, K, V>;

    fn next(&mut self) -> Option<Entry<'a, K, V>> {
        self.inner.next().map(Entry::new)
    }
}
impl<'a, 'k, Min, Max, K, V> DoubleEndedIterator for Range<'a, 'k, Min, Max, K, V>
where
    K: Ord + Borrow<Min> + Borrow<Max>,
    Min: Ord + ?Sized + 'k,
    Max: Ord + ?Sized + 'k,
{
    fn next_back(&mut self) -> Option<Entry<'a, K, V>> {
        self.inner.next_back().map(Entry::new)
    }
}

impl<'a, 'k, Min, Max, K, V> fmt::Debug for Range<'a, 'k, Min, Max, K, V>
where
    K: Ord + Borrow<Min> + Borrow<Max>,
    Min: Ord + ?Sized + 'k,
    Max: Ord + ?Sized + 'k,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Range {{ ... }}")
    }
}
