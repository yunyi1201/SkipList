use std::{
    borrow::Borrow,
    cmp,
    fmt::{self, Debug},
    marker::PhantomData,
    ops::{Deref, Index, IndexMut},
    ptr::{self, NonNull},
};

use rand::{self, Rng};

/// Number of bits needed to  store height.
const HIGHT_BITS: usize = 5;

/// Maximum height of a skip list tower.
const MAX_HEIGHT: usize = 1 << HIGHT_BITS;

/// An endpoint of a range of keys.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Bound<T> {
    /// An inclusive bound.
    Included(T),
    /// An exclusive bound.
    Excluded(T),
    /// An infinite endpoint. Indicates that there is no bound in this direction.
    Unbounded,
}

/// The tower of pointers.
#[repr(C)]
struct Tower<K, V> {
    pointers: [Option<NonNull<Node<K, V>>>; MAX_HEIGHT],
}

impl<K, V> Index<usize> for Tower<K, V> {
    type Output = Option<NonNull<Node<K, V>>>;

    fn index(&self, index: usize) -> &Self::Output {
        // This implementation is actually unsafe since we don't check if the
        // index is in-bound. But this fine since this is only used internally.
        unsafe { self.pointers.get_unchecked(index) }
    }
}

impl<K, V> IndexMut<usize> for Tower<K, V> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // This implementation is actually unsafe since we don't check if the
        // index is in-bound. But this fine since this is only used internally.
        debug_assert!(index < MAX_HEIGHT);
        unsafe { self.pointers.get_unchecked_mut(index) }
    }
}

/// Tower at the head of a skip list.
#[repr(C)]
struct Head<K, V> {
    pointers: [Option<NonNull<Node<K, V>>>; MAX_HEIGHT],
}

impl<K, V> Head<K, V> {
    fn new() -> Self {
        Head {
            pointers: Default::default(),
        }
    }
}

impl<K, V> Deref for Head<K, V> {
    type Target = Tower<K, V>;

    fn deref(&self) -> &Tower<K, V> {
        unsafe { &*(self as *const _ as *const Tower<K, V>) }
    }
}

/// A skip list node.
///
struct Node<K, V> {
    /// The Key.
    key: K,

    /// The Value.
    value: V,

    #[allow(dead_code)]
    /// The number of levels in which this node is installed.
    height: usize,

    /// The Tower of pointers point to next node.
    tower: Tower<K, V>,
}

impl<K, V> Node<K, V> {
    fn new(key: K, value: V, height: usize) -> Self {
        Node {
            key,
            value,
            height,
            tower: Tower {
                pointers: [None; MAX_HEIGHT],
            },
        }
    }
    fn height(&self) -> usize {
        self.height
    }
}

/// A search result.
///
/// the result indicates whether the key was found, as well as what were the adjacent nodes to the
/// key on each level of the skip list.
struct Position<K, V> {
    /// Reference to a node with the given key, if found.
    ///
    /// if this is `Some` then it will point to the same node as `right[0]`.
    found: Option<*mut Node<K, V>>,

    /// Adjacent nodes with smaller keys (predecessors) on each level.
    left: [*mut Tower<K, V>; MAX_HEIGHT],

    /// Adjacent nodes with equal or greater keys (successors) on each level.
    right: [*const Node<K, V>; MAX_HEIGHT],
}

/// A skip list.
///
pub struct SkipList<K, V> {
    /// The head of the skip list (just a dummy node, not a real entry).
    head: Head<K, V>,
    /// The number of entry in the skip list.
    len: usize,
    /// The height of the skip list.
    level: usize,

    level_generator: rand::rngs::ThreadRng,
    marker: PhantomData<Box<Node<K, V>>>,
}

impl<K, V> Default for SkipList<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> SkipList<K, V> {
    /// Return a new, emtry skip list.
    pub fn new() -> Self {
        SkipList {
            head: Head::new(),
            len: 0,
            level: 1,
            level_generator: rand::rng(),
            marker: PhantomData,
        }
    }

    /// Return `true` if the skip list is emtry.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the number of entries in the skip list.
    pub fn len(&self) -> usize {
        self.len
    }

    fn random_height(&mut self) -> usize {
        let mut height = 1;
        // Increase the height with a 50% probability.
        while height < MAX_HEIGHT && self.level_generator.random_bool(0.5) {
            height += 1;
        }
        height
    }
}

impl<K, V> SkipList<K, V>
where
    K: Ord,
{
    /// Return the entry with the smallest key.
    pub fn front(&self) -> Option<Entry<K, V>> {
        self.head[0].as_ref().map(|node| Entry {
            parent: self,
            node: unsafe { node.as_ref() },
        })
    }

    /// Return the entry with the largest key.
    pub fn back(&self) -> Option<Entry<K, V>> {
        let n = self.search_bound(Bound::Unbounded, true)?;
        Some(Entry {
            parent: self,
            node: n,
        })
    }

    /// Return `true` if the map contain a value for the specified key.
    pub fn contain_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(key).is_some()
    }

    pub fn get_or_insert(&mut self, key: K, value: V) -> Entry<K, V> {
        self.insert_internal(key, value, false)
    }

    /// Return an entry with the sepified `key`.
    pub fn get<'a, Q>(&'a self, key: &Q) -> Option<Entry<'a, K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let search = self.search_position(key);
        search.found.map(|node| Entry {
            parent: self,
            node: unsafe { &*node },
        })
    }

    /// Returns an interator over all entries in the skip list.
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            parent: self,
            head: None,
            tail: None,
        }
    }

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
            parent: self,
            lower_bound,
            upper_bound,
            head: None,
            tail: None,
        }
    }

    /// Searches for a key in the skip list and returns a list of all adjacent nodes.
    fn search_position<Q>(&self, key: &Q) -> Position<K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut result = Position {
            found: None,
            left: [(&self.head as &Tower<K, V>) as *const Tower<K, V> as *mut Tower<K, V>;
                MAX_HEIGHT],
            right: [ptr::null(); MAX_HEIGHT],
        };

        let mut level = (self.level - 1) as isize;
        // Two adjacent nodes at the current level;
        let mut pred = &*self.head;
        unsafe {
            loop {
                if level < 0 {
                    break;
                }
                let mut curr = pred[level as usize];
                // Iterate through the current level until we reach a node with a key greater
                // than or equal to `key`.
                while let Some(c) = curr.as_ref() {
                    let c = c.as_ref();
                    let succ = c.tower[level as usize];
                    match c.key.borrow().cmp(key) {
                        cmp::Ordering::Greater => break,
                        cmp::Ordering::Equal => {
                            result.found = Some(c as *const Node<K, V> as *mut Node<K, V>);
                            break;
                        }
                        cmp::Ordering::Less => {}
                    }
                    // Move one step forward.
                    pred = &c.tower;
                    curr = succ;
                }
                result.left[level as usize] = pred as *const Tower<K, V> as *mut Tower<K, V>;
                if let Some(c) = curr.as_ref() {
                    result.right[level as usize] = c.as_ref();
                } else {
                    result.right[level as usize] = ptr::null();
                }
                level -= 1;
            }
        }
        result
    }

    /// Insert an entry with the specified `key` and `value` into the skip list.
    ///
    /// If `replace` is `true`, then the existing entry with the same key will be removed before
    fn insert_internal(&mut self, key: K, value: V, replace: bool) -> Entry<'_, K, V> {
        unsafe {
            // First try search for the key.
            let search = self.search_position(&key);
            if let Some(r) = search.found {
                if replace {
                    // If a node with the key was found and we should replace it
                    //
                    (*r).value = value;
                }
                // If a node with the key was found and we're not going to replace it,
                // return the existing entry.
                return Entry {
                    parent: self,
                    node: &*r,
                };
            }
            // create a new node.
            let height = self.random_height();
            let mut new_node = Box::new(Node::new(key, value, height));

            // set the successors of the new node.
            new_node
                .tower
                .pointers
                .iter_mut()
                .enumerate()
                .for_each(|(i, next)| {
                    *next = search.right[i].as_ref().map(NonNull::from);
                });

            // set the predecessors of the new node.
            search
                .left
                .iter()
                .enumerate()
                .take_while(|(i, _)| *i < height)
                .for_each(|(i, pred)| {
                    let pred = *pred;
                    (*pred).pointers[i] = Some(NonNull::from(&*new_node));
                });
            self.level = cmp::max(self.level, height);
            self.len += 1;
            let new_node = Box::leak(new_node);
            Entry {
                parent: self,
                node: &*new_node,
            }
        }
    }

    /// Returns an `Entry` pointing to the lowest element whose key is above
    /// the given bound. If no such element is found then `None` is
    /// returned.
    pub fn lower_bound<Q>(&self, bound: Bound<&Q>) -> Option<Entry<'_, K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let n = self.search_bound(bound, false)?;
        Some(Entry {
            parent: self,
            node: n,
        })
    }

    /// Returns an `Entry` pointing to the highest element whose key is below
    /// the given bound. If no such element is found then `None` is
    /// returned.
    pub fn upper_bound<Q>(&self, bound: Bound<&Q>) -> Option<Entry<'_, K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let n = self.search_bound(bound, true)?;
        Some(Entry {
            parent: self,
            node: n,
        })
    }

    /// Searches for first/last node that is greater/less/equal to a key in the skip list.
    ///
    /// If `upper_bound` is `true`, then the search will return the last node that is less than (or equal to ) the key.
    ///
    /// If `upper_bound` is `false`, then the search will return the first node that is greater than (or equal to) the key.
    fn search_bound<'a, Q>(&'a self, bound: Bound<&Q>, upper_bound: bool) -> Option<&'a Node<K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut level = (self.level - 1) as isize;
        // the current best node.
        let mut result = None;

        let mut pred = &*self.head;
        unsafe {
            while level >= 0 {
                let mut curr = pred[level as usize];

                // Iterate through the current level until we reach a node with a key greater than or equal to `key`.
                while let Some(c) = curr.as_ref() {
                    let succ = c.as_ref().tower[level as usize];

                    // If `curr` contains a key that is greater than or equal to `key`, we're done with this level.
                    // The condition detemines whether we should  stop the search. For the upper bound, we return the last
                    // node before the condition became true. For the lower bound, we return the first node after the condition
                    // became true.
                    if upper_bound {
                        if !below_upper_bound(&bound, c.as_ref().key.borrow()) {
                            break;
                        }
                        result = Some(c.as_ref());
                    } else if above_lower_bound(&bound, c.as_ref().key.borrow()) {
                        result = Some(c.as_ref());
                        break;
                    }

                    pred = &c.as_ref().tower;
                    curr = succ;
                }
                level -= 1;
            }
            result
        }
    }

    /// Returns the successor of a node.
    fn next_node(&self, pred: &Tower<K, V>) -> Option<&Node<K, V>> {
        unsafe { pred[0].as_ref().map(|node| node.as_ref()) }
    }

    /// Unlink a node from the skip list.
    fn unlink_position(&mut self, search: Position<K, V>) -> Option<(K, V)> {
        unsafe {
            let n = search.found?;
            let key = ptr::read(&(*n).key);
            let value = ptr::read(&(*n).value);
            let n = &*n;

            // Unlink the node at each level of the skip list.
            for level in (0..n.height()).rev() {
                let succ = n.tower[level];
                let pred = &mut *search.left[level];
                pred[level] = succ;
            }
            let _ = Box::from_raw(n as *const Node<K, V> as *mut Node<K, V>);
            self.len -= 1;
            Some((key, value))
        }
    }

    ///
    ///
    pub fn clear(&mut self) {}
}

impl<K, V> SkipList<K, V>
where
    K: Ord,
{
    /// Insert a `key`-`value` pair into the skip list and return the new entry.
    /// If there is existing entry with this key, it will be replace with the new value.
    pub fn insert(&mut self, key: K, value: V) -> Entry<'_, K, V> {
        self.insert_internal(key, value, true)
    }

    /// Removes an entry with sepecified `key` from the skip list and return it.
    ///
    pub fn remove<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let search = self.search_position(key);
        self.unlink_position(search)
    }

    /// Removes an entry from the front of the skip list.
    pub fn pop_front(&mut self) -> Option<(K, V)> {
        let search = self.search_bound(Bound::Unbounded, false)?;
        let search = self.search_position(search.key.borrow());
        self.unlink_position(search)
    }

    /// Removes an entry from the back of the skip list.
    pub fn pop_back(&mut self) -> Option<(K, V)> {
        let search = self.search_bound(Bound::Unbounded, true)?;
        let search = self.search_position(search.key.borrow());
        self.unlink_position(search)
    }
}

impl<K, V> Drop for SkipList<K, V> {
    fn drop(&mut self) {
        unsafe {
            let mut curr = self.head[0];
            while let Some(c) = curr.as_ref() {
                let node = c.as_ref();
                let next = node.tower[0];

                let _ = Box::from_raw(c.as_ptr());

                curr = next;
            }
        }
    }
}

impl<K, V> IntoIterator for SkipList<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            node: self.head[0].map(|n| n.as_ptr()).unwrap_or(ptr::null_mut()),
        }
    }
}

/// An entry in a skip list.
///
/// the lifetime of the key and value are the same as that of the skip list.
pub struct Entry<'a, K: 'a, V: 'a> {
    #[allow(dead_code)]
    parent: &'a SkipList<K, V>,
    node: &'a Node<K, V>,
}

impl<K, V> Entry<'_, K, V> {
    /// Return a reference to the key of the entry.
    pub fn key(&self) -> &K {
        &self.node.key
    }

    /// Return a reference to the value of the entry.
    pub fn value(&self) -> &V {
        &self.node.value
    }
}

impl<'a, K, V> Clone for Entry<'a, K, V> {
    fn clone(&self) -> Entry<'a, K, V> {
        Entry {
            parent: self.parent,
            node: self.node,
        }
    }
}

impl<'a, K, V> Entry<'a, K, V>
where
    K: Ord,
{
    /// Moves to the next entry in the skip list.
    pub fn move_next(&mut self) -> bool {
        match self.next() {
            None => false,
            Some(n) => {
                *self = n;
                true
            }
        }
    }

    /// Returns the next entry in the skip list.
    pub fn next(&self) -> Option<Entry<'a, K, V>> {
        let n = self.parent.next_node(&self.node.tower)?;
        Some(Entry {
            parent: self.parent,
            node: n,
        })
    }

    /// Moves to the previous entry in the skip list.
    pub fn move_prev(&mut self) -> bool {
        match self.prev() {
            None => false,
            Some(n) => {
                *self = n;
                true
            }
        }
    }

    /// Returns the previous entry in the skip list.
    pub fn prev(&self) -> Option<Entry<'a, K, V>> {
        let n = self
            .parent
            .search_bound(Bound::Excluded(&self.node.key), true)?;
        Some(Entry {
            parent: self.parent,
            node: n,
        })
    }
}

impl<K, V> Debug for Entry<'_, K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Entry")
            .field(&self.key())
            .field(&self.value())
            .finish()
    }
}

pub struct Iter<'a, K: 'a, V: 'a> {
    parent: &'a SkipList<K, V>,
    head: Option<&'a Node<K, V>>,
    tail: Option<&'a Node<K, V>>,
}

impl<'a, K: 'a, V: 'a> Iterator for Iter<'a, K, V>
where
    K: Ord,
{
    type Item = Entry<'a, K, V>;
    fn next(&mut self) -> Option<Entry<'a, K, V>> {
        self.head = match self.head {
            Some(n) => self.parent.next_node(&n.tower),
            None => self.parent.next_node(&self.parent.head),
        };
        if let (Some(h), Some(t)) = (self.head, self.tail) {
            if h.key >= t.key {
                self.head = None;
                self.tail = None;
            }
        }
        self.head.map(|n| Entry {
            parent: self.parent,
            node: n,
        })
    }
}

impl<'a, K: 'a, V: 'a> DoubleEndedIterator for Iter<'a, K, V>
where
    K: Ord,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.tail = match self.tail {
            Some(n) => self.parent.search_bound(Bound::Excluded(&n.key), true),
            None => self.parent.search_bound(Bound::Unbounded, true),
        };
        if let (Some(h), Some(t)) = (self.head, self.tail) {
            if h.key >= t.key {
                self.head = None;
                self.tail = None;
            }
        }
        self.tail.map(|n| Entry {
            parent: self.parent,
            node: n,
        })
    }
}

/// An owning iterator over the entries of a skip list.
pub struct IntoIter<K, V> {
    /// All preceeding nodes have already been destroyed.
    node: *mut Node<K, V>,
}

impl<K, V> Drop for IntoIter<K, V> {
    fn drop(&mut self) {
        while !self.node.is_null() {
            unsafe {
                //
                let next = (*self.node).tower[0];

                let _ = Box::from_raw(self.node);

                self.node = match next {
                    Some(n) => n.as_ptr(),
                    None => ptr::null_mut(),
                };
            }
        }
    }
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);
    fn next(&mut self) -> Option<(K, V)> {
        unsafe {
            if self.node.is_null() {
                return None;
            }
            let key = ptr::read(&(*self.node).key);
            let value = ptr::read(&(*self.node).value);

            let next = (*self.node).tower[0];

            let _ = Box::from_raw(self.node);

            self.node = match next {
                Some(n) => n.as_ptr(),
                None => ptr::null_mut(),
            };

            Some((key, value))
        }
    }
}

/// Helper function to check if a value is above a lower bound
fn above_lower_bound<T: Ord + ?Sized>(bound: &Bound<&T>, other: &T) -> bool {
    match *bound {
        Bound::Unbounded => true,
        Bound::Included(key) => other >= key,
        Bound::Excluded(key) => other > key,
    }
}

/// Helper function to check if a value is below an upper bound
fn below_upper_bound<T: Ord + ?Sized>(bound: &Bound<&T>, other: &T) -> bool {
    match *bound {
        Bound::Unbounded => true,
        Bound::Included(key) => other <= key,
        Bound::Excluded(key) => other < key,
    }
}

/// An iterator over a subset of entries of a `SkipList`.
pub struct Range<'a, 'k, Min, Max, K: 'a, V: 'a>
where
    K: Ord + Borrow<Min> + Borrow<Max>,
    Min: Ord + ?Sized + 'k,
    Max: Ord + ?Sized + 'k,
{
    parent: &'a SkipList<K, V>,
    lower_bound: Bound<&'k Min>,
    upper_bound: Bound<&'k Max>,
    head: Option<&'a Node<K, V>>,
    tail: Option<&'a Node<K, V>>,
}

impl<'a, 'k, Min, Max, K, V> Iterator for Range<'a, 'k, Min, Max, K, V>
where
    K: Ord + Borrow<Min> + Borrow<Max>,
    Min: Ord + ?Sized + 'k,
    Max: Ord + ?Sized + 'k,
{
    type Item = Entry<'a, K, V>;

    fn next(&mut self) -> Option<Self::Item> {
        self.head = match self.head {
            Some(n) => self.parent.next_node(&n.tower),
            None => self.parent.search_bound(self.lower_bound, false),
        };

        if let Some(h) = self.head {
            let bound = match self.tail {
                Some(t) => Bound::Excluded(t.key.borrow()),
                None => self.upper_bound,
            };
            if !below_upper_bound(&bound, h.key.borrow()) {
                self.head = None;
                self.tail = None;
            }
        }
        self.head.map(|n| Entry {
            parent: self.parent,
            node: n,
        })
    }
}

impl<'k, Min, Max, K, V> DoubleEndedIterator for Range<'_, 'k, Min, Max, K, V>
where
    K: Ord + Borrow<Min> + Borrow<Max>,
    Min: Ord + ?Sized + 'k,
    Max: Ord + ?Sized + 'k,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.tail = match self.tail {
            Some(n) => self.parent.search_bound(Bound::Excluded(&n.key), true),
            None => self.parent.search_bound(self.upper_bound, true),
        };

        if let Some(t) = self.tail {
            let bound = match self.head {
                Some(h) => Bound::Excluded(h.key.borrow()),
                None => self.lower_bound,
            };
            if !above_lower_bound(&bound, t.key.borrow()) {
                self.head = None;
                self.tail = None;
            }
        }
        self.tail.map(|n| Entry {
            parent: self.parent,
            node: n,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    const N: i32 = 2000;
    const R: i32 = 5000;
    fn skip_list() -> (BTreeSet<i32>, SkipList<i32, i32>) {
        let mut rand = rand::rng();
        let mut list = SkipList::<i32, i32>::new();
        let mut set = BTreeSet::new();
        (0..N).for_each(|_| {
            let key = rand.random_range(0..R);
            if set.insert(key) {
                list.insert(key, key);
            }
        });
        (set, list)
    }

    #[test]
    fn test_empty() {
        let mut s = SkipList::<i32, i32>::new();
        assert!(!s.contain_key(&10));
        assert!(s.is_empty());
        s.insert(1, 10);
        assert!(!s.is_empty());
        s.insert(2, 20);
        s.insert(3, 30);
        assert!(!s.is_empty());

        s.remove(&2);
        assert!(!s.is_empty());

        s.remove(&1);
        assert!(!s.is_empty());

        s.remove(&3);
        assert!(s.is_empty());
    }

    #[test]
    fn test_insert() {
        let insert = [0, 4, 2, 12, 8, 7, 11, 5];
        let not_present = [1, 3, 6, 9, 10];
        let mut s = SkipList::new();

        for &x in &insert {
            s.insert(x, x * 10);
            assert_eq!(*s.get(&x).unwrap().value(), x * 10);
        }

        for &x in &not_present {
            assert!(s.get(&x).is_none());
        }
    }

    #[test]
    fn test_remove() {
        let insert = [0, 4, 2, 12, 8, 7, 11, 5];
        let not_present = [1, 3, 6, 9, 10];
        let remove = [2, 12, 8];
        let remaining = [0, 4, 5, 7, 11];

        let mut s = SkipList::new();

        for &x in &insert {
            s.insert(x, x * 10);
        }
        for x in &not_present {
            assert!(s.remove(x).is_none());
        }
        for x in &remove {
            assert!(s.remove(x).is_some());
        }

        let mut v = vec![];
        let mut e = s.front().unwrap();
        loop {
            v.push(*e.key());
            if !e.move_next() {
                break;
            }
        }

        assert_eq!(v, remaining);
        for x in &insert {
            s.remove(x);
        }
        assert!(s.is_empty());
    }

    #[test]
    fn test_entry() {
        let mut s = SkipList::new();

        assert!(s.front().is_none());
        assert!(s.back().is_none());

        for &x in &[4, 2, 12, 8, 7, 11, 5] {
            s.insert(x, x * 10);
        }

        let mut e = s.front().unwrap();
        assert_eq!(*e.key(), 2);
        assert!(!e.move_prev());
        assert!(e.move_next());
        assert_eq!(*e.key(), 4);

        e = s.back().unwrap();
        assert_eq!(*e.key(), 12);
        assert!(!e.move_next());
        assert!(e.move_prev());
        assert_eq!(*e.key(), 11);
    }
    #[test]
    fn test_len() {
        let mut s = SkipList::new();
        assert_eq!(s.len(), 0);
        for (i, &x) in [4, 2, 12, 8, 7, 11, 5].iter().enumerate() {
            s.insert(x, x);
            assert_eq!(s.len(), i + 1);
        }

        s.insert(5, 0);
        assert_eq!(s.len(), 7);
        s.insert(5, 0);
        assert_eq!(s.len(), 7);
    }

    #[test]
    fn test_insert_and_get() {
        let (set, list) = skip_list();
        assert_eq!(list.len(), set.len());
        for i in 0..R {
            if list.contain_key(&i) {
                assert!(set.contains(&i));
            } else {
                assert!(!set.contains(&i));
            }
        }

        for i in 0..R {
            if set.contains(&i) {
                let entry = list.get(&i).unwrap();
                assert_eq!(entry.key(), &i);
                assert_eq!(entry.value(), &i);
            }
        }
    }
    #[test]
    fn test_search_bound() {
        let (set, list) = skip_list();
        let mut key = 0;
        let node = list.search_bound(Bound::Included(&key), false);
        assert!(node.is_some());
        assert!(node.unwrap().key >= key);

        key = R;
        let node = list.search_bound(Bound::Excluded(&key), true);
        assert!(node.is_some());
        assert!(node.unwrap().key < key);

        for i in 0..R {
            let node = list
                .search_bound(Bound::Included(&i), true)
                .map(|node| node.key);
            let expected = set.range(..=i).next_back().map(|&x| x);
            assert_eq!(node, expected);

            let expected = set.range(i..).next().map(|&x| x);
            let node = list
                .search_bound(Bound::Included(&i), false)
                .map(|node| node.key);
            assert_eq!(node, expected);
        }
    }

    #[test]
    fn test_front_and_back() {
        let mut s = SkipList::new();
        assert!(s.front().is_none());
        assert!(s.back().is_none());

        for &x in &[4, 2, 12, 8, 7, 11, 5] {
            s.insert(x, x * 10);
        }

        assert_eq!(*s.front().unwrap().key(), 2);
        assert_eq!(*s.back().unwrap().key(), 12);
    }

    #[test]
    fn test_get_or_insert() {
        let mut s = SkipList::new();
        s.insert(3, 3);
        s.insert(5, 5);
        s.insert(1, 1);
        s.insert(4, 4);
        s.insert(2, 2);

        assert_eq!(*s.get(&4).unwrap().value(), 4);
        assert_eq!(*s.insert(4, 40).value(), 40);
        assert_eq!(*s.get(&4).unwrap().value(), 40);

        assert_eq!(*s.get_or_insert(4, 400).value(), 40);
        assert_eq!(*s.get(&4).unwrap().value(), 40);
        assert_eq!(*s.get_or_insert(6, 600).value(), 600);
    }

    #[test]
    fn test_iter() {
        let mut s = SkipList::new();
        for &x in &[4, 2, 12, 8, 7, 11, 5] {
            s.insert(x, x * 10);
        }

        assert_eq!(
            s.iter().map(|e| *e.key()).collect::<Vec<_>>(),
            &[2, 4, 5, 7, 8, 11, 12]
        );
    }

    #[test]
    fn test_range() {
        use Bound::*;
        let mut s = SkipList::new();
        let v = (0..10).map(|x| x * 10).collect::<Vec<_>>();
        for &x in v.iter() {
            s.insert(x, x);
        }

        assert_eq!(
            s.iter().map(|x| *x.value()).collect::<Vec<_>>(),
            vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        );

        assert_eq!(
            s.iter().rev().map(|x| *x.value()).collect::<Vec<_>>(),
            vec![90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
        );

        assert_eq!(
            s.range(Unbounded, Unbounded)
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        );

        assert_eq!(
            s.range(Included(&0), Unbounded)
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        );

        assert_eq!(
            s.range(Excluded(&0), Unbounded)
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![10, 20, 30, 40, 50, 60, 70, 80, 90]
        );

        assert_eq!(
            s.range(Included(&25), Unbounded)
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![30, 40, 50, 60, 70, 80, 90]
        );
        assert_eq!(
            s.range(Excluded(&25), Unbounded)
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![30, 40, 50, 60, 70, 80, 90]
        );
        assert_eq!(
            s.range(Included(&70), Unbounded)
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![70, 80, 90]
        );

        assert_eq!(
            s.range(Excluded(&70), Unbounded)
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![80, 90]
        );
        assert_eq!(
            s.range(Included(&100), Unbounded)
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![]
        );
        assert_eq!(
            s.range(Excluded(&100), Unbounded)
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![]
        );

        assert_eq!(
            s.range(Unbounded, Included(&90))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        );
        assert_eq!(
            s.range(Unbounded, Excluded(&90))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![0, 10, 20, 30, 40, 50, 60, 70, 80]
        );
        assert_eq!(
            s.range(Unbounded, Included(&25))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![0, 10, 20]
        );
        assert_eq!(
            s.range(Unbounded, Excluded(&25))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![0, 10, 20]
        );

        assert_eq!(
            s.range(Unbounded, Included(&70))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![0, 10, 20, 30, 40, 50, 60, 70]
        );
        assert_eq!(
            s.range(Unbounded, Excluded(&70))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![0, 10, 20, 30, 40, 50, 60]
        );
        assert_eq!(
            s.range(Unbounded, Included(&-1))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![]
        );
        assert_eq!(
            s.range(Unbounded, Excluded(&-1))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![]
        );

        assert_eq!(
            s.range(Included(&25), Included(&80))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![30, 40, 50, 60, 70, 80]
        );
        assert_eq!(
            s.range(Included(&25), Excluded(&80))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![30, 40, 50, 60, 70]
        );
        assert_eq!(
            s.range(Excluded(&25), Included(&80))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![30, 40, 50, 60, 70, 80]
        );
        assert_eq!(
            s.range(Excluded(&25), Excluded(&80))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![30, 40, 50, 60, 70]
        );

        assert_eq!(
            s.range(Included(&25), Included(&25))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![]
        );
        assert_eq!(
            s.range(Included(&25), Excluded(&25))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![]
        );
        assert_eq!(
            s.range(Excluded(&25), Included(&25))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![]
        );
        assert_eq!(
            s.range(Excluded(&25), Excluded(&25))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![]
        );

        assert_eq!(
            s.range(Included(&50), Included(&50))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![50]
        );
        assert_eq!(
            s.range(Included(&50), Excluded(&50))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![]
        );
        assert_eq!(
            s.range(Excluded(&50), Included(&50))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![]
        );
        assert_eq!(
            s.range(Excluded(&50), Excluded(&50))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![]
        );

        assert_eq!(
            s.range(Included(&100), Included(&-2))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![]
        );
        assert_eq!(
            s.range(Included(&100), Excluded(&-2))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![]
        );
        assert_eq!(
            s.range(Excluded(&100), Included(&-2))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![]
        );
        assert_eq!(
            s.range(Excluded(&100), Excluded(&-2))
                .map(|x| *x.value())
                .collect::<Vec<_>>(),
            vec![]
        );
    }
}
