use std::{
    borrow::Borrow,
    cmp, fmt,
    fmt::Debug,
    marker::PhantomData,
    ops::{Deref, Index},
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
                            result.found = Some(c as *const Node<K, V> as *mut Node<K, V>)
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
        let list = SkipList::<i32, i32>::new();
        assert!(!list.contain_key(&10));
        assert!(list.is_empty());
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
}
