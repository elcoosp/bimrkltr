//! Generic [`Hasher`] abstraction and ready-made implementations.
//!
//! The crate is hashing-agnostic: implement [`Hasher`] for your own
//! algorithm, or use the provided [`Sha256Hasher`] (backed by the `sha2`
//! crate) or the toy [`TestHasher`].

use core::fmt::Debug;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Domain separation for hashing algorithms usable by this crate.
///
/// A [`Hasher`] maps raw byte payloads to fixed-size hashes. Two roles are
/// fundamental:
///
/// - [`Hasher::hash_leaf`] hashes *leaf payloads* (arbitrary bytes),
/// - [`Hasher::hash_node`] combines two child hashes into a parent hash.
///
/// Every other hashing scheme used by the crate (key commitments, internal
/// nodes with more than two children, empty subtrees, keyed leaves) is
/// *derived* from those two primitives through the provided methods, so a
/// correct, collision-resistant implementation of the two primitives yields
/// a correct implementation of every tree in this crate.
///
/// # Domain separation
///
/// Implementations **must** domain-separate leaves from nodes (i.e.
/// `hash_leaf(x) == hash_node(x, y)` must be infeasible to arrange);
/// otherwise proofs could be forged by reinterpreting leaves as nodes. The
/// provided [`Sha256Hasher`] prefixes payloads with distinct domain tags.
pub trait Hasher: Clone + Default + Debug + PartialEq + Eq + 'static {
    /// Fixed-size hash produced by this hasher (e.g. `[u8; 32]` for SHA-256).
    type Output: Copy
        + Clone
        + Debug
        + Default
        + PartialEq
        + Eq
        + AsRef<[u8]>
        + Serialize
        + for<'de> Deserialize<'de>;

    /// Hash an arbitrary leaf payload.
    fn hash_leaf(data: &[u8]) -> Self::Output;

    /// Combine two child hashes into their parent hash.
    fn hash_node(left: &Self::Output, right: &Self::Output) -> Self::Output;

    /// Hash of the empty subtree.
    ///
    /// Defaults to hashing the empty byte string as a leaf.
    #[must_use]
    fn empty_hash() -> Self::Output {
        Self::hash_leaf(&[])
    }

    /// Commitment of a key, used to derive its position in the sparse tree.
    ///
    /// Defaults to hashing the key as a leaf payload. Must be
    /// collision-resistant for the sparse tree guarantees to hold.
    #[must_use]
    fn commit_key(key: &[u8]) -> Self::Output {
        Self::hash_leaf(key)
    }

    /// Hash a leaf bound to a specific key.
    ///
    /// Defaults to hashing the key commitment and the value hash together as
    /// a node, which binds the leaf to its key.
    #[must_use]
    fn hash_leaf_with_key(key: &[u8], data: &[u8]) -> Self::Output {
        let commitment = Self::commit_key(key);
        let value = Self::hash_leaf(data);
        Self::hash_node(&commitment, &value)
    }

    /// Hash an internal node with an arbitrary number of children.
    ///
    /// Defaults to a left-to-right fold of [`Hasher::hash_node`]; an empty
    /// slice hashes to [`Hasher::empty_hash`], a single child hashes to the
    /// child itself (sparse promotion).
    #[must_use]
    fn hash_internal_node(children: &[Self::Output]) -> Self::Output {
        let mut children = children.iter();
        let Some(first) = children.next() else {
            return Self::empty_hash();
        };
        children.fold(*first, |acc, next| Self::hash_node(&acc, next))
    }
}

/// Production-grade SHA-256 hasher with domain-separated roles.
///
/// Leaf payloads are prefixed with `bimrkltr:leaf:` and node inputs with
/// `bimrkltr:node:`, which makes leaf/node confusion attacks infeasible.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Sha256Hasher;

impl Hasher for Sha256Hasher {
    type Output = [u8; 32];

    fn hash_leaf(data: &[u8]) -> Self::Output {
        let mut hasher = Sha256::new();
        hasher.update(b"bimrkltr:leaf:");
        hasher.update(data);
        hasher.finalize().into()
    }

    fn hash_node(left: &Self::Output, right: &Self::Output) -> Self::Output {
        let mut hasher = Sha256::new();
        hasher.update(b"bimrkltr:node:");
        hasher.update(left);
        hasher.update(right);
        hasher.finalize().into()
    }
}

/// Toy additive hasher with predictable results, for tests and examples
/// only. It is *not* collision-resistant and must never be used in
/// production.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TestHasher;

impl Hasher for TestHasher {
    type Output = [u8; 32];

    fn hash_leaf(data: &[u8]) -> Self::Output {
        let mut result = [0_u8; 32];
        let sum = data.iter().fold(0_u8, |acc, &x| acc.wrapping_add(x));
        result[0] = sum;
        result[1] = sum.wrapping_mul(2);
        result
    }

    fn hash_node(left: &Self::Output, right: &Self::Output) -> Self::Output {
        let mut result = [0_u8; 32];
        for (slot, (l, r)) in result.iter_mut().zip(left.iter().zip(right.iter())) {
            *slot = l.wrapping_add(*r);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_hasher_domains_are_separated() {
        let leaf = Sha256Hasher::hash_leaf(b"payload");
        let node = Sha256Hasher::hash_node(&leaf, &Sha256Hasher::empty_hash());
        assert_ne!(leaf, node);
    }

    #[test]
    fn sha256_hasher_matches_reference_vectors() {
        // Reference value computed with an independent SHA-256 tool.
        assert_eq!(
            Sha256Hasher::hash_leaf(b"abc"),
            [
                0x0c, 0x3d, 0x68, 0x0d, 0x42, 0x6f, 0xc7, 0x15, 0x68, 0x57, 0xa1, 0x6d, 0x11, 0x11,
                0xd6, 0xbc, 0x70, 0x7f, 0x3d, 0xfc, 0x6c, 0xdc, 0xf7, 0x20, 0x5d, 0x2c, 0xa1, 0x20,
                0xd8, 0x73, 0x7c, 0x86,
            ]
        );
    }

    #[test]
    fn derived_defaults_are_consistent() {
        // empty_hash = hash_leaf(&[])
        assert_eq!(TestHasher::empty_hash(), TestHasher::hash_leaf(&[]));
        // commit_key = hash_leaf(key)
        assert_eq!(TestHasher::commit_key(b"k"), TestHasher::hash_leaf(b"k"));
        // hash_leaf_with_key = hash_node(commit_key, hash_leaf)
        let expected =
            TestHasher::hash_node(&TestHasher::commit_key(b"k"), &TestHasher::hash_leaf(b"v"));
        assert_eq!(TestHasher::hash_leaf_with_key(b"k", b"v"), expected);
    }

    #[test]
    fn hash_internal_node_folds_left_to_right() {
        let a = [1_u8; 32];
        let b = [2_u8; 32];
        let c = [3_u8; 32];
        // Empty children -> empty hash.
        assert_eq!(
            TestHasher::hash_internal_node(&[]),
            TestHasher::empty_hash()
        );
        // Single child -> the child itself (sparse promotion).
        assert_eq!(TestHasher::hash_internal_node(&[a]), a);
        // Two children -> hash_node.
        assert_eq!(
            TestHasher::hash_internal_node(&[a, b]),
            TestHasher::hash_node(&a, &b)
        );
        // More children fold strictly left-to-right.
        let ab = TestHasher::hash_node(&a, &b);
        assert_eq!(
            TestHasher::hash_internal_node(&[a, b, c]),
            TestHasher::hash_node(&ab, &c)
        );
    }
}
