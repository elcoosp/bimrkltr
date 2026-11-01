//! # bimrkltr
//!
//! High-performance, compile-time sized Merkle trees for Rust, with
//! bidirectional tree support and generic hash functions.
//!
//! The crate exposes two complementary data structures:
//!
//! - [`MerkleTree`]: a *complete* binary tree whose shape is fixed at
//!   compile time by a `const HEIGHT` generic — `2^HEIGHT` leaves, zero
//!   runtime sizing overhead, `O(log n)` updates and proofs.
//! - [`JellyfishMerkleTree`]: a *sparse* tree over an arbitrary key set
//!   with membership and non-membership proofs, versioned storage
//!   ([`VersionedJMTStorage`]) and a pluggable persistence layer
//!   ([`JMTDatabase`]).
//!
//! Both share the [`Hasher`] abstraction: implement it once for your
//! favorite hash function (or use the provided [`Sha256Hasher`]) and every
//! tree, proof and storage structure works with it.
//!
//! [`BidirectionalMerkleTree`] layers hierarchical commitments on top of
//! [`MerkleTree`]: child trees linked under main-tree leaves for cross-chain
//! bridges and layered state commitments.
//!
//! The crate is `no_std`-compatible (needs `alloc` only).
//!
//! # Quick start
//!
//! ```
//! use bimrkltr::{MerkleTree, Sha256Hasher};
//!
//! type Tree8 = MerkleTree<3, Sha256Hasher>; // 2^3 = 8 leaves
//!
//! let leaves: Vec<[u8; 32]> = (0..8).map(|i| [i; 32]).collect();
//! let tree = Tree8::from_leaves(&leaves);
//!
//! // Inclusion proof for leaf 5.
//! let proof = tree.prove(5);
//! assert!(proof.verify(&leaves[5], tree.root()));
//!
//! // Update a leaf; the root follows in O(log n).
//! let mut tree = tree;
//! tree.update_leaf(5, [42_u8; 32]);
//! assert!(tree.prove(5).verify(&[42_u8; 32], tree.root()));
//! ```
//!
//! # Sparse tree over arbitrary keys
//!
//! ```
//! use bimrkltr::{JellyfishMerkleTree, Sha256Hasher};
//!
//! let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(0);
//! tree.insert(b"alice", b"100").unwrap();
//! tree.insert(b"bob", b"200").unwrap();
//!
//! // Membership proof.
//! assert!(tree.prove(b"alice").verify(tree.root(), b"alice", Some(b"100")));
//!
//! // Non-membership proof for an absent key.
//! assert!(tree.prove(b"carol").verify(tree.root(), b"carol", None::<&[u8]>));
//! ```
//!
//! # Choosing a hasher
//!
//! A hasher must domain-separate leaves from nodes (see [`Hasher`]). The
//! provided [`Sha256Hasher`] does; [`TestHasher`] is a predictable toy for
//! tests only.

#![cfg_attr(not(test), no_std)]
extern crate alloc;

mod bidirectional;
mod error;
mod hasher;
mod jmt;
mod proof;
mod tree;

pub use crate::bidirectional::BidirectionalMerkleTree;
pub use crate::error::JMTError;
pub use crate::hasher::{Hasher, Sha256Hasher, TestHasher};
pub use crate::jmt::{
    JMTDatabase, JMTDelta, JMTProof, JellyfishMerkleTree, MemoryJMTDatabase, VersionedJMTStorage,
};
pub use crate::proof::MerkleProof;
pub use crate::tree::MerkleTree;

/// One-stop import for the most frequently used items.
pub mod prelude {
    pub use crate::{
        BidirectionalMerkleTree, Hasher, JMTDatabase, JMTProof, JellyfishMerkleTree,
        MemoryJMTDatabase, MerkleProof, MerkleTree, Sha256Hasher, TestHasher, VersionedJMTStorage,
    };
}

#[cfg(test)]
mod tests {
    use crate::JellyfishMerkleTree;
    use crate::prelude::*;
    use insta::assert_snapshot;

    fn assert_hasher<H: Hasher>() {}

    #[test]
    fn prelude_imports_resolve() {
        // Every prelude item is usable through the single glob import.
        let tree = MerkleTree::<1, TestHasher>::from_leaves(&[[1_u8; 32], [2_u8; 32]]);
        let proof = tree.prove(0);
        assert!(proof.verify(&[1_u8; 32], tree.root()));

        let mut sparse = JellyfishMerkleTree::<Sha256Hasher>::new(0);
        sparse.insert(b"k", b"v").unwrap();
        assert!(sparse.validate());

        let mut bi = BidirectionalMerkleTree::<1, 2, TestHasher>::new();
        bi.link_child_tree(0, tree);
        assert_eq!(bi.child_tree_count(), 1);

        let db = MemoryJMTDatabase::<Sha256Hasher>::new();
        assert_eq!(db.get_version().unwrap(), 0);
        let storage = VersionedJMTStorage::<Sha256Hasher>::new();
        assert_eq!(storage.version_count(), 0);
        assert_hasher::<Sha256Hasher>();
        assert_hasher::<TestHasher>();
    }

    #[test]
    fn display_snapshots() {
        let tree = MerkleTree::<2, TestHasher>::from_leaves(&[
            [1_u8; 32], [2_u8; 32], [3_u8; 32], [4_u8; 32],
        ]);
        assert_snapshot!("merkle_tree_display", tree.to_string());
        assert_snapshot!("merkle_proof_display", tree.prove(1).to_string());

        let mut bi = BidirectionalMerkleTree::<2, 4, TestHasher>::new();
        assert_snapshot!("empty_bidirectional_tree_display", bi.to_string());
        bi.link_child_tree(
            0,
            MerkleTree::<2, TestHasher>::from_leaves(&[[5_u8; 32]; 4]),
        );
        assert_snapshot!("bidirectional_tree_display", bi.to_string());
    }

    #[test]
    fn jmt_display_snapshots() {
        let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(1);
        tree.insert(b"key", b"value").unwrap();
        assert_snapshot!("jmt_tree_display", tree.to_string());
        assert_snapshot!("jmt_proof_display", tree.prove(b"key").to_string());
    }
}
