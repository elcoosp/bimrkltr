//! Tests for the hierarchical [`BidirectionalMerkleTree`].

use crate::bidirectional::BidirectionalMerkleTree;
use crate::hasher::{Hasher, TestHasher};
use crate::tree::MerkleTree;

/// Leaf `i` is filled with `start + i`.
fn leaves(start: u8) -> [[u8; 32]; 4] {
    core::array::from_fn(|i| [start.wrapping_add(u8::try_from(i).unwrap_or(0)); 32])
}

#[test]
fn new_tree_has_no_children() {
    let bi = BidirectionalMerkleTree::<2, 4, TestHasher>::new();
    assert_eq!(bi.child_tree_count(), 0);
    assert!(bi.child_tree(0).is_none());
    assert!(bi.child_root(0).is_none());
    assert_eq!(bi.linked_children().count(), 0);
    assert_eq!(bi.main_tree().root(), [0_u8; 32]);
}

#[test]
fn default_matches_new() {
    assert_eq!(
        BidirectionalMerkleTree::<2, 4, TestHasher>::default(),
        BidirectionalMerkleTree::<2, 4, TestHasher>::new()
    );
}

#[test]
fn linking_commits_child_root_into_main_leaf() {
    let child = MerkleTree::<2, TestHasher>::from_leaves(&leaves(5));
    let child_root = child.root();

    let mut bi = BidirectionalMerkleTree::<2, 4, TestHasher>::new();
    bi.link_child_tree(0, child);

    assert_eq!(bi.child_tree_count(), 1);
    assert_eq!(bi.child_root(0), Some(child_root));
    assert_eq!(bi.child_tree(0).unwrap().root(), child_root);
    // The main leaf 0 commits to the child root bytes.
    assert_eq!(
        bi.main_tree().get_leaf(0),
        TestHasher::hash_leaf(child_root.as_ref())
    );
    assert_eq!(
        bi.linked_children().map(|(i, _)| i).collect::<Vec<_>>(),
        vec![0]
    );
}

#[test]
fn relinking_replaces_the_child() {
    let first = MerkleTree::<2, TestHasher>::from_leaves(&leaves(1));
    let second = MerkleTree::<2, TestHasher>::from_leaves(&leaves(2));

    let second_root = second.root();
    let mut bi = BidirectionalMerkleTree::<2, 4, TestHasher>::new();
    bi.link_child_tree(1, first);
    let first_root = bi.child_root(1);
    bi.link_child_tree(1, second);

    assert_eq!(bi.child_tree_count(), 1);
    assert_ne!(bi.child_root(1), first_root);
    assert_eq!(bi.child_root(1), Some(second_root));
}

#[test]
fn unlinking_removes_but_keeps_the_commitment() {
    let child = MerkleTree::<2, TestHasher>::from_leaves(&leaves(5));

    let mut bi = BidirectionalMerkleTree::<2, 4, TestHasher>::new();
    bi.link_child_tree(2, child);
    assert!(bi.unlink_child_tree(2));
    assert!(!bi.unlink_child_tree(2));
    assert!(bi.child_tree(2).is_none());
    // The commitment hash cannot be reversed.
    assert_ne!(bi.main_tree().get_leaf(2), [0_u8; 32]);
}

#[test]
fn bidirectional_paths_verify() {
    let child_leaves = leaves(5);

    let mut bi = BidirectionalMerkleTree::<2, 4, TestHasher>::new();
    bi.link_child_tree(0, MerkleTree::<2, TestHasher>::from_leaves(&child_leaves));

    // The main leaf preimage is the child root; the child leaf preimage is
    // the original child leaf payload.
    let child_root = bi.child_root(0).unwrap();
    assert!(bi.verify_bidirectional_path(0, 3, &child_leaves[3]));

    // Unlinked main leaf.
    assert!(!bi.verify_bidirectional_path(1, 0, &child_leaves[0]));
    // Wrong child payload.
    assert!(!bi.verify_bidirectional_path(0, 0, &[99_u8; 32]));
    // Child index out of bounds.
    assert!(!bi.verify_bidirectional_path(0, 9, &child_leaves[0]));
    // Main index out of bounds.
    assert!(!bi.verify_bidirectional_path(7, 0, &child_leaves[0]));
    // Tampered main commitment: rehash main leaf 0 to something else.
    bi.main_tree_mut().update_leaf(0, [1_u8; 32]);
    assert!(!bi.verify_bidirectional_path(0, 3, &child_leaves[3]));
    let _ = child_root;
}

#[test]
fn multiple_children_are_tracked() {
    let mut bi = BidirectionalMerkleTree::<2, 4, TestHasher>::new();
    for i in 0_u8..3 {
        bi.link_child_tree(
            i as usize,
            MerkleTree::<2, TestHasher>::from_leaves(&leaves(10 * (i + 1))),
        );
    }
    assert_eq!(bi.child_tree_count(), 3);
    for i in 0..3 {
        assert!(bi.child_root(i).is_some());
        assert!(bi.child_tree(i).is_some());
    }
    // Slot 3 exists (MAX_CHILDREN = 4) but is empty.
    assert!(bi.child_root(3).is_none());
}

#[test]
#[should_panic(expected = "child slot 4 out of bounds")]
fn linking_beyond_max_children_panics() {
    let mut bi = BidirectionalMerkleTree::<2, 4, TestHasher>::new();
    bi.link_child_tree(4, MerkleTree::<2, TestHasher>::from_leaves(&leaves(1)));
}

#[test]
#[should_panic(expected = "child slot 9 out of bounds")]
fn unlinking_beyond_max_children_panics() {
    let mut bi = BidirectionalMerkleTree::<2, 4, TestHasher>::new();
    bi.unlink_child_tree(9);
}

#[test]
fn display_is_human_readable() {
    let mut bi = BidirectionalMerkleTree::<2, 4, TestHasher>::new();
    bi.link_child_tree(1, MerkleTree::<2, TestHasher>::from_leaves(&leaves(3)));

    let display = bi.to_string();
    assert!(display.contains("Bidirectional Merkle Tree"));
    assert!(display.contains("Main Tree:"));
    assert!(display.contains("Child Trees: 1/4"));
    assert!(display.contains("Leaf 1 → Child Root:"));
    assert!(display.contains("Height: 2, Max Children: 4"));

    let debug = format!("{bi:?}");
    assert!(debug.contains("BidirectionalMerkleTree"));
}

#[test]
fn serde_roundtrips() {
    let mut bi = BidirectionalMerkleTree::<2, 3, TestHasher>::new();
    bi.link_child_tree(0, MerkleTree::<2, TestHasher>::from_leaves(&leaves(1)));
    bi.link_child_tree(2, MerkleTree::<2, TestHasher>::from_leaves(&leaves(2)));

    let encoded = postcard::to_allocvec(&bi).unwrap();
    let decoded: BidirectionalMerkleTree<2, 3, TestHasher> =
        postcard::from_bytes(&encoded).unwrap();
    assert_eq!(decoded, bi);
    assert_eq!(decoded.child_tree_count(), 2);
}

#[test]
fn serde_rejects_wrong_slot_count() {
    let mut bi = BidirectionalMerkleTree::<2, 4, TestHasher>::new();
    bi.link_child_tree(0, MerkleTree::<2, TestHasher>::from_leaves(&leaves(1)));
    let encoded = postcard::to_allocvec(&bi).unwrap();
    // Serialized with 4 slots; loading as a 2-slot tree must fail.
    let result = postcard::from_bytes::<BidirectionalMerkleTree<2, 2, TestHasher>>(&encoded);
    assert!(result.is_err());
}

#[test]
fn serde_error_path_for_mismatched_slot_count() {
    // Serialize a tree with MAX_CHILDREN = 2 and load it as MAX_CHILDREN = 3.
    let mut bi = BidirectionalMerkleTree::<2, 2, TestHasher>::new();
    bi.link_child_tree(1, MerkleTree::<2, TestHasher>::from_leaves(&leaves(4)));
    let encoded = postcard::to_allocvec(&bi).unwrap();
    assert!(postcard::from_bytes::<BidirectionalMerkleTree<2, 3, TestHasher>>(&encoded).is_err());
}

#[test]
fn serde_roundtrip_without_children() {
    let bi = BidirectionalMerkleTree::<2, 2, TestHasher>::new();
    let encoded = postcard::to_allocvec(&bi).unwrap();
    let decoded: BidirectionalMerkleTree<2, 2, TestHasher> =
        postcard::from_bytes(&encoded).unwrap();
    assert_eq!(decoded, bi);
    assert_eq!(decoded.child_tree_count(), 0);
}
