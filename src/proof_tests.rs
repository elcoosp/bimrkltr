//! Tests for [`MerkleProof`] specifics (the happy paths are exercised
//! through `MerkleTree::prove` in the tree test module).

use crate::hasher::{Sha256Hasher, TestHasher};
use crate::proof::MerkleProof;
use crate::tree::MerkleTree;

#[test]
fn from_parts_matches_prove() {
    let leaves = [[1_u8; 32], [2_u8; 32], [3_u8; 32], [4_u8; 32]];
    let tree = MerkleTree::<2, TestHasher>::from_leaves(&leaves);
    let proven = tree.prove(1);
    let path: [_; 2] = proven.path().try_into().unwrap();
    let manual = MerkleProof::<2, TestHasher>::from_parts(1, path);
    assert_eq!(proven, manual);
    assert!(manual.verify(&leaves[1], tree.root()));
    assert_eq!(manual.leaf_index(), 1);
    assert_eq!(manual.path_length(), 2);
}

#[test]
fn verify_uses_leaf_index_parity_for_direction() {
    // Leaf 0 (even index, left child) and leaf 1 (odd index, right child)
    // must produce different roots when combined with the same sibling.
    let leaves = [[1_u8; 32], [2_u8; 32]];
    let tree = MerkleTree::<1, TestHasher>::from_leaves(&leaves);

    let left = tree.prove(0);
    let right = tree.prove(1);
    assert!(left.verify(&leaves[0], tree.root()));
    assert!(right.verify(&leaves[1], tree.root()));

    // Swapping the indices breaks both.
    assert!(!left.verify(&leaves[1], tree.root()));
    assert!(!right.verify(&leaves[0], tree.root()));
}

#[test]
fn zero_height_proof_verifies_trivially() {
    let tree = MerkleTree::<0, TestHasher>::from_leaves(&[[1_u8; 32]]);
    let proof = tree.prove(0);
    assert_eq!(proof.path_length(), 0);
    assert_eq!(proof.path().len(), 0);
    assert!(proof.verify(&[1_u8; 32], tree.root()));
}

#[test]
fn display_is_human_readable() {
    let tree =
        MerkleTree::<2, TestHasher>::from_leaves(&[[1_u8; 32], [2_u8; 32], [3_u8; 32], [4_u8; 32]]);
    let proof = tree.prove(1);
    let display = proof.to_string();
    assert!(display.contains("Merkle Proof"));
    assert!(display.contains("Leaf Index: 1"));
    assert!(display.contains("Path Length: 2"));
    assert!(display.contains("Level 0:"));

    let debug = format!("{proof:?}");
    assert!(debug.contains("MerkleProof"));
}

#[test]
fn serde_matrix_covers_success_and_failure_per_height() {
    let leaves = [[1_u8; 32], [2_u8; 32], [3_u8; 32], [4_u8; 32]];
    let tree = MerkleTree::<2, Sha256Hasher>::from_leaves(&leaves);

    // HEIGHT = 2: successful roundtrip…
    let proof = tree.prove(1);
    let encoded = postcard::to_allocvec(&proof).unwrap();
    let decoded: MerkleProof<2, Sha256Hasher> = postcard::from_bytes(&encoded).unwrap();
    assert_eq!(decoded, proof);
    assert!(decoded.verify(&leaves[1], tree.root()));

    // …and rejection of a mismatched path length.
    let deep = MerkleTree::<3, Sha256Hasher>::from_leaves(&[
        [1_u8; 32], [2_u8; 32], [3_u8; 32], [4_u8; 32], [5_u8; 32], [6_u8; 32], [7_u8; 32],
        [8_u8; 32],
    ])
    .prove(0);
    let deep_encoded = postcard::to_allocvec(&deep).unwrap();
    assert!(postcard::from_bytes::<MerkleProof<2, Sha256Hasher>>(&deep_encoded).is_err());

    // HEIGHT = 3: successful roundtrip…
    let deep_decoded: MerkleProof<3, Sha256Hasher> = postcard::from_bytes(&deep_encoded).unwrap();
    assert_eq!(deep_decoded, deep);

    // …and rejection of a too-short path.
    assert!(postcard::from_bytes::<MerkleProof<3, Sha256Hasher>>(&encoded).is_err());
}
