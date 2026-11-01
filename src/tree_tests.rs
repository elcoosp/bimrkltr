//! Tests for the compile-time sized [`MerkleTree`] and its proofs.

use alloc::vec;
use alloc::vec::Vec;

use crate::hasher::{Hasher, Sha256Hasher, TestHasher};
use crate::proof::MerkleProof;
use crate::tree::MerkleTree;

/// Build deterministic test leaves: leaf `i` is filled with `start + i`.
fn leaves<const N: usize>(start: u8) -> [[u8; 32]; N] {
    core::array::from_fn(|i| [start.wrapping_add(u8::try_from(i).unwrap_or(0)); 32])
}

/// Independent oracle: recursive divide-and-conquer root over the leaves,
/// sharing no code with the level-order array construction.
fn oracle_root<const HEIGHT: usize, H: Hasher>(leaf_hashes: &[H::Output]) -> H::Output {
    fn fold<H: Hasher>(slice: &[H::Output]) -> H::Output {
        if slice.len() == 1 {
            return slice[0];
        }
        let mid = slice.len() / 2;
        H::hash_node(&fold::<H>(&slice[..mid]), &fold::<H>(&slice[mid..]))
    }
    fold::<H>(leaf_hashes)
}

#[test]
fn construction_matches_independent_oracle() {
    for start in [1_u8, 42, 200] {
        let data = leaves::<8>(start);
        let tree = MerkleTree::<3, TestHasher>::from_leaves(&data);
        let leaf_hashes: Vec<_> = data.iter().map(|p| TestHasher::hash_leaf(p)).collect();
        assert_eq!(tree.root(), oracle_root::<3, TestHasher>(&leaf_hashes));
        assert_eq!(tree.height(), 3);
        assert_eq!(tree.leaf_count(), 8);
        assert_eq!(tree.leaf_hashes().len(), 8);
        assert_eq!(tree.leaf_hashes()[2], tree.get_leaf(2));
    }
}

#[test]
fn single_node_tree_is_its_own_root() {
    type Tree = MerkleTree<0, TestHasher>;
    let data = [[7_u8; 32]];
    let tree = Tree::from_leaves(&data);
    assert_eq!(tree.leaf_count(), 1);
    assert_eq!(tree.height(), 0);
    assert_eq!(tree.root(), TestHasher::hash_leaf(&[7_u8; 32]));
    assert_eq!(tree.get_leaf(0), tree.root());
    assert_eq!(tree.leaf_hashes(), &[tree.root()]);
}

#[test]
fn every_leaf_has_a_verifiable_proof() {
    let data = leaves::<16>(9);
    let tree = MerkleTree::<4, Sha256Hasher>::from_leaves(&data);
    for (i, payload) in data.iter().enumerate() {
        let proof = tree.prove(i);
        assert_eq!(proof.leaf_index(), i);
        assert_eq!(proof.path_length(), 4);
        assert_eq!(proof.path().len(), 4);
        assert!(proof.verify(payload, tree.root()), "proof for leaf {i}");
    }
}

#[test]
fn tampered_proofs_and_payloads_fail() {
    let data = leaves::<8>(3);
    let tree = MerkleTree::<3, TestHasher>::from_leaves(&data);

    // Wrong payload.
    let proof = tree.prove(3);
    assert!(!proof.verify(&[99_u8; 32], tree.root()));

    // Wrong root.
    assert!(!proof.verify(&data[3], TestHasher::hash_leaf(b"other")));

    // Wrong sibling hash at the deepest level.
    let mut path = [TestHasher::hash_leaf(&[0; 32]); 3];
    path.copy_from_slice(proof.path());
    path[0] = TestHasher::hash_leaf(b"evil");
    let forged = MerkleProof::<3, TestHasher>::from_parts(3, path);
    assert!(!forged.verify(&data[3], tree.root()));

    // Proof for another index does not verify for this payload.
    let other = tree.prove(0);
    assert!(!other.verify(&data[3], tree.root()));
}

#[test]
fn updates_refresh_root_and_proofs() {
    let data = leaves::<8>(1);
    let mut tree = MerkleTree::<3, TestHasher>::from_leaves(&data);
    let root_before = tree.root();

    tree.update_leaf(5, [77_u8; 32]);
    assert_ne!(tree.root(), root_before);
    assert_eq!(tree.get_leaf(5), TestHasher::hash_leaf(&[77_u8; 32]));
    assert!(tree.prove(5).verify(&[77_u8; 32], tree.root()));

    // Every other proof still verifies.
    for i in [0_usize, 1, 2, 3, 4, 6, 7] {
        assert!(tree.prove(i).verify(&data[i], tree.root()));
    }
}

#[test]
fn batch_updates_match_sequential_updates() {
    let data = leaves::<16>(2);
    let updates = [(0_usize, 90_u8), (5, 91), (5, 92), (15, 93)];

    let mut sequential = MerkleTree::<4, TestHasher>::from_leaves(&data);
    for &(i, byte) in &updates {
        sequential.update_leaf(i, [byte; 32]);
    }

    let mut batch = MerkleTree::<4, TestHasher>::from_leaves(&data);
    batch.update_leaves(
        &updates
            .iter()
            .map(|&(i, byte)| (i, [byte; 32]))
            .collect::<Vec<_>>(),
    );

    assert_eq!(sequential.root(), batch.root(), "later entries must win");
    assert_eq!(batch.get_leaf(5), TestHasher::hash_leaf(&[92_u8; 32]));

    // The batched tree is fully consistent.
    let expected = {
        let mut data = data;
        for &(i, byte) in &updates {
            data[i] = [byte; 32];
        }
        MerkleTree::<4, TestHasher>::from_leaves(&data)
    };
    assert_eq!(batch.root(), expected.root());
}

#[test]
fn zero_height_updates_replace_the_root() {
    type Tree = MerkleTree<0, TestHasher>;
    let mut tree = Tree::from_leaves(&[[1_u8; 32]]);
    tree.update_leaf(0, [2_u8; 32]);
    assert_eq!(tree.root(), TestHasher::hash_leaf(&[2_u8; 32]));
    tree.update_leaves(&[(0, [3_u8; 32])]);
    assert_eq!(tree.root(), TestHasher::hash_leaf(&[3_u8; 32]));
}

#[test]
fn default_tree_is_zero_filled() {
    let tree = MerkleTree::<2, TestHasher>::default();
    assert_eq!(tree.root(), [0_u8; 32]);
    assert_eq!(tree.leaf_count(), 4);
}

#[test]
#[should_panic(expected = "expected exactly 4 leaves")]
fn from_leaves_rejects_wrong_count() {
    let _ = MerkleTree::<2, TestHasher>::from_leaves(&[[0_u8; 32]; 3]);
}

#[test]
#[should_panic(expected = "leaf index 4 out of bounds")]
fn get_leaf_rejects_out_of_bounds() {
    let _ = MerkleTree::<2, TestHasher>::from_leaves(&[[0_u8; 32]; 4]).get_leaf(4);
}

#[test]
#[should_panic(expected = "leaf index 9 out of bounds")]
fn update_leaf_rejects_out_of_bounds() {
    MerkleTree::<2, TestHasher>::from_leaves(&[[0_u8; 32]; 4]).update_leaf(9, [1_u8; 32]);
}

#[test]
#[should_panic(expected = "leaf index 4 out of bounds")]
fn prove_rejects_out_of_bounds() {
    let _ = MerkleTree::<2, TestHasher>::from_leaves(&[[0_u8; 32]; 4]).prove(4);
}

#[test]
fn display_is_human_readable() {
    let tree = MerkleTree::<2, TestHasher>::from_leaves(&leaves::<4>(1));
    let display = tree.to_string();
    assert!(display.contains("Merkle Tree (height: 2)"));
    assert!(display.contains("Root:"));
    assert!(display.contains("Leaves: 4"));
    assert!(display.contains("Total Nodes: 7"));

    let debug = format!("{tree:?}");
    assert!(debug.contains("MerkleTree"));
}

#[test]
fn serde_roundtrips_tree_and_proof() {
    let data = leaves::<8>(5);
    let tree = MerkleTree::<3, Sha256Hasher>::from_leaves(&data);
    let proof = tree.prove(2);

    let encoded = postcard::to_allocvec(&tree).unwrap();
    let decoded: MerkleTree<3, Sha256Hasher> = postcard::from_bytes(&encoded).unwrap();
    assert_eq!(decoded, tree);
    assert!(decoded.prove(2).verify(&data[2], decoded.root()));

    let encoded = postcard::to_allocvec(&proof).unwrap();
    let decoded: MerkleProof<3, Sha256Hasher> = postcard::from_bytes(&encoded).unwrap();
    assert_eq!(decoded, proof);
    assert!(decoded.verify(&data[2], tree.root()));
}

#[test]
fn proof_deserialization_rejects_wrong_path_length() {
    // HEIGHT = 2 expects exactly 2 sibling hashes; serialize a 3-deep proof
    // and try to load it as a 2-deep one.
    let deep = MerkleTree::<3, Sha256Hasher>::from_leaves(&leaves::<8>(1)).prove(0);
    let encoded = postcard::to_allocvec(&deep).unwrap();
    let result = postcard::from_bytes::<MerkleProof<2, Sha256Hasher>>(&encoded);
    assert!(result.is_err());
}

fn clone_via_generic<T: Clone>(value: &T) -> T {
    value.clone()
}

#[test]
fn proof_is_copy_and_clone() {
    let tree = MerkleTree::<2, TestHasher>::from_leaves(&leaves::<4>(1));
    let proof = tree.prove(1);
    let copied: MerkleProof<2, TestHasher> = proof; // Copy semantics
    let cloned = clone_via_generic(&proof);
    assert_eq!(copied, proof);
    assert_eq!(cloned, proof);
}

#[test]
fn clone_is_independent() {
    let mut tree = MerkleTree::<2, TestHasher>::from_leaves(&leaves::<4>(1));
    let mut clone = tree.clone();
    tree.update_leaf(0, [7_u8; 32]);
    assert_ne!(tree.root(), clone.root());
    clone.update_leaf(0, [7_u8; 32]);
    assert_eq!(tree.root(), clone.root());
}

#[test]
fn generic_leaf_types_are_accepted() {
    // Vec<u8>, &[u8] and [u8; N] all implement AsRef<[u8]>.
    let vecs: Vec<Vec<u8>> = vec![vec![1_u8; 4], vec![2_u8; 4]];
    let a = MerkleTree::<1, TestHasher>::from_leaves(&vecs);

    let slices: [&[u8]; 2] = [&[1_u8; 4], &[2_u8; 4]];
    let b = MerkleTree::<1, TestHasher>::from_leaves(&slices);

    let arrays: [[u8; 4]; 2] = [[1_u8; 4], [2_u8; 4]];
    let c = MerkleTree::<1, TestHasher>::from_leaves(&arrays);

    assert_eq!(a.root(), b.root());
    assert_eq!(b.root(), c.root());
}

#[test]
fn zero_height_tree_roundtrips_through_serde() {
    type Tree = MerkleTree<0, Sha256Hasher>;
    let tree = Tree::from_leaves(&[[3_u8; 32]]);
    let encoded = postcard::to_allocvec(&tree).unwrap();
    let decoded: Tree = postcard::from_bytes(&encoded).unwrap();
    assert_eq!(decoded, tree);
    assert!(decoded.prove(0).verify(&[3_u8; 32], decoded.root()));
}
