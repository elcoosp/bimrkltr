//! Tests for the sparse [`JellyfishMerkleTree`] and its ecosystem.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use super::{
    JMTDatabase, JMTDelta, JMTError, JMTProof, JellyfishMerkleTree, MemoryJMTDatabase,
    VersionedJMTStorage,
};
use crate::hasher::{Hasher, Sha256Hasher, TestHasher};

/// Hasher with a 4-byte output and *controllable* key commitments: the
/// position of a single-byte key is `0x000000kk`, which lets tests place
/// leaves at chosen bit relationships.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TinyHasher;

impl Hasher for TinyHasher {
    type Output = [u8; 4];

    fn hash_leaf(data: &[u8]) -> Self::Output {
        let mut out = [0_u8; 4];
        for (i, byte) in data.iter().enumerate() {
            out[i % 4] = out[i % 4].wrapping_add(*byte);
        }
        out
    }

    fn hash_node(left: &Self::Output, right: &Self::Output) -> Self::Output {
        let mut out = [0_u8; 4];
        for i in 0..4 {
            out[i] = left[i].wrapping_add(right[i].rotate_left(1));
        }
        out
    }

    fn commit_key(key: &[u8]) -> Self::Output {
        let mut out = [0_u8; 4];
        out[3] = key.last().copied().unwrap_or(0);
        out
    }
}

/// Hasher whose commitments collide for *every* key (used to exercise the
/// `KeyCollision` path).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CollidingHasher;

impl Hasher for CollidingHasher {
    type Output = [u8; 8];

    fn hash_leaf(data: &[u8]) -> Self::Output {
        let mut out = [0_u8; 8];
        out[0] = u8::try_from(data.len()).unwrap_or(u8::MAX);
        out
    }

    fn hash_node(left: &Self::Output, right: &Self::Output) -> Self::Output {
        let mut out = [0_u8; 8];
        for i in 0..8 {
            out[i] = left[i].wrapping_add(right[i]);
        }
        out
    }

    fn commit_key(_key: &[u8]) -> Self::Output {
        [0_u8; 8]
    }
}

/// Hasher with a zero-size output (used to exercise the `new` panic).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct EmptyHasher;

impl Hasher for EmptyHasher {
    type Output = [u8; 0];

    fn hash_leaf(_data: &[u8]) -> Self::Output {
        []
    }

    fn hash_node(_left: &Self::Output, _right: &Self::Output) -> Self::Output {
        []
    }
}

#[test]
fn insert_get_update_remove_roundtrip() {
    let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(3);
    assert!(tree.is_empty());
    assert_eq!(tree.version(), 3);

    tree.insert(b"alice", b"100").unwrap();
    tree.insert(b"bob", b"200").unwrap();
    assert_eq!(tree.leaf_count(), 2);
    assert!(!tree.is_empty());

    let alice = tree.get(b"alice").unwrap();
    assert_eq!(alice, Sha256Hasher::hash_leaf_with_key(b"alice", b"100"));

    // Missing key.
    assert_eq!(
        tree.get(b"carol"),
        Err(JMTError::KeyNotFound(b"carol".to_vec()))
    );

    // Replace existing key: root must change.
    let root_before = tree.root();
    tree.insert(b"alice", b"101").unwrap();
    assert_ne!(tree.root(), root_before);
    assert_eq!(
        tree.get(b"alice").unwrap(),
        Sha256Hasher::hash_leaf_with_key(b"alice", b"101")
    );

    // Strict update rejects absent keys.
    assert_eq!(
        tree.update(b"carol", b"300"),
        Err(JMTError::KeyNotFound(b"carol".to_vec()))
    );
    tree.update(b"alice", b"102").unwrap();
    assert_eq!(
        tree.get(b"alice").unwrap(),
        Sha256Hasher::hash_leaf_with_key(b"alice", b"102")
    );

    // Removal.
    assert!(tree.remove(b"bob"));
    assert!(!tree.remove(b"bob"));
    assert!(!tree.contains(b"bob"));
    assert!(tree.contains(b"alice"));
    assert_eq!(tree.leaf_count(), 1);
}

#[test]
fn insert_reports_key_collision() {
    let mut tree = JellyfishMerkleTree::<CollidingHasher>::new(0);
    tree.insert(b"k1", b"v1").unwrap();
    assert_eq!(
        tree.insert(b"k2", b"v2"),
        Err(JMTError::KeyCollision(b"k1".to_vec()))
    );
    // Same key twice is a legitimate overwrite, not a collision.
    tree.insert(b"k1", b"v2").unwrap();
}

#[test]
fn batch_insert_overwrites_existing_keys() {
    let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(0);
    tree.insert(b"a", b"1").unwrap();
    let root_before = tree.root();
    tree.batch_insert(&[(b"a", b"2")]).unwrap();
    assert_eq!(
        tree.get(b"a").unwrap(),
        Sha256Hasher::hash_leaf_with_key(b"a", b"2")
    );
    assert_ne!(tree.root(), root_before);
}

#[test]
fn membership_proof_cannot_prove_absence() {
    let mut tree = JellyfishMerkleTree::<TinyHasher>::new(0);
    tree.insert([0x00_u8], b"a").unwrap();
    let proof = tree.prove([0x00_u8]);
    // The leaf matches the queried key, so a non-membership claim fails.
    assert!(!proof.verify(tree.root(), [0x00_u8], None::<&[u8]>));
}

#[test]
fn equal_positions_for_different_keys_prove_absence() {
    // With CollidingHasher every key shares one position, so any leaf
    // proves that a different key cannot be committed.
    let mut tree = JellyfishMerkleTree::<CollidingHasher>::new(0);
    tree.insert(b"present", b"v").unwrap();
    let proof = tree.prove(b"other");
    assert!(proof.verify(tree.root(), b"other", None::<&[u8]>));
}

#[test]
fn batch_operations_are_atomic_and_counted() {
    let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(0);
    tree.batch_insert(&[(b"a", b"1"), (b"b", b"2"), (b"c", b"3")])
        .unwrap();
    assert_eq!(tree.leaf_count(), 3);

    // Duplicate key inside a batch is a legitimate overwrite.
    tree.batch_insert(&[(b"x", b"9"), (b"x", b"9")]).unwrap();
    assert_eq!(tree.leaf_count(), 4);

    // A colliding batch must leave the tree untouched.
    let mut tree = JellyfishMerkleTree::<CollidingHasher>::new(0);
    tree.batch_insert(&[(b"a", b"1")]).unwrap();
    let snapshot = tree.clone();
    let bad: Vec<(&[u8], &[u8])> = vec![(b"z", b"9")];
    assert_eq!(
        tree.batch_insert(&bad),
        Err(JMTError::KeyCollision(b"a".to_vec()))
    );
    assert_eq!(tree, snapshot);

    // batch_remove counts removals and refreshes the root only when needed.
    let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(0);
    tree.batch_insert(&[(b"a", b"1"), (b"b", b"2")]).unwrap();
    let root = tree.root();
    assert_eq!(tree.batch_remove(&[b"a", b"b"]), 2);
    assert_ne!(tree.root(), root);
    assert!(tree.is_empty());

    // Removing absent keys neither changes the root nor rebuilds.
    let root = tree.root();
    assert_eq!(tree.batch_remove(&[b"a"]), 0);
    assert_eq!(tree.root(), root);
}

#[test]
fn from_leaves_builds_and_rejects_collisions() {
    let tree =
        JellyfishMerkleTree::<Sha256Hasher>::from_leaves(7, &[(b"k1", b"v1"), (b"k2", b"v2")])
            .unwrap();
    assert_eq!(tree.version(), 7);
    assert_eq!(tree.leaf_count(), 2);

    let collision =
        JellyfishMerkleTree::<CollidingHasher>::from_leaves(0, &[(b"k1", b"v1"), (b"k2", b"v2")]);
    assert_eq!(collision, Err(JMTError::KeyCollision(b"k1".to_vec())));
}

#[test]
fn empty_tree_proves_and_verifies() {
    let tree = JellyfishMerkleTree::<Sha256Hasher>::new(0);
    let proof = tree.prove(b"ghost");
    assert_eq!(proof.level_count(), 0);
    assert_eq!(proof.leaf_key(), None);
    assert_eq!(proof.leaf_hash(), None);
    assert!(proof.verify(tree.root(), b"ghost", None::<&[u8]>));

    // A value can never be proven against an empty-tree proof.
    assert!(!proof.verify(tree.root(), b"ghost", Some(b"x")));
    // Nor against a non-empty root.
    let mut other = JellyfishMerkleTree::<Sha256Hasher>::new(0);
    other.insert(b"real", b"v").unwrap();
    assert!(!proof.verify(other.root(), b"ghost", None::<&[u8]>));

    // Empty root equals the canonical empty hash.
    assert_eq!(tree.root(), Sha256Hasher::empty_hash());
    assert!(tree.validate());
}

#[test]
fn membership_proofs_with_last_bit_pairs() {
    let mut tree = JellyfishMerkleTree::<TinyHasher>::new(0);
    tree.insert([0x00_u8], b"a").unwrap();
    tree.insert([0x01_u8], b"b").unwrap();

    let proof = tree.prove([0x00_u8]);
    assert!(proof.verify(tree.root(), [0x00_u8], Some(b"a")));
    assert!(!proof.verify(tree.root(), [0x00_u8], Some(b"wrong")));
    let proof = tree.prove([0x01_u8]);
    assert!(proof.verify(tree.root(), [0x01_u8], Some(b"b")));
}

#[test]
fn membership_proofs_with_first_bit_pairs() {
    // Keys differing in a HIGH position bit: a promotion chain (None
    // siblings) followed by a single merge. [0x00] vs [0x80] differ at bit
    // 24 of 32, so the chain spans 8 levels.
    let mut tree = JellyfishMerkleTree::<TinyHasher>::new(0);
    tree.insert([0x00_u8], b"a").unwrap();
    tree.insert([0x80_u8], b"b").unwrap();

    let proof = tree.prove([0x00_u8]);
    assert_eq!(proof.level_count(), 8);
    assert!(proof.verify(tree.root(), [0x00_u8], Some(b"a")));
    let proof = tree.prove([0x80_u8]);
    assert!(proof.verify(tree.root(), [0x80_u8], Some(b"b")));
}

#[test]
fn membership_proofs_with_three_keys() {
    let mut tree = JellyfishMerkleTree::<TinyHasher>::new(0);
    tree.insert([0x00_u8], b"a").unwrap();
    tree.insert([0x01_u8], b"b").unwrap();
    tree.insert([0x02_u8], b"c").unwrap();

    for (key, value) in [
        ([0x00_u8], &b"a"[..]),
        ([0x01_u8], &b"b"[..]),
        ([0x02_u8], &b"c"[..]),
    ] {
        let proof = tree.prove(key);
        assert!(proof.verify(tree.root(), key, Some(value)));
    }
}

#[test]
fn single_leaf_tree() {
    let mut tree = JellyfishMerkleTree::<TinyHasher>::new(0);
    let hash = TinyHasher::hash_leaf_with_key(&[0x42_u8], b"solo");
    tree.insert([0x42_u8], b"solo").unwrap();

    // A one-leaf subtree hashes to the leaf itself and the chain is empty.
    assert_eq!(tree.root(), hash);
    let proof = tree.prove([0x42_u8]);
    assert_eq!(proof.level_count(), 0);
    assert!(proof.verify(tree.root(), [0x42_u8], Some(b"solo")));
    assert!(!proof.verify(tree.root(), [0x42_u8], Some(b"other")));
}

#[test]
fn non_membership_proofs_verify() {
    let mut tree = JellyfishMerkleTree::<TinyHasher>::new(0);
    tree.insert([0x00_u8], b"a").unwrap();
    tree.insert([0x01_u8], b"b").unwrap();

    // Absent key.
    let proof = tree.prove([0x0F_u8]);
    assert!(proof.verify(tree.root(), [0x0F_u8], None::<&[u8]>));
    // A membership claim with a non-membership proof fails.
    assert!(!proof.verify(tree.root(), [0x0F_u8], Some(b"x")));
    // Excluding a key that IS present fails.
    let wrong = tree.prove([0x02_u8]);
    assert!(!wrong.verify(tree.root(), [0x00_u8], None::<&[u8]>));
}

#[test]
fn forged_proofs_are_rejected() {
    let mut tree = JellyfishMerkleTree::<TinyHasher>::new(0);
    tree.insert([0x00_u8], b"a").unwrap();
    tree.insert([0x01_u8], b"b").unwrap();

    // Oversized sibling chains cannot be folded.
    let mut proof = tree.prove([0x00_u8]);
    assert!(proof.verify(tree.root(), [0x00_u8], Some(b"a")));
    proof.siblings = vec![Some([0; 4]); 33];
    assert!(!proof.verify(tree.root(), [0x00_u8], Some(b"a")));

    // Wrong root.
    let proof = tree.prove([0x00_u8]);
    assert!(!proof.verify([0xEE; 4], [0x00_u8], Some(b"a")));

    // Proof bound to a different key.
    let proof = tree.prove([0x01_u8]);
    assert!(!proof.verify(tree.root(), [0x00_u8], Some(b"b")));

    // Empty-tree proof must not smuggle siblings.
    let empty = JellyfishMerkleTree::<TinyHasher>::new(0);
    let mut proof = empty.prove([0x00_u8]);
    proof.siblings = vec![None];
    assert!(!proof.verify(empty.root(), [0x00_u8], None::<&[u8]>));
}

#[test]
fn validate_detects_tampering() {
    let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(0);
    tree.insert(b"a", b"1").unwrap();
    assert!(tree.validate());
    tree.root = Sha256Hasher::hash_leaf(b"tampered");
    assert!(!tree.validate());
}

#[test]
fn keys_iterate_in_position_order() {
    let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(0);
    tree.insert(b"zebra", b"1").unwrap();
    tree.insert(b"apple", b"2").unwrap();
    tree.insert(b"mango", b"3").unwrap();

    let keys: Vec<&[u8]> = tree.keys().collect();
    assert_eq!(keys.len(), 3);

    let mut sorted: Vec<&[u8]> = keys.clone();
    sorted.sort_by_key(|key| Sha256Hasher::commit_key(key));
    assert_eq!(keys, sorted);
}

#[test]
fn len_matches_leaf_count() {
    let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(0);
    assert_eq!(tree.len(), 0);
    tree.insert(b"a", b"1").unwrap();
    assert_eq!(tree.len(), tree.leaf_count());
}

#[test]
fn serde_roundtrips_tree_and_proof() {
    let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(5);
    tree.insert(b"a", b"1").unwrap();
    tree.insert(b"b", b"2").unwrap();
    let proof = tree.prove(b"a");

    let encoded = postcard::to_allocvec(&tree).unwrap();
    let decoded: JellyfishMerkleTree<Sha256Hasher> = postcard::from_bytes(&encoded).unwrap();
    assert_eq!(decoded, tree);
    assert!(decoded.validate());

    let encoded = postcard::to_allocvec(&proof).unwrap();
    let decoded: JMTProof<Sha256Hasher> = postcard::from_bytes(&encoded).unwrap();
    assert_eq!(decoded, proof);
    assert!(decoded.verify(tree.root(), b"a", Some(b"1")));
}

#[test]
fn displays_are_human_readable() {
    let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(2);
    tree.insert(b"a", b"1").unwrap();
    let display = alloc::format!("{tree}");
    assert!(display.contains("Jellyfish Merkle Tree"));
    assert!(display.contains("Version: 2"));
    assert!(display.contains("Leaves: 1"));

    let proof = tree.prove(b"a");
    let display = alloc::format!("{proof}");
    assert!(display.contains("JMT Proof"));
    assert!(display.contains("Leaf Key:"));
    assert!(display.contains("Siblings:"));

    let empty_proof = JellyfishMerkleTree::<Sha256Hasher>::new(0).prove(b"x");
    let display = alloc::format!("{empty_proof}");
    assert!(display.contains("empty tree"));

    let debug = alloc::format!("{tree:?}");
    assert!(debug.contains("JellyfishMerkleTree"));
}

#[test]
fn versioned_storage_records_and_reconstructs() {
    let mut storage = VersionedJMTStorage::<Sha256Hasher>::new();
    assert_eq!(storage.current_version(), 0);
    assert_eq!(storage.version_count(), 0);

    let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(0);
    tree.insert(b"a", b"1").unwrap();

    // Version 1: delta-based snapshot of the initial key.
    let mut inserted = BTreeMap::new();
    inserted.insert(b"a".to_vec(), tree.get(b"a").unwrap());
    let root_v1 = tree.root();
    let v1 = storage.create_version_with_delta(
        JMTDelta {
            inserted,
            deleted: BTreeSet::new(),
        },
        tree.root(),
    );
    assert_eq!(v1, 1);

    // Version 2: delta-based.
    tree.insert(b"b", b"2").unwrap();
    let mut inserted = BTreeMap::new();
    inserted.insert(b"b".to_vec(), tree.get(b"b").unwrap());
    let root_v2 = tree.root();
    let v2 = storage.create_version_with_delta(
        JMTDelta {
            inserted,
            deleted: BTreeSet::new(),
        },
        tree.root(),
    );
    assert_eq!(v2, 2);
    assert_eq!(storage.version_count(), 2);

    assert_eq!(storage.get_root(1), Some(root_v1));
    assert_eq!(storage.get_root(42), None);
    assert!(storage.get_delta(2).is_some());
    assert!(storage.get_delta(1).is_some());

    // Reconstruct version 2 from deltas.
    let reconstructed = storage.reconstruct_tree(2).unwrap();
    assert_eq!(reconstructed.root(), root_v2);
    assert_eq!(reconstructed.leaf_count(), 2);
    assert!(reconstructed.validate());

    // Unknown versions are rejected.
    assert_eq!(
        storage.reconstruct_tree(9),
        Err(JMTError::InvalidVersion(9))
    );

    // A version without a delta cannot be replayed.
    storage.create_new_version(tree.root()); // version 3, no delta
    assert_eq!(
        storage.reconstruct_tree(3),
        Err(JMTError::InvalidVersion(3))
    );
}

#[test]
fn versioned_storage_replays_deletions() {
    let mut storage = VersionedJMTStorage::<Sha256Hasher>::new();
    let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(0);
    tree.insert(b"a", b"1").unwrap();
    tree.insert(b"b", b"2").unwrap();

    // Version 1 inserts both keys.
    let mut inserted = BTreeMap::new();
    inserted.insert(b"a".to_vec(), tree.get(b"a").unwrap());
    inserted.insert(b"b".to_vec(), tree.get(b"b").unwrap());
    storage.create_version_with_delta(
        JMTDelta {
            inserted,
            deleted: BTreeSet::new(),
        },
        tree.root(),
    );

    // Version 2 deletes "a".
    tree.remove(b"a");
    let mut deleted = BTreeSet::new();
    deleted.insert(b"a".to_vec());
    storage.create_version_with_delta(
        JMTDelta {
            inserted: BTreeMap::new(),
            deleted,
        },
        tree.root(),
    );

    let reconstructed = storage.reconstruct_tree(2).unwrap();
    assert_eq!(reconstructed.root(), tree.root());
    assert_eq!(reconstructed.leaf_count(), 1);
    assert!(reconstructed.validate());
    assert!(!reconstructed.contains(b"a"));
}

#[test]
fn versioned_storage_compaction_keeps_a_usable_base() {
    let mut storage = VersionedJMTStorage::<Sha256Hasher>::new();
    let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(0);
    for i in 0..4_u8 {
        tree.insert([i], b"v").unwrap();
        let mut inserted = BTreeMap::new();
        inserted.insert(vec![i], tree.get([i]).unwrap());
        storage.create_version_with_delta(
            JMTDelta {
                inserted,
                deleted: BTreeSet::new(),
            },
            tree.root(),
        );
    }
    let root_v4 = tree.root();

    // Keep only version 4: versions 1-3 fold into the base.
    storage.prune_with_compaction(1);
    assert_eq!(storage.version_count(), 1);
    assert!(storage.get_delta(1).is_none());

    let reconstructed = storage.reconstruct_tree(4).unwrap();
    assert_eq!(reconstructed.root(), root_v4);
    assert_eq!(reconstructed.leaf_count(), 4);
    assert!(reconstructed.validate());
    assert!(reconstructed.contains([0]));
    assert!(reconstructed.contains([3]));
}

#[test]
fn versioned_storage_prunes_old_versions() {
    let mut storage = VersionedJMTStorage::<Sha256Hasher>::new();
    // Build the tree cumulatively so each version's root matches the
    // replayed delta chain. Version 2 removes "a" again so the compaction
    // also folds a deletion into the base.
    let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(0);
    tree.insert(b"a", b"1").unwrap();
    let mut inserted = BTreeMap::new();
    inserted.insert(b"a".to_vec(), tree.get(b"a").unwrap());
    storage.create_version_with_delta(
        JMTDelta {
            inserted,
            deleted: BTreeSet::new(),
        },
        tree.root(),
    );
    tree.remove(b"a");
    let mut deleted = BTreeSet::new();
    deleted.insert(b"a".to_vec());
    storage.create_version_with_delta(
        JMTDelta {
            inserted: BTreeMap::new(),
            deleted,
        },
        tree.root(),
    );
    for i in 2_u8..4 {
        tree.insert([i], b"v").unwrap();
        let mut inserted = BTreeMap::new();
        inserted.insert(vec![i], tree.get([i]).unwrap());
        storage.create_version_with_delta(
            JMTDelta {
                inserted,
                deleted: BTreeSet::new(),
            },
            tree.root(),
        );
    }
    assert_eq!(storage.version_count(), 4);
    let root_v4 = tree.root();

    // No-op when keeping everything.
    storage.prune_with_compaction(4);
    assert_eq!(storage.version_count(), 4);

    storage.prune_with_compaction(2);
    assert_eq!(storage.version_count(), 2);
    assert!(storage.get_root(1).is_none());
    assert!(storage.get_root(3).is_some());
    assert!(storage.get_delta(3).is_some());
    assert!(storage.get_delta(1).is_none());

    // Compacted versions are gone for good…
    assert_eq!(
        storage.reconstruct_tree(1),
        Err(JMTError::InvalidVersion(1))
    );
    assert_eq!(
        storage.reconstruct_tree(2),
        Err(JMTError::InvalidVersion(2))
    );

    // …but retained versions are still reconstructible through the
    // compacted base ("a" was inserted and deleted again, so the base is
    // empty and version 4 holds exactly the two later keys).
    let reconstructed = storage.reconstruct_tree(4).unwrap();
    assert_eq!(reconstructed.root(), root_v4);
    assert_eq!(reconstructed.leaf_count(), 2);
    assert!(reconstructed.validate());
    assert!(reconstructed.contains([2]));
    assert!(reconstructed.contains([3]));
    assert!(!reconstructed.contains(b"a"));
    assert!(storage.reconstruct_tree(3).is_ok());
}

#[test]
fn memory_database_roundtrip() {
    let mut db = MemoryJMTDatabase::<Sha256Hasher>::new();
    assert_eq!(db.get_version().unwrap(), 0);

    db.set_version(5).unwrap();
    assert_eq!(db.get_version().unwrap(), 5);

    db.set_root(1, [1_u8; 32]).unwrap();
    assert_eq!(db.get_root(1).unwrap(), Some([1_u8; 32]));
    assert_eq!(db.get_root(2).unwrap(), None);

    db.set_leaf(b"key1", [2_u8; 32]).unwrap();
    assert_eq!(db.get_leaf(b"key1").unwrap(), Some([2_u8; 32]));
    assert_eq!(db.get_leaf(b"key2").unwrap(), None);

    assert!(db.delete_leaf(b"key1").unwrap());
    assert!(!db.delete_leaf(b"key1").unwrap());
    assert_eq!(db.get_leaf(b"key1").unwrap(), None);
}

#[test]
fn new_panics_on_empty_hasher_output() {
    let result = std::panic::catch_unwind(|| JellyfishMerkleTree::<EmptyHasher>::new(0));
    assert!(result.is_err());
}

#[test]
fn default_constructs_empty_tree() {
    let tree = JellyfishMerkleTree::<TestHasher>::default();
    assert!(tree.is_empty());
    assert_eq!(tree.version(), 0);

    let storage = VersionedJMTStorage::<TestHasher>::default();
    assert_eq!(storage.current_version(), 0);
    assert_eq!(storage.version_count(), 0);

    let db = MemoryJMTDatabase::<TestHasher>::default();
    assert_eq!(db.get_version().unwrap(), 0);
}

#[test]
fn serde_roundtrip_with_tiny_hasher() {
    let mut tree = JellyfishMerkleTree::<TinyHasher>::new(9);
    tree.insert([0x00_u8], b"a").unwrap();
    tree.insert([0x80_u8], b"b").unwrap();
    let proof = tree.prove([0x00_u8]);

    let encoded = postcard::to_allocvec(&tree).unwrap();
    let decoded: JellyfishMerkleTree<TinyHasher> = postcard::from_bytes(&encoded).unwrap();
    assert_eq!(decoded, tree);
    assert!(decoded.validate());

    let encoded = postcard::to_allocvec(&proof).unwrap();
    let decoded: JMTProof<TinyHasher> = postcard::from_bytes(&encoded).unwrap();
    assert_eq!(decoded, proof);
    assert!(decoded.verify(tree.root(), [0x00_u8], Some(b"a")));
}

#[test]
fn versioned_storage_serde_roundtrip() {
    let mut storage = VersionedJMTStorage::<Sha256Hasher>::new();
    let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(0);
    tree.insert(b"a", b"1").unwrap();
    let mut inserted = BTreeMap::new();
    inserted.insert(b"a".to_vec(), tree.get(b"a").unwrap());
    storage.create_version_with_delta(
        JMTDelta {
            inserted,
            deleted: BTreeSet::new(),
        },
        tree.root(),
    );

    let encoded = postcard::to_allocvec(&storage).unwrap();
    let decoded: VersionedJMTStorage<Sha256Hasher> = postcard::from_bytes(&encoded).unwrap();
    assert_eq!(decoded, storage);
    assert!(decoded.reconstruct_tree(1).is_ok());
}

#[test]
fn generic_signatures_are_exercised_for_common_key_shapes() {
    // Different key shapes ([u8; N] by value, &[u8; N], &str, Vec<u8>) each
    // instantiate the generic API separately; exercise them all.
    let mut sha = JellyfishMerkleTree::<Sha256Hasher>::new(0);
    sha.insert([0_u8; 1], b"v").unwrap();
    sha.insert([1_u8; 3], b"v").unwrap();
    sha.insert("str-key", b"v").unwrap();
    sha.insert(vec![9_u8, 9], b"v").unwrap();
    assert_eq!(sha.leaf_count(), 4);
    assert!(sha.get([0_u8; 1]).is_ok());
    assert!(sha.get([1_u8; 3]).is_ok());
    assert!(sha.get("str-key").is_ok());
    assert!(sha.get(vec![9_u8, 9]).is_ok());

    let proof_one = sha.prove([0_u8; 1]);
    assert!(proof_one.verify(sha.root(), [0_u8; 1], Some(b"v")));
    let proof_three = sha.prove([1_u8; 3]);
    assert!(proof_three.verify(sha.root(), [1_u8; 3], Some(b"v")));
    let proof = sha.prove("str-key");
    assert!(proof.verify(sha.root(), "str-key", Some(b"v")));
    assert_eq!(proof.leaf_key(), Some("str-key".as_bytes()));
    assert!(!sha.prove(b"ghost").verify(sha.root(), b"ghost", Some(b"x")));

    let mut colliding = JellyfishMerkleTree::<CollidingHasher>::new(0);
    colliding.batch_insert(&[("k1", "v1")]).unwrap();
    assert_eq!(
        colliding.batch_insert(&[("k2", "v2")]),
        Err(JMTError::KeyCollision(b"k1".to_vec()))
    );
    let proof = colliding.prove("ghost");
    assert!(proof.verify(colliding.root(), "ghost", None::<&[u8]>));

    // A separate tree: array keys exercise the same generic surface.
    let mut colliding_arrays = JellyfishMerkleTree::<CollidingHasher>::new(0);
    colliding_arrays.insert([0_u8; 1], b"v").unwrap();
    let proof = colliding_arrays.prove([0_u8; 1]);
    assert!(proof.verify(colliding_arrays.root(), [0_u8; 1], Some(b"v")));
}
