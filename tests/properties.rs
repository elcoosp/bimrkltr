//! Property-based tests: random inputs hammering the core invariants with
//! an independent root oracle.

use proptest::prelude::*;

use bimrkltr::{Hasher, JellyfishMerkleTree, MerkleTree, Sha256Hasher};

/// Derive a deterministic leaf payload from a seed.
fn leaf(seed: u32) -> [u8; 32] {
    let mut out = [0_u8; 32];
    for (i, chunk) in out.chunks_exact_mut(4).enumerate() {
        let salt = u32::try_from(i).unwrap_or(u32::MAX);
        let mixed = seed
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
        chunk.copy_from_slice(&mixed.to_le_bytes());
    }
    out
}

/// Independent oracle: recursive divide-and-conquer root, sharing no code
/// with the level-order array construction.
fn oracle_root<H: Hasher>(leaf_hashes: &[H::Output]) -> H::Output {
    fn fold<H: Hasher>(slice: &[H::Output]) -> H::Output {
        if slice.len() == 1 {
            return slice[0];
        }
        let mid = slice.len() / 2;
        H::hash_node(&fold::<H>(&slice[..mid]), &fold::<H>(&slice[mid..]))
    }
    fold::<H>(leaf_hashes)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn fixed_tree_root_matches_oracle_and_proofs_verify(seeds in prop::collection::vec(any::<u32>(), 8)) {
        let payloads: Vec<[u8; 32]> = seeds.iter().map(|&s| leaf(s)).collect();
        let tree = MerkleTree::<3, Sha256Hasher>::from_leaves(&payloads);

        let hashes: Vec<_> = payloads.iter().map(|p| Sha256Hasher::hash_leaf(p)).collect();
        prop_assert_eq!(tree.root(), oracle_root::<Sha256Hasher>(&hashes));

        for (i, payload) in payloads.iter().enumerate() {
            prop_assert!(tree.prove(i).verify(payload, tree.root()));
        }
    }

    #[test]
    fn fixed_tree_update_matches_rebuild(seed in any::<u32>(), index in 0_usize..8, byte in any::<u8>()) {
        let payloads: Vec<[u8; 32]> = (0_u32..8).map(|i| leaf(seed.wrapping_add(i))).collect();
        let mut tree = MerkleTree::<3, Sha256Hasher>::from_leaves(&payloads);
        let updated = [byte; 32];

        tree.update_leaf(index, updated);
        prop_assert!(tree.prove(index).verify(&updated, tree.root()));

        let mut rebuilt = payloads;
        rebuilt[index] = updated;
        prop_assert_eq!(tree.root(), MerkleTree::<3, Sha256Hasher>::from_leaves(&rebuilt).root());
    }

    #[test]
    fn jmt_insert_order_does_not_matter(seeds in prop::collection::vec(any::<u64>(), 1..24)) {
        let unique: std::collections::BTreeSet<u64> = seeds.into_iter().collect();
        let pairs: Vec<([u8; 8], [u8; 8])> = unique
            .iter()
            .map(|&s| {
                let mut key = [0_u8; 8];
                key.copy_from_slice(&s.to_le_bytes());
                let value = s.wrapping_mul(31).to_le_bytes();
                (key, value)
            })
            .collect();

        let mut individual = JellyfishMerkleTree::<Sha256Hasher>::new(0);
        for (key, value) in &pairs {
            individual.insert(key, value).unwrap();
        }
        let mut batched = JellyfishMerkleTree::<Sha256Hasher>::new(0);
        batched.batch_insert(&pairs).unwrap();

        prop_assert_eq!(individual.root(), batched.root());
        prop_assert_eq!(individual.leaf_count(), unique.len());
        prop_assert!(individual.validate());
    }

    #[test]
    fn jmt_proofs_verify_for_members_and_absent_keys(
        present in prop::collection::vec(any::<u64>(), 1..16),
        absent in prop::collection::vec(any::<u64>(), 1..8),
    ) {
        let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(0);
        let mut members = std::collections::BTreeMap::new();
        for seed in &present {
            let mut key = [0_u8; 8];
            key.copy_from_slice(&seed.to_le_bytes());
            let value = seed.wrapping_mul(17).to_le_bytes();
            if tree.insert(key, value).is_ok() {
                members.insert(key, value);
            }
        }

        for (key, value) in &members {
            let proof = tree.prove(key);
            prop_assert!(proof.verify(tree.root(), key, Some(value)), "membership {key:?}");
            prop_assert!(!proof.verify(tree.root(), key, Some(b"wrong")));
        }

        for seed in &absent {
            let mut key = [0_u8; 8];
            key.copy_from_slice(&seed.to_le_bytes());
            if members.contains_key(&key) {
                continue;
            }
            let proof = tree.prove(key);
            prop_assert!(proof.verify(tree.root(), key, None::<&[u8]>), "absence {key:?}");
            prop_assert!(!proof.verify(tree.root(), key, Some(b"x")));
        }
    }

    #[test]
    fn jmt_removal_keeps_proofs_sound(seeds in prop::collection::vec(any::<u64>(), 2..16)) {
        let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(0);
        let mut keys = Vec::new();
        for seed in &seeds {
            let mut key = [0_u8; 8];
            key.copy_from_slice(&seed.to_le_bytes());
            if tree.insert(key, b"v").is_ok() {
                keys.push(key);
            }
        }
        prop_assume!(keys.len() >= 2);

        let victim = keys[0];
        prop_assert!(tree.remove(victim));
        let proof = tree.prove(victim);
        prop_assert!(proof.verify(tree.root(), victim, None::<&[u8]>));
        prop_assert!(tree.validate());

        // The survivor is still provable.
        let survivor = keys[1];
        prop_assert!(tree.prove(survivor).verify(tree.root(), survivor, Some(b"v")));
    }
}
