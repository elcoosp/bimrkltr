//! Benchmarks for the fixed-size [`MerkleTree`] and the sparse
//! [`JellyfishMerkleTree`](bimrkltr::JellyfishMerkleTree).
//!
//! Run with `cargo bench`.

use std::hint::black_box;

use bimrkltr::{JellyfishMerkleTree, MerkleTree, Sha256Hasher};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

/// Build `count` pseudo-random 32-byte leaves deterministically.
fn leaves(count: usize) -> Vec<[u8; 32]> {
    (0..count)
        .map(|i| {
            let mut leaf = [0_u8; 32];
            leaf[..8].copy_from_slice(&(i as u64).to_le_bytes());
            leaf[8..16].copy_from_slice(&(i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).to_le_bytes());
            leaf
        })
        .collect()
}

fn bench_from_leaves(c: &mut Criterion) {
    const HEIGHT: usize = 16;
    let leaves = leaves(1 << HEIGHT);
    c.bench_function("fixed_tree/from_leaves/height_16", |b| {
        b.iter(|| MerkleTree::<HEIGHT, Sha256Hasher>::from_leaves(black_box(&leaves)));
    });
}

fn bench_update_leaf(c: &mut Criterion) {
    const HEIGHT: usize = 16;
    let leaves = leaves(1 << HEIGHT);
    let mut tree = MerkleTree::<HEIGHT, Sha256Hasher>::from_leaves(&leaves);
    let mut i = 0_usize;
    c.bench_function("fixed_tree/update_leaf/height_16", |b| {
        b.iter(|| {
            i = (i + 1) % (1 << HEIGHT);
            tree.update_leaf(black_box(i), black_box(leaves[i]));
            tree.root()
        });
    });
}

fn bench_prove_verify(c: &mut Criterion) {
    const HEIGHT: usize = 16;
    let leaves = leaves(1 << HEIGHT);
    let tree = MerkleTree::<HEIGHT, Sha256Hasher>::from_leaves(&leaves);
    c.bench_function("fixed_tree/prove_verify/height_16", |b| {
        b.iter_batched(
            || tree.prove(black_box(1234)),
            |proof| {
                assert!(proof.verify(&leaves[1234], tree.root()));
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_jmt(c: &mut Criterion) {
    let keys: Vec<[u8; 16]> = (0_u64..1_000)
        .map(|i| {
            let mut key = [0_u8; 16];
            key[..8].copy_from_slice(&i.to_le_bytes());
            key
        })
        .collect();

    c.bench_function("jmt/batch_insert_1000", |b| {
        b.iter(|| {
            let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(0);
            let items: Vec<(&[u8], &[u8])> =
                keys.iter().map(|k| (k.as_slice(), k.as_slice())).collect();
            let _ = tree.batch_insert(&items);
            tree.root()
        });
    });

    let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(0);
    let items: Vec<(&[u8], &[u8])> = keys.iter().map(|k| (k.as_slice(), k.as_slice())).collect();
    let _ = tree.batch_insert(&items);
    c.bench_function("jmt/prove_1000", |b| {
        b.iter(|| tree.prove(black_box(&keys[500])));
    });
}

criterion_group!(
    benches,
    bench_from_leaves,
    bench_update_leaf,
    bench_prove_verify,
    bench_jmt,
);
criterion_main!(benches);
