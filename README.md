<div align="center">
  <img src="docs/logo.png" alt="bimrkltr Logo" width="300"/>
  <p>
    <strong>High-Performance, Compile-Time Sized Merkle Trees for Rust</strong><br/>
    Featuring bidirectional tree support, zero-copy proofs, and generic hash functions.
  </p>
  <p>
    <a href="https://crates.io/crates/bimrkltr"><img src="https://img.shields.io/crates/v/bimrkltr.svg?style=flat-square" alt="Crates.io"/></a>
    <a href="https://docs.rs/bimrkltr"><img src="https://img.shields.io/badge/docs.rs-bimrkltr-blue?style=flat-square" alt="Documentation"/></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License MIT"/></a>
    <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/Rust-1.70+-blue?style=flat-square&logo=rust" alt="Rust 1.70+"/></a>
  </p>
</div>

---

`bimrkltr` is a high-performance Merkle tree implementation designed for Rust. By leveraging const generics, it determines tree dimensions at compile time, ensuring optimal performance, predictable memory usage, and strict type safety. It supports generic hashing algorithms and advanced hierarchical (bidirectional) tree structures.

## Features

- **Const-Generic Sizing**: Tree dimensions are resolved at compile time for zero runtime overhead and predictable memory layouts.
- **Bidirectional Trees**: Deepest leaves can serve as roots for subordinate Merkle trees, enabling complex hierarchical structures.
- **Generic Hashing**: Agnostic to the hashing algorithm; supports any function implementing the `Hasher` trait.
- **Zero-Copy Proofs**: Efficient Merkle proof generation and verification with minimal allocations.
- **Idiomatic Rust**: Built with strict safety guarantees and performance-oriented design patterns.

## Quick Start

### Installation

Add `bimrkltr` to your `Cargo.toml`:

```toml
[dependencies]
bimrkltr = "0.1"
```

### Basic Usage

Because `bimrkltr` is hashing-agnostic, you must first implement the `Hasher` trait for your chosen algorithm.

```rust
use bimrkltr::{MerkleTree, Hasher};

// 1. Define a custom hasher
pub struct SimpleHasher;

impl Hasher for SimpleHasher {
    type Output = [u8; 32];

    fn hash_leaf(data: &[u8]) -> Self::Output {
        let mut output = [0u8; 32];
        output.copy_from_slice(&data[..32.min(data.len())]);
        output
    }

    fn hash_node(left: &Self::Output, right: &Self::Output) -> Self::Output {
        let mut output = [0u8; 32];
        for i in 0..32 {
            output[i] = left[i] ^ right[i];
        }
        output
    }
}

// 2. Define the tree type (Height = 5 means 2^5 = 32 leaves)
type Tree32 = MerkleTree<5, SimpleHasher>;

fn main() {
    let leaves = vec![[0u8; 32]; 32];
    let tree = Tree32::from_leaves(&leaves);

    // Get the root hash
    let root = tree.root();

    // Generate and verify an inclusion proof for leaf at index 5
    let proof = tree.prove(5);
    assert!(proof.verify(&[0u8; 32], root));
}
```

## Advanced Usage

### Bidirectional Trees

Link subordinate trees to the leaves of a parent tree for hierarchical verification (e.g., cross-chain bridges or multi-layer scaling).

```rust
use bimrkltr::{MerkleTree, BidirectionalMerkleTree, Hasher};

pub struct MyHasher;
impl Hasher for MyHasher {
    type Output = [u8; 32];
    fn hash_leaf(data: &[u8]) -> Self::Output { [0u8; 32] }
    fn hash_node(left: &Self::Output, right: &Self::Output) -> Self::Output { [0u8; 32] }
}

type Tree32 = MerkleTree<5, MyHasher>;
type BiTree32 = BidirectionalMerkleTree<5, 32, MyHasher>; // HEIGHT=5, MAX_CHILDREN=32

fn main() {
    let main_leaves = vec![[0u8; 32]; 32];
    let main_tree = Tree32::from_leaves(&main_leaves);

    let child_leaves = vec![[1u8; 32]; 32];
    let child_tree = Tree32::from_leaves(&child_leaves);

    let mut bi_tree = BiTree32::new();
    bi_tree.link_child_tree(0, child_tree); // Link child tree to main tree leaf index 0

    if let Some(child_root) = bi_tree.get_child_root(0) {
        println!("Child root: {:?}", child_root);
    }

    // Verify cross-tree paths
    assert!(bi_tree.verify_bidirectional_path(
        0, &[0u8; 32], // Main tree leaf index and expected data
        1, &[1u8; 32], // Child tree leaf index and expected data
    ));
}
```

### Custom Cryptographic Hash Functions

Integrate with established crates like `sha3` for production-grade cryptography.

```rust
use bimrkltr::{MerkleTree, Hasher};
use sha3::{Keccak256, Digest};

pub struct KeccakHasher;

impl Hasher for KeccakHasher {
    type Output = [u8; 32];

    fn hash_leaf(data: &[u8]) -> Self::Output {
        let mut hasher = Keccak256::new();
        hasher.update(b"leaf:");
        hasher.update(data);
        hasher.finalize().into()
    }

    fn hash_node(left: &Self::Output, right: &Self::Output) -> Self::Output {
        let mut hasher = Keccak256::new();
        hasher.update(b"node:");
        hasher.update(left);
        hasher.update(right);
        hasher.finalize().into()
    }
}

type CustomTree = MerkleTree<4, KeccakHasher>;
```

## Architecture & Performance

- **Core Components**: `MerkleTree`, `BidirectionalMerkleTree`, `MerkleProof`, and the `Hasher` trait.
- **Time Complexity**: `O(log n)` for leaf updates, proof generation, and verification. `O(1)` for bidirectional linking.
- **Memory Layout**: Cache-friendly access patterns optimized for modern CPU architectures, with zero runtime sizing overhead.

## API Reference

| Method | Description | Complexity |
| :--- | :--- | :--- |
| `from_leaves()` | Build tree from a slice of leaf data | `O(n)` |
| `update_leaf()` | Update a single leaf and recompute affected nodes | `O(log n)` |
| `prove()` | Generate an inclusion proof for a specific leaf | `O(log n)` |
| `verify()` | Verify a proof against a known root hash | `O(log n)` |
| `link_child_tree()` | Link a subordinate tree to a parent leaf | `O(1)` |

> [!IMPORTANT]
> **No Built-in Hashers**  
> This crate does not provide built-in hash functions. You must implement the `Hasher` trait for your specific cryptographic or non-cryptographic needs (e.g., SHA-2, Keccak, BLAKE3).

> [!NOTE]
> **Tree Sizing Constraints**  
> - The `HEIGHT` const generic determines the total number of leaves: `2^HEIGHT`.  
> - In `BidirectionalMerkleTree`, the `MAX_CHILDREN` const generic must be greater than or equal to the number of leaves in the main tree.  
> - Requires Rust 1.70+ due to advanced const generic features.
