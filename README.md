# bimrkltr

[![Crates.io](https://img.shields.io/crates/v/bimrkltr.svg)](https://crates.io/crates/bimrkltr)
[![Docs.rs](https://docs.rs/bimrkltr/badge.svg)](https://docs.rs/bimrkltr)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70+-blue.svg)](https://www.rust-lang.org)

A high-performance, compile-time sized Merkle tree implementation for Rust, featuring bidirectional tree support and generic hash functions.

## 🌳 Features

- **Const Generic Based**: Tree size determined at compile time for optimal performance
- **Bidirectional Trees**: Deepest leaves can serve as roots for other Merkle trees
- **Generic Hasher**: Support for any hash function implementing the `Hasher` trait
- **Zero-copy Proofs**: Efficient Merkle proof generation and verification
- **Safe & Fast**: Built with Rust's safety guarantees and performance in mind

## 🚀 Quick Start

### Installation

```toml
[dependencies]
bimrkltr = "0.1"
```

### Basic Usage

```rust
use bimrkltr::{MerkleTree, Hasher};

// You need to implement your own Hasher - here's a simple example
pub struct SimpleHasher;

impl Hasher for SimpleHasher {
    type Output = [u8; 32];

    fn hash_leaf(data: &[u8]) -> Self::Output {
        // Your hash implementation here
        let mut output = [0u8; 32];
        output.copy_from_slice(&data[..32]);
        output
    }

    fn hash_node(left: &Self::Output, right: &Self::Output) -> Self::Output {
        // Your hash implementation here  
        let mut output = [0u8; 32];
        for i in 0..32 {
            output[i] = left[i] ^ right[i];
        }
        output
    }
}

// Create a 32-leaf tree (height = 5, since 2^5 = 32 leaves)
type Tree32 = MerkleTree<5, SimpleHasher>;

let leaves = vec![[0u8; 32]; 32];
let tree = Tree32::from_leaves(&leaves);

// Get the root hash
let root = tree.root();

// Generate a proof for leaf at index 5
let proof = tree.prove(5);
assert!(proof.verify(&[0u8; 32], root));
```

## 📖 Examples

### Creating a Simple Tree

```rust
use bimrkltr::{MerkleTree, Hasher};

pub struct MyHasher;
impl Hasher for MyHasher {
    type Output = [u8; 32];
    
    fn hash_leaf(data: &[u8]) -> Self::Output {
        // Your leaf hashing logic
        [0u8; 32] // placeholder
    }
    
    fn hash_node(left: &Self::Output, right: &Self::Output) -> Self::Output {
        // Your node hashing logic  
        [0u8; 32] // placeholder
    }
}

// 16-leaf tree (height = 4, since 2^4 = 16 leaves)
type MyTree = MerkleTree<4, MyHasher>;

let leaves = vec![[0u8; 32]; 16]; // Must match 2^HEIGHT
let tree = MyTree::from_leaves(&leaves);
println!("Root: {:?}", tree.root());
```

### Bidirectional Trees

```rust
use bimrkltr::{MerkleTree, BidirectionalMerkleTree, Hasher};

pub struct MyHasher;
impl Hasher for MyHasher {
    type Output = [u8; 32];
    
    fn hash_leaf(data: &[u8]) -> Self::Output {
        [0u8; 32] // placeholder
    }
    
    fn hash_node(left: &Self::Output, right: &Self::Output) -> Self::Output {
        [0u8; 32] // placeholder
    }
}

// Create main tree (32 leaves)
type Tree32 = MerkleTree<5, MyHasher>;
type BidirectionalTree32 = BidirectionalMerkleTree<5, 32, MyHasher>; // HEIGHT=5, MAX_CHILDREN=32

// Create main tree
let main_leaves = vec![[0u8; 32]; 32];
let main_tree = Tree32::from_leaves(&main_leaves);

// Create child tree that will be linked to a leaf of the main tree  
let child_leaves = vec![[1u8; 32]; 32];
let child_tree = Tree32::from_leaves(&child_leaves);

// Create bidirectional tree and link them
let mut bi_tree = BidirectionalTree32::new();
bi_tree.link_child_tree(0, child_tree); // Link child tree to leaf index 0

// Get the child root
if let Some(child_root) = bi_tree.get_child_root(0) {
    println!("Child root: {:?}", child_root);
}

// Verify cross-tree paths
assert!(bi_tree.verify_bidirectional_path(
    0,           // main tree leaf index
    &[0u8; 32],  // main tree leaf data (must match what's stored)
    1,           // child tree leaf index  
    &[1u8; 32]   // child tree leaf data
));
```

### Custom Hash Function

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

// Use with custom hasher
type CustomTree = MerkleTree<4, KeccakHasher>;
```

## 🏗️ Architecture

### Core Components

- **`MerkleTree<const HEIGHT: usize, H: Hasher>`**: The main tree structure
- **`BidirectionalMerkleTree<const HEIGHT: usize, const MAX_CHILDREN: usize, H: Hasher>`**: Extension for hierarchical tree structures  
- **`MerkleProof`**: Efficient proof representation and verification
- **`Hasher`**: Generic trait for hash function implementations

### Tree Structure

```
Height = 3 (8 leaves)
Level 0:                 [Root]
Level 1:         [N0]           [N1]
Level 2:     [N2]   [N3]     [N4]   [N5]
Level 3: [L0] [L1] [L2] [L3] [L4] [L5] [L6] [L7]
```

## 🎯 Use Cases

### Blockchain & Cryptocurrency
- Transaction merklization
- State root calculations
- Light client verification

### Data Integrity
- File system verification
- Database consistency proofs
- Audit trail validation

### Complex Structures
- Hierarchical merkle trees
- Cross-chain bridges
- Multi-layer scaling solutions

## 🔧 Performance

The crate is designed for maximum performance:

- **O(log n)** leaf updates and proof generation
- **Compile-time sizing** for predictable memory usage
- **Efficient memory layout** for cache-friendly access

## 📚 API Reference

### Key Methods

| Method | Description | Complexity |
|--------|-------------|------------|
| `from_leaves()` | Build tree from leaf data | O(n) |
| `update_leaf()` | Update single leaf | O(log n) |
| `prove()` | Generate inclusion proof | O(log n) |
| `verify()` | Verify proof | O(log n) |
| `link_child_tree()` | Bidirectional linking | O(1) |

## 🌟 Important Notes

### Hasher Implementation
This crate does not provide built-in hashers. You must implement the `Hasher` trait for your specific hash function needs.

### Tree Sizing
- `HEIGHT` determines the tree size: `2^HEIGHT` leaves
- `MAX_CHILDREN` in bidirectional trees must be at least the number of leaves in the main tree

### Rust Version
Requires Rust 1.70+ due to const generic features.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests, open issues, or suggest new features.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
