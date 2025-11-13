# Merkle Tree Const

[![Crates.io](https://img.shields.io/crates/v/merkle-tree-const.svg)](https://crates.io/crates/merkle-tree-const)
[![Docs.rs](https://docs.rs/merkle-tree-const/badge.svg)](https://docs.rs/merkle-tree-const)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.60+-blue.svg)](https://www.rust-lang.org)

A high-performance, compile-time sized Merkle tree implementation for Rust, featuring bidirectional tree support and generic hash functions.

## 🌳 Features

- **Const Generic Based**: Tree size determined at compile time for optimal performance
- **Bidirectional Trees**: Deepest leaves can serve as roots for other Merkle trees
- **Generic Hasher**: Support for any hash function implementing the `Hasher` trait
- **Zero-copy Proofs**: Efficient Merkle proof generation and verification
- **Safe & Fast**: Built with Rust's safety guarantees and performance in mind
- **No Std Support**: Can be used in `no_std` environments

## 🚀 Quick Start

### Installation

```toml
[dependencies]
merkle-tree-const = "0.1"
```

### Basic Usage

```rust
use merkle_tree_const::{MerkleTree, Sha256Hasher};

// Create a 32-leaf tree (height = 5)
type Tree32 = MerkleTree<5, Sha256Hasher>;

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
use merkle_tree_const::{MerkleTree, Sha256Hasher};

// 16-leaf tree (height = 4)
type MyTree = MerkleTree<4, Sha256Hasher>;

let data = [
    b"leaf0".to_vec(),
    b"leaf1".to_vec(),
    b"leaf2".to_vec(),
    // ... more leaves
];

let leaves: Vec<[u8; 32]> = data.iter()
    .map(|d| Sha256Hasher::hash_leaf(d))
    .collect();

let tree = MyTree::from_leaves(&leaves);
println!("Root: {:?}", tree.root());
```

### Bidirectional Trees

```rust
use merkle_tree_const::{MerkleTree, BidirectionalMerkleTree, Sha256Hasher};

type Tree32 = MerkleTree<5, Sha256Hasher>;
type BidirectionalTree32 = BidirectionalMerkleTree<5, Sha256Hasher>;

// Create main tree
let main_leaves = vec![[0u8; 32]; 32];
let mut main_tree = Tree32::from_leaves(&main_leaves);

// Create child tree that will be linked to a leaf of the main tree
let child_leaves = vec![[1u8; 32]; 32];
let child_tree = Tree32::from_leaves(&child_leaves);

// Create bidirectional tree and link them
let mut bi_tree = BidirectionalTree32::new();
bi_tree.link_child_tree(0, child_tree);

// Verify cross-tree paths
assert!(bi_tree.verify_bidirectional_path(
    0,       // main tree leaf index
    &[0u8; 32], // main tree leaf data
    1,       // child tree leaf index  
    &[1u8; 32]  // child tree leaf data
));
```

### Custom Hash Function

```rust
use merkle_tree_const::{MerkleTree, Hasher};
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
- **`BidirectionalMerkleTree`**: Extension for hierarchical tree structures  
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
- **Compile-time sizing** eliminates dynamic allocations
- **Efficient memory layout** for cache-friendly access
- **Zero-copy operations** where possible

## 📚 API Reference

### Key Methods

| Method | Description | Complexity |
|--------|-------------|------------|
| `from_leaves()` | Build tree from leaf data | O(n) |
| `update_leaf()` | Update single leaf | O(log n) |
| `prove()` | Generate inclusion proof | O(log n) |
| `verify()` | Verify proof | O(log n) |
| `link_child_tree()` | Bidirectional linking | O(1) |

### Built-in Hashers

- `Sha256Hasher` - Standard SHA-256 implementation
- `KeccakHasher` - Keccak-256 (Ethereum compatible)
- Easy to implement custom hashers

## 🌟 Advanced Features

### No-std Support

```rust
#![no_std]
use merkle_tree_const::{MerkleTree, Sha256Hasher};

// Works in embedded environments
```

### Serialization

```rust
// Trees can be serialized/deserialized with serde
#[derive(Serialize, Deserialize)]
struct TreeWrapper {
    tree: MerkleTree<5, Sha256Hasher>,
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests, open issues, or suggest new features.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by Ethereum's Merkle Patricia Trie
- Design influenced by Bitcoin's Merkle tree implementation
- Thanks to all contributors and the Rust community

---

<div align="center">

**Built with ❤️ and Rust**

</div>
