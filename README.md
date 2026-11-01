<div align="center">
  <img src="docs/logo.png" alt="bimrkltr Logo" width="300"/>
  <p>
    <strong>High-Performance, Compile-Time Sized Merkle Trees for Rust</strong><br/>
    Bidirectional trees, sparse jellyfish trees, zero-copy proofs, generic hashing — <code>no_std</code> ready.
  </p>
  <p>
    <a href="https://crates.io/crates/bimrkltr"><img src="https://img.shields.io/crates/v/bimrkltr.svg?style=flat-square" alt="Crates.io"/></a>
    <a href="https://docs.rs/bimrkltr"><img src="https://img.shields.io/badge/docs.rs-bimrkltr-blue?style=flat-square" alt="Documentation"/></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License MIT"/></a>
    <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/rustc-1.85+-blue?style=flat-square&logo=rust" alt="MSRV 1.85"/></a>
    <a href="https://github.com/elcoosp/bimrkltr/actions"><img src="https://img.shields.io/github/actions/workflow/status/elcoosp/bimrkltr/ci.yml?branch=main&style=flat-square&logo=githubactions" alt="CI"/></a>
  </p>
</div>

---

`bimrkltr` is a Merkle tree library designed for Rust. By leveraging const generics, it
determines tree dimensions at compile time for optimal performance, predictable memory usage
and strict type safety. A sound sparse tree ("jellyfish"-style) complements the fixed-size
tree for arbitrary key sets, and both share one generic hashing abstraction.

## Features

- **Const-generic sizing** — tree dimensions resolved at compile time: zero runtime sizing
  overhead, cache-friendly level-order layout.
- **Bidirectional trees** — link entire subordinate trees under main-tree leaves for
  hierarchical verification (cross-chain bridges, layered state commitments).
- **Sparse jellyfish tree** — commit to an arbitrary key set with membership *and*
  non-membership proofs, versioned storage with delta compaction, and a pluggable
  persistence trait.
- **Generic hashing** — implement two functions (`hash_leaf`, `hash_node`) and every derived
  scheme (key commitments, internal nodes, empty subtrees) comes for free.
- **`no_std` compatible** — needs only `alloc`; `serde` support on every public type.
- **Quality gates** — `forbid(unsafe_code)`, `clippy::pedantic` clean with zero `allow`s,
  property-based tests against an independent root oracle, criterion benchmarks.

## Quick Start

```toml
[dependencies]
bimrkltr = "0.2"
```

### Fixed-size tree

```rust
use bimrkltr::{MerkleTree, Sha256Hasher};

type Tree8 = MerkleTree<3, Sha256Hasher>; // 2^3 = 8 leaves

let leaves: Vec<[u8; 32]> = (0..8).map(|i| [i; 32]).collect();
let tree = Tree8::from_leaves(&leaves);

// Inclusion proof for leaf 5.
let proof = tree.prove(5);
assert!(proof.verify(&leaves[5], tree.root()));

// Update a leaf; the root follows in O(log n).
let mut tree = tree;
tree.update_leaf(5, [42_u8; 32]);
assert!(tree.prove(5).verify(&[42_u8; 32], tree.root()));
```

### Sparse tree over arbitrary keys

```rust
use bimrkltr::{JellyfishMerkleTree, Sha256Hasher};

let mut tree = JellyfishMerkleTree::<Sha256Hasher>::new(0);
tree.insert(b"alice", b"100").unwrap();
tree.insert(b"bob", b"200").unwrap();

// Membership proof.
assert!(tree.prove(b"alice").verify(tree.root(), b"alice", Some(b"100")));

// Non-membership proof for an absent key.
assert!(tree.prove(b"carol").verify(tree.root(), b"carol", None::<&[u8]>));
```

### Bidirectional trees

Link a child tree under a main-tree leaf; the main leaf commits to
`hash_leaf(child_root_bytes)`:

```rust
use bimrkltr::{BidirectionalMerkleTree, MerkleTree, Sha256Hasher};

type Tree = MerkleTree<2, Sha256Hasher>;

let child = Tree::from_leaves(&[[1u8; 32]; 4]);
let mut bi = BidirectionalMerkleTree::<2, 4, Sha256Hasher>::new();
bi.link_child_tree(0, child);

// Verifies the child leaf *and* that the child tree is committed by the main root.
let child_leaves = [[1u8; 32]; 4];
assert!(bi.verify_bidirectional_path(0, 2, &child_leaves[2]));
```

### Custom hashers

```rust
use bimrkltr::{Hasher, MerkleTree};
use sha2::{Digest, Sha256};

pub struct KeccakLikeHasher;

impl Hasher for KeccakLikeHasher {
    type Output = [u8; 32];

    fn hash_leaf(data: &[u8]) -> Self::Output {
        let mut hasher = Sha256::new();
        hasher.update(b"my-app:leaf:");
        hasher.update(data);
        hasher.finalize().into()
    }

    fn hash_node(left: &Self::Output, right: &Self::Output) -> Self::Output {
        let mut hasher = Sha256::new();
        hasher.update(b"my-app:node:");
        hasher.update(left);
        hasher.update(right);
        hasher.finalize().into()
    }
}

type CustomTree = MerkleTree<4, KeccakLikeHasher>;
```

> [!IMPORTANT]
> **Domain separation is mandatory.** Implementations must make
> `hash_leaf(x) == hash_node(x, y)` infeasible, or proofs can be forged by reinterpreting
> leaves as nodes. The built-in `Sha256Hasher` uses distinct `bimrkltr:leaf:` /
> `bimrkltr:node:` prefixes.

> [!NOTE]
> **Tree sizing.** `MerkleTree<HEIGHT, H>` has `2^HEIGHT` leaves (the degenerate
> `HEIGHT == 0` tree is a single leaf that is also the root). The sparse tree's depth is
> `8 × commit_key` output length. Keys are identified by their `Hasher::commit_key`
> commitment; commitment collisions between distinct keys are rejected with
> `JMTError::KeyCollision`.

## Architecture

| Module | Contents |
| :--- | :--- |
| `hasher` | `Hasher` trait (with derived default methods), `Sha256Hasher`, `TestHasher` |
| `tree` | `MerkleTree<HEIGHT, H>` — complete binary tree, level-order node array |
| `proof` | `MerkleProof<HEIGHT, H>` — `HEIGHT` sibling hashes, `O(log n)` verify |
| `bidirectional` | `BidirectionalMerkleTree` — child trees under main-tree leaves |
| `jmt` | `JellyfishMerkleTree<H>` — sparse trie, proofs, versioned storage, `JMTDatabase` |
| `error` | `JMTError` |

**Complexities** (`n` = leaves/keys, `k` = batch size):

| Operation | Fixed tree | Sparse tree |
| :--- | :--- | :--- |
| Build | `O(n)` | `O(n · depth)` fold (once per batch) |
| Update | `O(log n)`; batch `k` leaves: `O(k log n)` | `O(depth)` lookup + root refresh |
| Prove | `O(log n)` | `O(n · depth)` walk (root refresh dominates) |
| Verify | `O(log n)` | `O(depth)` |
| Memory | `(2^(HEIGHT+1) − 1)` hashes | one map entry per key |

## Development

```sh
cargo test            # unit + property + doc tests
cargo clippy --all-targets   # pedantic lints (configured in Cargo.toml)
cargo llvm-cov        # coverage (cargo install cargo-llvm-cov)
cargo bench           # criterion benchmarks
```

CI runs formatting, clippy, the test matrix (Linux/macOS/Windows + MSRV), coverage upload,
`no_std` build checks and documentation lint. Releases are cut by pushing a `v*` tag:
[git-cliff](https://git-cliff.org) generates the changelog, the crate is published to
crates.io and a GitHub release is created — see `.github/workflows/`.

## License

MIT — see [LICENSE-MIT](LICENSE-MIT).
