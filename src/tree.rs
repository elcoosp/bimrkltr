//! Compile-time sized complete binary [`MerkleTree`].

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use core::marker::PhantomData;

use core::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::hasher::Hasher;
use crate::proof::MerkleProof;

/// A complete binary Merkle tree whose shape is fixed at compile time.
///
/// `HEIGHT` determines the number of leaves: `2^HEIGHT` (with the degenerate
/// case `HEIGHT == 0` describing a single leaf that is also the root). The
/// node array is laid out in level order: the root at index 0, its children
/// at indices 1 and 2, and so on, with leaves occupying the last
/// `2^HEIGHT` slots.
///
/// # Panics
///
/// All leaf-index-taking methods panic when the index is out of bounds, and
/// [`MerkleTree::from_leaves`] panics when the slice length differs from
/// `2^HEIGHT`. These are programming errors caused by mixing up tree sizes;
/// they are cheap assertions that keep every other operation branch-free.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MerkleTree<const HEIGHT: usize, H: Hasher> {
    /// Level-order node array; `nodes[0]` is the root.
    nodes: Vec<H::Output>,
    /// Marker tying the tree to the hasher type.
    _hasher: PhantomData<H>,
}

impl<const HEIGHT: usize, H: Hasher> MerkleTree<HEIGHT, H> {
    /// Number of leaves: `2^HEIGHT` (1 when `HEIGHT == 0`).
    const LEAF_COUNT: usize = 1 << HEIGHT;

    /// Index of the first leaf in the level-order node array.
    const LEAF_OFFSET: usize = Self::LEAF_COUNT - 1;

    /// Total number of nodes: `2^(HEIGHT + 1) - 1`.
    const TOTAL_NODES: usize = (1 << (HEIGHT + 1)) - 1;

    /// Create a tree filled with default (zero) hashes.
    ///
    /// The root of a fresh tree is meaningless; use
    /// [`MerkleTree::from_leaves`] to build a committed tree, or update
    /// every leaf before trusting the root.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: vec![H::Output::default(); Self::TOTAL_NODES],
            _hasher: PhantomData,
        }
    }

    /// Build a tree from exactly `2^HEIGHT` leaf payloads.
    ///
    /// # Panics
    ///
    /// Panics if `leaves.len() != 2^HEIGHT`.
    pub fn from_leaves<L: AsRef<[u8]>>(leaves: &[L]) -> Self {
        assert!(
            leaves.len() == Self::LEAF_COUNT,
            "expected exactly {} leaves, got {}",
            Self::LEAF_COUNT,
            leaves.len()
        );

        let mut tree = Self::new();

        for (i, leaf) in leaves.iter().enumerate() {
            tree.nodes[Self::LEAF_OFFSET + i] = H::hash_leaf(leaf.as_ref());
        }

        // Bottom-up construction: pair the nodes of row `level` into their
        // parents in row `level - 1`, starting at the leaf row.
        for level in (1..=HEIGHT).rev() {
            let row_start = (1 << level) - 1;
            let parent_start = row_start / 2;
            for j in 0..(1 << (level - 1)) {
                let left = tree.nodes[row_start + 2 * j];
                let right = tree.nodes[row_start + 2 * j + 1];
                tree.nodes[parent_start + j] = H::hash_node(&left, &right);
            }
        }

        tree
    }

    /// Root hash of the tree.
    #[must_use]
    pub fn root(&self) -> H::Output {
        self.nodes[0]
    }

    /// Tree height (leaves = `2^height`).
    #[must_use]
    pub const fn height(&self) -> usize {
        HEIGHT
    }

    /// Number of leaves: `2^HEIGHT`.
    #[must_use]
    pub const fn leaf_count(&self) -> usize {
        Self::LEAF_COUNT
    }

    /// Zero-copy view of the leaf hashes, in leaf order.
    #[must_use]
    pub fn leaf_hashes(&self) -> &[H::Output] {
        &self.nodes[Self::LEAF_OFFSET..]
    }

    /// Hash of the leaf at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= 2^HEIGHT`.
    #[must_use]
    pub fn get_leaf(&self, index: usize) -> H::Output {
        assert!(index < Self::LEAF_COUNT, "leaf index {index} out of bounds");
        self.nodes[Self::LEAF_OFFSET + index]
    }

    /// Update the leaf at `index` and rehash its path to the root in
    /// `O(log n)`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= 2^HEIGHT`.
    pub fn update_leaf<L: AsRef<[u8]>>(&mut self, index: usize, data: L) {
        self.update_leaves(core::slice::from_ref(&(index, data)));
    }

    /// Update several leaves at once and rehash only the affected paths.
    ///
    /// Updating `k` distinct leaves costs `O(k log n)` hash operations
    /// instead of the `O(n)` a full rebuild would take, because shared
    /// ancestors are recomputed exactly once. Later entries win when the
    /// same index appears twice.
    ///
    /// # Panics
    ///
    /// Panics if any index is `>= 2^HEIGHT`.
    pub fn update_leaves<L: AsRef<[u8]>>(&mut self, updates: &[(usize, L)]) {
        let mut dirty: BTreeSet<usize> = BTreeSet::new();
        for &(index, ref data) in updates {
            assert!(index < Self::LEAF_COUNT, "leaf index {index} out of bounds");
            self.nodes[Self::LEAF_OFFSET + index] = H::hash_leaf(data.as_ref());
            dirty.insert(index);
        }

        if HEIGHT == 0 {
            return;
        }

        // Walk the dirty set upwards level by level; a parent is rehashed
        // iff at least one of its children was rehashed.
        let mut row_indices = dirty;
        for level in (1..=HEIGHT).rev() {
            let row_start = (1 << level) - 1;
            let parent_start = row_start / 2;
            let mut parents = BTreeSet::new();
            for &j in &row_indices {
                let parent_j = j / 2;
                let left = self.nodes[row_start + 2 * parent_j];
                let right = self.nodes[row_start + 2 * parent_j + 1];
                self.nodes[parent_start + parent_j] = H::hash_node(&left, &right);
                parents.insert(parent_j);
            }
            row_indices = parents;
        }
    }

    /// Generate an inclusion proof for the leaf at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= 2^HEIGHT`.
    #[must_use]
    pub fn prove(&self, index: usize) -> MerkleProof<HEIGHT, H> {
        assert!(index < Self::LEAF_COUNT, "leaf index {index} out of bounds");

        // In a complete tree every non-root node has an in-bounds sibling,
        // so the path never contains gaps.
        let mut path = [H::Output::default(); HEIGHT];
        let mut pos = Self::LEAF_OFFSET + index;
        for slot in &mut path {
            let sibling = if pos % 2 == 0 { pos - 1 } else { pos + 1 };
            *slot = self.nodes[sibling];
            pos = (pos - 1) / 2;
        }

        MerkleProof::from_parts(index, path)
    }
}

impl<const HEIGHT: usize, H: Hasher> Default for MerkleTree<HEIGHT, H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const HEIGHT: usize, H: Hasher> core::fmt::Display for MerkleTree<HEIGHT, H> {
    /// Renders a summary card; formatting into the intermediate `String`
    /// cannot fail, so the only fallible operation is the final flush.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut out = alloc::string::String::new();
        let root = hex::encode(self.root().as_ref());
        let _ = writeln!(out, "🌳 Merkle Tree (height: {HEIGHT})");
        let _ = writeln!(out, "├── Root: {root}");
        let _ = writeln!(out, "├── Leaves: {}", self.leaf_count());
        let _ = writeln!(out, "├── Total Nodes: {}", Self::TOTAL_NODES);

        // Show the first few leaves for a quick sanity glance.
        let show_leaves = 3.min(self.leaf_count());
        let _ = writeln!(out, "└── First {show_leaves} leaves:");
        for i in 0..show_leaves {
            let leaf = hex::encode(self.get_leaf(i).as_ref());
            let _ = writeln!(out, "    {i}: {leaf}");
        }

        f.write_str(&out)
    }
}

#[cfg(test)]
#[path = "tree_tests.rs"]
mod tests;
