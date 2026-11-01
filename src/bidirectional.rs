//! Hierarchical [`BidirectionalMerkleTree`] linking child trees under the
//! leaves of a main tree.

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec::Vec;

use core::fmt::Write as _;
use serde::ser::SerializeTuple;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::hasher::Hasher;
use crate::tree::MerkleTree;

/// A Merkle tree whose leaves can commit to entire subordinate trees.
///
/// The *main* tree is a [`MerkleTree<HEIGHT, H>`]; each of its
/// [`MerkleTree::leaf_count`] slots can host a *child* tree (also
/// `MerkleTree<HEIGHT, H>`). Linking a child tree replaces the main-tree
/// leaf hash with `hash_leaf(child_root_bytes)`, so the main root commits
/// to every linked child root.
///
/// This enables hierarchical verification (cross-chain bridges, layered
/// state commitments): proving a child leaf requires the child proof plus a
/// main-tree proof whose leaf payload is the child root.
///
/// The generic parameter `MAX_CHILDREN` bounds how many child slots exist;
/// it should normally be `>= 2^HEIGHT` so every main leaf can host a child.
///
/// # Panics
///
/// Write methods ([`BidirectionalMerkleTree::link_child_tree`],
/// [`BidirectionalMerkleTree::unlink_child_tree`]) panic when
/// `leaf_index` is out of bounds; read methods return [`None`] instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BidirectionalMerkleTree<const HEIGHT: usize, const MAX_CHILDREN: usize, H: Hasher> {
    /// The top-level tree whose leaves commit to child roots.
    main_tree: MerkleTree<HEIGHT, H>,
    /// Child trees indexed by main-tree leaf index.
    child_trees: [Option<Box<MerkleTree<HEIGHT, H>>>; MAX_CHILDREN],
}

impl<const HEIGHT: usize, const MAX_CHILDREN: usize, H: Hasher>
    BidirectionalMerkleTree<HEIGHT, MAX_CHILDREN, H>
{
    /// Create an empty bidirectional tree (no children linked).
    #[must_use]
    pub fn new() -> Self {
        Self {
            main_tree: MerkleTree::new(),
            child_trees: [const { None }; MAX_CHILDREN],
        }
    }

    /// Link `child_tree` under main-tree leaf `leaf_index`.
    ///
    /// The main leaf is rehashed to `hash_leaf(child_root_bytes)` and the
    /// path to the main root is refreshed. Linking a second child tree at
    /// the same index replaces the previous link.
    ///
    /// # Panics
    ///
    /// Panics if `leaf_index >= MAX_CHILDREN` or if it is not a valid main
    /// tree leaf index.
    pub fn link_child_tree(&mut self, leaf_index: usize, child_tree: MerkleTree<HEIGHT, H>) {
        assert!(
            leaf_index < MAX_CHILDREN,
            "child slot {leaf_index} out of bounds (MAX_CHILDREN = {MAX_CHILDREN})"
        );
        let child_root = child_tree.root();
        self.main_tree.update_leaf(leaf_index, child_root.as_ref());
        self.child_trees[leaf_index] = Some(Box::new(child_tree));
    }

    /// Remove the child tree linked at `leaf_index`, if any.
    ///
    /// Returns `true` when a link existed. The main-tree leaf keeps the
    /// stale commitment (a hash commitment cannot be reversed); update the
    /// leaf explicitly via [`BidirectionalMerkleTree::main_tree_mut`] if
    /// you need to revoke it.
    ///
    /// # Panics
    ///
    /// Panics if `leaf_index >= MAX_CHILDREN`.
    pub fn unlink_child_tree(&mut self, leaf_index: usize) -> bool {
        assert!(
            leaf_index < MAX_CHILDREN,
            "child slot {leaf_index} out of bounds (MAX_CHILDREN = {MAX_CHILDREN})"
        );
        self.child_trees[leaf_index].take().is_some()
    }

    /// Child tree linked at `leaf_index`, if any (and if the slot exists).
    #[must_use]
    pub fn child_tree(&self, leaf_index: usize) -> Option<&MerkleTree<HEIGHT, H>> {
        self.child_trees.get(leaf_index)?.as_deref()
    }

    /// Root of the child tree linked at `leaf_index`, if any.
    #[must_use]
    pub fn child_root(&self, leaf_index: usize) -> Option<H::Output> {
        self.child_tree(leaf_index).map(MerkleTree::root)
    }

    /// Verify that `child_leaf_data` is committed in the child tree under
    /// `main_leaf_index`, and that this child tree is committed by the main
    /// tree root.
    ///
    /// Never panics: malformed indices simply yield `false`.
    #[must_use]
    pub fn verify_bidirectional_path(
        &self,
        main_leaf_index: usize,
        child_leaf_index: usize,
        child_leaf_data: &[u8],
    ) -> bool {
        let Some(child) = self.child_tree(main_leaf_index) else {
            return false;
        };
        if child_leaf_index >= child.leaf_count() || main_leaf_index >= self.main_tree.leaf_count()
        {
            return false;
        }
        let child_root = child.root();
        child
            .prove(child_leaf_index)
            .verify(child_leaf_data, child_root)
            && self
                .main_tree
                .prove(main_leaf_index)
                .verify(child_root.as_ref(), self.main_tree.root())
    }

    /// Read-only view of the main tree.
    #[must_use]
    pub const fn main_tree(&self) -> &MerkleTree<HEIGHT, H> {
        &self.main_tree
    }

    /// Mutable view of the main tree.
    ///
    /// Updating a leaf that hosts a linked child tree breaks the
    /// `hash_leaf(child_root)` commitment; re-link the child tree (or update
    /// the leaf with its root bytes) to restore consistency.
    pub const fn main_tree_mut(&mut self) -> &mut MerkleTree<HEIGHT, H> {
        &mut self.main_tree
    }

    /// Iterate over the linked child trees with their main-leaf indices.
    pub fn linked_children(&self) -> impl Iterator<Item = (usize, &MerkleTree<HEIGHT, H>)> {
        self.child_trees
            .iter()
            .enumerate()
            .filter_map(|(i, child)| child.as_deref().map(|tree| (i, tree)))
    }

    /// Number of linked child trees.
    #[must_use]
    pub fn child_tree_count(&self) -> usize {
        self.child_trees
            .iter()
            .filter(|child| child.is_some())
            .count()
    }
}

impl<const HEIGHT: usize, const MAX_CHILDREN: usize, H: Hasher> Default
    for BidirectionalMerkleTree<HEIGHT, MAX_CHILDREN, H>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const HEIGHT: usize, const MAX_CHILDREN: usize, H: Hasher> core::fmt::Display
    for BidirectionalMerkleTree<HEIGHT, MAX_CHILDREN, H>
{
    /// Renders a summary card; formatting into the intermediate `String`
    /// cannot fail, so the only fallible operation is the final flush.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut out = alloc::string::String::new();
        let _ = writeln!(out, "🔄 Bidirectional Merkle Tree");
        let _ = writeln!(out, "├── Main Tree:");
        // Indent every line of the nested main tree for the hierarchy view.
        let main = self.main_tree.to_string();
        for (i, line) in main.lines().enumerate() {
            if i > 0 {
                out.push_str("│   ");
            }
            out.push_str(line);
            out.push('\n');
        }
        let child_count = self.child_tree_count();
        let _ = writeln!(out, "├── Child Trees: {child_count}/{MAX_CHILDREN}");

        for (i, child) in self.linked_children() {
            let root = hex::encode(child.root().as_ref());
            let _ = writeln!(out, "│   ├── Leaf {i} → Child Root: {root}");
        }

        let _ = writeln!(out, "└── Configuration:");
        let _ = writeln!(out, "    Height: {HEIGHT}, Max Children: {MAX_CHILDREN}");

        f.write_str(&out)
    }
}

// Manual serde impls: serde provides no blanket array impls for
// const-generic lengths, so the child slots travel as a `Vec` alongside the
// main tree, with an exact-length check on the way back.
impl<const HEIGHT: usize, const MAX_CHILDREN: usize, H: Hasher> Serialize
    for BidirectionalMerkleTree<HEIGHT, MAX_CHILDREN, H>
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_tuple(2)?;
        state.serialize_element(&self.main_tree)?;
        let children: Vec<Option<&MerkleTree<HEIGHT, H>>> = self
            .child_trees
            .iter()
            .map(|child| child.as_deref())
            .collect();
        state.serialize_element(&children)?;
        state.end()
    }
}

impl<'de, const HEIGHT: usize, const MAX_CHILDREN: usize, H: Hasher> Deserialize<'de>
    for BidirectionalMerkleTree<HEIGHT, MAX_CHILDREN, H>
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Wire format: (main_tree, child_slots) with the slot vector
        // length-checked against MAX_CHILDREN.
        type Raw<const HEIGHT: usize, H> = (
            MerkleTree<HEIGHT, H>,
            Vec<Option<Box<MerkleTree<HEIGHT, H>>>>,
        );
        let (main_tree, child_trees): Raw<HEIGHT, H> = Deserialize::deserialize(deserializer)?;
        let slot_count = child_trees.len();
        let Ok(child_trees) = child_trees.try_into() else {
            return Err(serde::de::Error::invalid_length(
                slot_count,
                &"exactly MAX_CHILDREN child trees",
            ));
        };
        Ok(Self {
            main_tree,
            child_trees,
        })
    }
}

#[cfg(test)]
#[path = "bidirectional_tests.rs"]
mod tests;
