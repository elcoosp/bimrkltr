//! Zero-copy style inclusion proofs for the fixed-size [`MerkleTree`](crate::MerkleTree).

use alloc::vec::Vec;
use core::marker::PhantomData;

use core::fmt::Write as _;
use serde::ser::SerializeTuple;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::hasher::Hasher;

/// Inclusion proof for a leaf of a compile-time sized [`MerkleTree`](crate::MerkleTree).
///
/// The proof carries the `HEIGHT` sibling hashes along the leaf-to-root
/// path; verification recomputes the root from the leaf payload and the
/// siblings. Proofs are `HEIGHT` hashes in size and verify in `O(log n)`.
///
/// Because a complete tree always has in-bounds siblings, every level
/// carries a real hash: there are no gaps to special-case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MerkleProof<const HEIGHT: usize, H: Hasher> {
    /// Index of the proven leaf.
    leaf_index: usize,
    /// Sibling hashes from the leaf level up to (but excluding) the root.
    path: [H::Output; HEIGHT],
    /// Marker tying the proof to the hasher type.
    _hasher: PhantomData<H>,
}

impl<const HEIGHT: usize, H: Hasher> MerkleProof<HEIGHT, H> {
    /// Verify the proof: recompute the root from `leaf_data` and compare it
    /// against the trusted `root`.
    #[must_use]
    pub fn verify(&self, leaf_data: &[u8], root: H::Output) -> bool {
        let mut hash = H::hash_leaf(leaf_data);
        let mut index = self.leaf_index;
        for sibling in &self.path {
            hash = if index % 2 == 0 {
                // Current node is a left child.
                H::hash_node(&hash, sibling)
            } else {
                // Current node is a right child.
                H::hash_node(sibling, &hash)
            };
            index /= 2;
        }
        hash == root
    }

    /// Index of the leaf this proof was generated for.
    #[must_use]
    pub const fn leaf_index(&self) -> usize {
        self.leaf_index
    }

    /// Number of sibling hashes in the path (always `HEIGHT`).
    #[must_use]
    pub const fn path_length(&self) -> usize {
        self.path.len()
    }

    /// Read-only view of the sibling hashes, leaf level first.
    #[must_use]
    pub fn path(&self) -> &[H::Output] {
        &self.path
    }
}

impl<const HEIGHT: usize, H: Hasher> MerkleProof<HEIGHT, H> {
    /// Assemble a proof from its parts; useful for deserialization-heavy
    /// workflows and tests.
    #[must_use]
    pub fn from_parts(leaf_index: usize, path: [H::Output; HEIGHT]) -> Self {
        Self {
            leaf_index,
            path,
            _hasher: PhantomData,
        }
    }
}

impl<const HEIGHT: usize, H: Hasher> core::fmt::Display for MerkleProof<HEIGHT, H> {
    /// Renders a summary card; formatting into the intermediate `String`
    /// cannot fail, so the only fallible operation is the final flush.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut out = alloc::string::String::new();
        let _ = writeln!(out, "🔐 Merkle Proof");
        let _ = writeln!(out, "├── Leaf Index: {}", self.leaf_index);
        let _ = writeln!(out, "├── Path Length: {}", self.path.len());
        let _ = writeln!(out, "└── Sibling Hashes:");

        for (i, hash) in self.path.iter().enumerate() {
            let sibling = hex::encode(hash.as_ref());
            let _ = writeln!(out, "    Level {i}: {sibling}");
        }

        f.write_str(&out)
    }
}

// Manual serde impls: serde provides no blanket array impls for
// const-generic lengths, so the path travels as a `Vec` alongside the leaf
// index, with an exact-length check on the way back.
impl<const HEIGHT: usize, H: Hasher> Serialize for MerkleProof<HEIGHT, H> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_tuple(2)?;
        state.serialize_element(&self.leaf_index)?;
        state.serialize_element(&self.path.as_slice())?;
        state.end()
    }
}

impl<'de, const HEIGHT: usize, H: Hasher> Deserialize<'de> for MerkleProof<HEIGHT, H> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (leaf_index, path): (usize, Vec<H::Output>) = Deserialize::deserialize(deserializer)?;
        let path_length = path.len();
        let Ok(path) = path.try_into() else {
            return Err(serde::de::Error::invalid_length(
                path_length,
                &"a path of exactly HEIGHT sibling hashes",
            ));
        };
        Ok(Self {
            leaf_index,
            path,
            _hasher: PhantomData,
        })
    }
}

#[cfg(test)]
#[path = "proof_tests.rs"]
mod tests;
