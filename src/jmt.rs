//! Sparse binary Merkle trees ("jellyfish"-style): a [`JellyfishMerkleTree`]
//! committing to an arbitrary key set, with membership and non-membership
//! proofs, versioned storage and a pluggable database trait.
//!
//! # How it works
//!
//! Every key is mapped to a fixed-size *position* via
//! [`Hasher::commit_key`] and the tree is the canonical sparse binary trie
//! over those bit strings:
//!
//! - a subtree with exactly one leaf hashes to that leaf hash,
//! - a subtree with several leaves hashes to
//!   [`Hasher::hash_internal_node`] over the recursively computed child
//!   subtree hashes (left child first),
//! - an empty tree hashes to [`Hasher::empty_hash`].
//!
//! A proof lists, for every level from the leaf to the root, the sibling
//! subtree hash when the queried node has one ([`Some`]) or [`None`] when
//! the node was promoted as an only child. Verification replays exactly the
//! same fold, so honestly generated proofs always verify, and forging one
//! requires a hash preimage.
//!
//! The same proof type covers membership (proof of a leaf equal to the
//! queried key) and non-membership (proof of a *different* leaf whose
//! position shadows the queried key); [`JMTProof::verify`] distinguishes
//! the two through the `value` argument.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt::{self, Display, Formatter};

use core::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::error::JMTError;
use crate::hasher::Hasher;

/// Is bit `j` (0 = most significant) of `position` set?
fn bit_msb(position: &[u8], j: usize) -> bool {
    (position[j / 8] >> (7 - j % 8)) & 1 == 1
}

/// Do `a` and `b` share their first `bits` bits?
///
/// Both slices must have the same length; `bits` must not exceed
/// `a.len() * 8`.
fn prefix_eq(a: &[u8], b: &[u8], bits: usize) -> bool {
    let full = bits / 8;
    if a[..full] != b[..full] {
        return false;
    }
    let rem = bits % 8;
    if rem == 0 {
        return true;
    }
    let mask = 0xFF_u8 << (8 - rem);
    (a[full] & mask) == (b[full] & mask)
}

/// Number of leading bits `a` and `b` have in common.
fn lcp_bits(a: &[u8], b: &[u8]) -> usize {
    let mut count = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        if x == y {
            count += 8;
        } else {
            count += (x ^ y).leading_zeros() as usize;
            break;
        }
    }
    count
}

/// A sparse Merkle tree committing to an arbitrary set of byte-string keys.
///
/// Unlike [`MerkleTree`](crate::MerkleTree) the tree is not complete: its
/// depth is fixed at `8 * commit_key_len` bits, but only the branches
/// leading to existing leaves exist. The tree maps keys to *leaf hashes*
/// (`hash_leaf_with_key(key, data)`); the underlying values live in
/// userland and are only ever committed to.
///
/// Every mutating operation refreshes the root; batch variants recompute it
/// once for the whole batch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JellyfishMerkleTree<H: Hasher> {
    /// Cached root hash, refreshed after every mutation.
    root: H::Output,
    /// Version tag carried by the tree (managed by the user or by
    /// [`VersionedJMTStorage`]).
    version: u64,
    /// Position (full key commitment) → (original key, leaf hash).
    leaves: BTreeMap<Vec<u8>, (Vec<u8>, H::Output)>,
}

impl<H: Hasher> JellyfishMerkleTree<H> {
    /// Create an empty tree tagged with `version`.
    ///
    /// # Panics
    ///
    /// Panics if the hasher output is empty (the tree would have no depth).
    #[must_use]
    pub fn new(version: u64) -> Self {
        assert!(
            !H::Output::default().as_ref().is_empty(),
            "hasher output must not be empty"
        );
        Self {
            root: H::empty_hash(),
            version,
            leaves: BTreeMap::new(),
        }
    }

    /// Create a tree from key/value pairs.
    ///
    /// # Errors
    ///
    /// Returns [`JMTError::KeyCollision`] if two distinct keys commit to the
    /// same position; in that case nothing is inserted.
    pub fn from_leaves<K: AsRef<[u8]>, V: AsRef<[u8]>>(
        version: u64,
        items: &[(K, V)],
    ) -> Result<Self, JMTError> {
        let mut tree = Self::new(version);
        tree.batch_insert(items)?;
        Ok(tree)
    }

    /// Insert (or replace) `key` with `data` and refresh the root.
    ///
    /// # Errors
    ///
    /// Returns [`JMTError::KeyCollision`] if a *different* key is already
    /// committed to the same position.
    pub fn insert<K: AsRef<[u8]>, V: AsRef<[u8]>>(
        &mut self,
        key: K,
        data: V,
    ) -> Result<(), JMTError> {
        let key = key.as_ref();
        let commitment = H::commit_key(key);
        let hash = H::hash_leaf_with_key(key, data.as_ref());
        if let Some((existing, _)) = self.leaves.get(commitment.as_ref()) {
            if existing.as_slice() != key {
                return Err(JMTError::KeyCollision(existing.clone()));
            }
        }
        self.leaves
            .insert(commitment.as_ref().to_vec(), (key.to_vec(), hash));
        self.root = self.collapse(None).0;
        Ok(())
    }

    /// Replace the data of an existing `key`.
    ///
    /// # Errors
    ///
    /// Returns [`JMTError::KeyNotFound`] if the key is absent and
    /// [`JMTError::KeyCollision`] on commitment collision.
    pub fn update<K: AsRef<[u8]>, V: AsRef<[u8]>>(
        &mut self,
        key: K,
        data: V,
    ) -> Result<(), JMTError> {
        if !self.contains(key.as_ref()) {
            return Err(JMTError::KeyNotFound(key.as_ref().to_vec()));
        }
        self.insert(key, data)
    }

    /// Insert a batch atomically and refresh the root once.
    ///
    /// # Errors
    ///
    /// Returns [`JMTError::KeyCollision`] if any pair of distinct keys (or
    /// any key and an existing entry) commits to the same position; in that
    /// case nothing is inserted.
    pub fn batch_insert<K: AsRef<[u8]>, V: AsRef<[u8]>>(
        &mut self,
        items: &[(K, V)],
    ) -> Result<(), JMTError> {
        let mut staged: Vec<(Vec<u8>, Vec<u8>, H::Output)> = Vec::with_capacity(items.len());
        for (key, data) in items {
            let key = key.as_ref();
            let commitment = H::commit_key(key);
            let position = commitment.as_ref();
            if let Some((existing, _)) = self.leaves.get(position) {
                if existing.as_slice() != key {
                    return Err(JMTError::KeyCollision(existing.clone()));
                }
            }
            if let Some((_, existing_key, _)) = staged
                .iter()
                .find(|(staged_position, _, _)| staged_position.as_slice() == position)
            {
                if existing_key.as_slice() != key {
                    return Err(JMTError::KeyCollision(existing_key.clone()));
                }
            }
            let hash = H::hash_leaf_with_key(key, data.as_ref());
            staged.push((position.to_vec(), key.to_vec(), hash));
        }
        for (position, key, hash) in staged {
            self.leaves.insert(position, (key, hash));
        }
        self.root = self.collapse(None).0;
        Ok(())
    }

    /// Remove `key`; returns whether it was present.
    pub fn remove<K: AsRef<[u8]>>(&mut self, key: K) -> bool {
        self.batch_remove(core::slice::from_ref(&key)) == 1
    }

    /// Remove several keys and refresh the root at most once; returns how
    /// many keys were actually removed.
    pub fn batch_remove<K: AsRef<[u8]>>(&mut self, keys: &[K]) -> usize {
        let mut removed = 0;
        for key in keys {
            let commitment = H::commit_key(key.as_ref());
            if self.leaves.remove(commitment.as_ref()).is_some() {
                removed += 1;
            }
        }
        if removed > 0 {
            self.root = self.collapse(None).0;
        }
        removed
    }

    /// Leaf hash committed for `key`.
    ///
    /// # Errors
    ///
    /// Returns [`JMTError::KeyNotFound`] if the key is absent.
    pub fn get<K: AsRef<[u8]>>(&self, key: K) -> Result<H::Output, JMTError> {
        let commitment = H::commit_key(key.as_ref());
        self.leaves
            .get(commitment.as_ref())
            .map(|(_, hash)| *hash)
            .ok_or_else(|| JMTError::KeyNotFound(key.as_ref().to_vec()))
    }

    /// Whether `key` is committed in the tree.
    #[must_use]
    pub fn contains<K: AsRef<[u8]>>(&self, key: K) -> bool {
        let commitment = H::commit_key(key.as_ref());
        self.leaves.contains_key(commitment.as_ref())
    }

    /// Iterate over the original keys, ordered by tree position.
    pub fn keys(&self) -> impl Iterator<Item = &[u8]> {
        self.leaves.values().map(|(key, _)| key.as_slice())
    }

    /// Number of committed keys.
    #[must_use]
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    /// Number of committed keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    /// Whether the tree commits to no key at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    /// Current root hash.
    #[must_use]
    pub fn root(&self) -> H::Output {
        self.root
    }

    /// Version tag of the tree.
    #[must_use]
    pub const fn version(&self) -> u64 {
        self.version
    }

    /// Recompute the root from the committed leaves and compare it with the
    /// cached root — a cheap integrity check after deserialization.
    #[must_use]
    pub fn validate(&self) -> bool {
        self.collapse(None).0 == self.root
    }

    /// Generate a proof for `key`.
    ///
    /// - key present → membership proof,
    /// - key absent, tree non-empty → non-membership proof bound to the
    ///   leaf sharing the longest common position prefix with the queried
    ///   position,
    /// - empty tree → empty proof.
    #[must_use]
    pub fn prove<K: AsRef<[u8]>>(&self, key: K) -> JMTProof<H> {
        let key = key.as_ref();
        let commitment = H::commit_key(key);
        let position = commitment.as_ref();
        if let Some((stored_key, hash)) = self.leaves.get(position) {
            return JMTProof {
                leaf: Some((stored_key.clone(), *hash)),
                siblings: self.collapse(Some(position)).1,
            };
        }
        // Non-membership: bind the proof to the "closest" existing leaf.
        let neighbor = self
            .leaves
            .iter()
            .max_by_key(|(leaf_position, _)| lcp_bits(leaf_position, position));
        if let Some((neighbor_position, (neighbor_key, neighbor_hash))) = neighbor {
            JMTProof {
                leaf: Some((neighbor_key.clone(), *neighbor_hash)),
                siblings: self.collapse(Some(neighbor_position)).1,
            }
        } else {
            JMTProof {
                leaf: None,
                siblings: Vec::new(),
            }
        }
    }

    /// Canonical bottom-up fold over all committed leaves.
    ///
    /// Returns the root hash and, when `target` is `Some(position)`, the
    /// sibling chain (`None` for promoted only-children) recorded along the
    /// way for that target. Entries are processed in position order; two
    /// adjacent entries sharing their first `depth - level - 1` bits are
    /// siblings and merge left-first.
    fn collapse(&self, target: Option<&[u8]>) -> (H::Output, Vec<Option<H::Output>>) {
        let mut chain = Vec::new();
        if self.leaves.is_empty() {
            return (H::empty_hash(), chain);
        }
        let mut entries: Vec<(&[u8], H::Output)> = self
            .leaves
            .iter()
            .map(|(position, (_, hash))| (position.as_slice(), *hash))
            .collect();
        // A merged node is tracked by its *left* child's position, so when
        // the target was the right child of a merge it inherits the left
        // position for the remaining levels.
        let mut target = target;
        let depth = entries[0].0.len() * 8;
        let mut level = 0;
        while entries.len() > 1 {
            let shared_bits = depth - level - 1;
            let mut next: Vec<(&[u8], H::Output)> = Vec::with_capacity(entries.len());
            let mut i = 0;
            while i < entries.len() {
                let current = entries[i];
                if i + 1 < entries.len() && prefix_eq(current.0, entries[i + 1].0, shared_bits) {
                    let right = entries[i + 1];
                    if let Some(tracked) = target {
                        if current.0 == tracked {
                            chain.push(Some(right.1));
                        } else if right.0 == tracked {
                            chain.push(Some(current.1));
                            target = Some(current.0);
                        }
                    }
                    let parent = H::hash_internal_node(&[current.1, right.1]);
                    next.push((current.0, parent));
                    i += 2;
                } else {
                    if let Some(tracked) = target {
                        if current.0 == tracked {
                            chain.push(None);
                        }
                    }
                    next.push(current);
                    i += 1;
                }
            }
            entries = next;
            level += 1;
        }
        (entries[0].1, chain)
    }
}

impl<H: Hasher> Default for JellyfishMerkleTree<H> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<H: Hasher> Display for JellyfishMerkleTree<H> {
    /// Renders a summary card; formatting into the intermediate `String`
    /// cannot fail, so the only fallible operation is the final flush.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut out = alloc::string::String::new();
        let depth = H::Output::default().as_ref().len() * 8;
        let root = hex::encode(self.root.as_ref());
        let _ = writeln!(out, "🎯 Jellyfish Merkle Tree");
        let _ = writeln!(out, "├── Version: {}", self.version);
        let _ = writeln!(out, "├── Root: {root}");
        let _ = writeln!(out, "├── Leaves: {}", self.leaf_count());
        let _ = writeln!(out, "└── Depth: {depth} bits");
        f.write_str(&out)
    }
}

/// Proof for a [`JellyfishMerkleTree`]: membership, non-membership or
/// empty-tree, selected implicitly by [`JMTProof::verify`] arguments.
///
/// `siblings` has one entry per level from the leaf to the root:
/// [`Some`] where the proof node had a sibling, [`None`] where it was
/// promoted as an only child.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JMTProof<H: Hasher> {
    /// The proven leaf: `(key, leaf_hash)`.
    ///
    /// [`None`] only for the empty-tree proof.
    leaf: Option<(Vec<u8>, H::Output)>,
    /// Sibling hashes, leaf level first; [`None`] marks a promotion level.
    siblings: Vec<Option<H::Output>>,
}

impl<H: Hasher> JMTProof<H> {
    /// Verify the proof against a trusted `root`.
    ///
    /// - `value = Some(data)`: membership proof of `key` → `data`;
    /// - `value = None`: non-membership proof that `key` is absent.
    #[must_use]
    pub fn verify<K: AsRef<[u8]>, V: AsRef<[u8]>>(
        &self,
        root: H::Output,
        key: K,
        value: Option<V>,
    ) -> bool {
        let key = key.as_ref();
        let Some((leaf_key, leaf_hash)) = &self.leaf else {
            // Empty-tree proof: sound only when nothing could be there.
            return value.is_none() && self.siblings.is_empty() && root == H::empty_hash();
        };

        let commitment = H::commit_key(leaf_key);
        let leaf_position = commitment.as_ref();
        if self.siblings.len() > leaf_position.len() * 8 {
            return false;
        }

        if let Some(value) = value {
            if leaf_key.as_slice() != key {
                return false;
            }
            if H::hash_leaf_with_key(key, value.as_ref()) != *leaf_hash {
                return false;
            }
            self.fold(leaf_position, *leaf_hash) == root
        } else if leaf_key.as_slice() == key {
            // A membership proof can never prove absence.
            false
        } else {
            // Non-membership: the proof commits to a *different* leaf.
            // Soundness requires that leaf to be the one "shadowing" the
            // queried key: at the level where the two positions diverge,
            // the chain must show a *promotion* (no sibling), proving
            // that the subtree on the queried side is empty. A pair at
            // that level would mean a closer leaf exists (or the queried
            // key itself), and the fold alone cannot rule it out.
            let key_commitment = H::commit_key(key);
            let key_position = key_commitment.as_ref();
            let depth = leaf_position.len() * 8;
            let divergence = lcp_bits(leaf_position, key_position);
            if divergence == depth {
                // Identical positions for different keys: the position
                // is occupied by another key, so `key` cannot be in an
                // honestly built tree.
                return self.fold(leaf_position, *leaf_hash) == root;
            }
            let divergence_level = depth - 1 - divergence;
            let promoted = match self.siblings.get(divergence_level) {
                None | Some(None) => true,
                Some(Some(_)) => false,
            };
            promoted && self.fold(leaf_position, *leaf_hash) == root
        }
    }

    /// Replay the sibling chain from a leaf position up to the root.
    ///
    /// # Panics
    ///
    /// Panics if the chain is longer than the position (a forged proof);
    /// [`JMTProof::verify`] rejects those before folding.
    fn fold(&self, position: &[u8], mut hash: H::Output) -> H::Output {
        let depth = position.len() * 8;
        debug_assert!(self.siblings.len() <= depth);
        for (level, sibling) in self.siblings.iter().enumerate() {
            if let Some(sibling) = sibling {
                // The distinguishing bit of the level: `depth - level - 1`.
                hash = if bit_msb(position, depth - level - 1) {
                    H::hash_internal_node(&[*sibling, hash])
                } else {
                    H::hash_internal_node(&[hash, *sibling])
                };
            }
        }
        hash
    }

    /// The proven leaf key, if the proof carries one.
    #[must_use]
    pub fn leaf_key(&self) -> Option<&[u8]> {
        self.leaf.as_ref().map(|(key, _)| key.as_slice())
    }

    /// The proven leaf hash, if the proof carries one.
    #[must_use]
    pub fn leaf_hash(&self) -> Option<&H::Output> {
        self.leaf.as_ref().map(|(_, hash)| hash)
    }

    /// Number of levels the proof spans.
    #[must_use]
    pub fn level_count(&self) -> usize {
        self.siblings.len()
    }
}

impl<H: Hasher> Display for JMTProof<H>
where
    H::Output: AsRef<[u8]>,
{
    /// Renders a summary card; formatting into the intermediate `String`
    /// cannot fail, so the only fallible operation is the final flush.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut out = alloc::string::String::new();
        let _ = writeln!(out, "🔐 JMT Proof");
        match &self.leaf {
            Some((key, hash)) => {
                let leaf_key = hex::encode(key);
                let leaf_hash = hex::encode(hash.as_ref());
                let _ = writeln!(out, "├── Leaf Key: {leaf_key}");
                let _ = writeln!(out, "├── Leaf Hash: {leaf_hash}");
            }
            None => {
                let _ = writeln!(out, "├── Leaf: (empty tree)");
            }
        }
        let present = self.siblings.iter().filter(|s| s.is_some()).count();
        let _ = writeln!(
            out,
            "└── Siblings: {present} of {} levels",
            self.siblings.len()
        );
        f.write_str(&out)
    }
}

/// Set of changes applied between two versions of a
/// [`JellyfishMerkleTree`].
// bound = "": H itself is never serialized, only its associated Output
// type, which already carries the serde bounds via the Hasher trait.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct JMTDelta<H: Hasher> {
    /// Keys inserted (or replaced) in the version, with their leaf hashes.
    pub inserted: BTreeMap<Vec<u8>, H::Output>,
    /// Keys removed in the version.
    pub deleted: BTreeSet<Vec<u8>>,
}

/// History of [`JellyfishMerkleTree`] roots with per-version deltas,
/// supporting reconstruction of past versions and compaction.
///
/// Reconstruction replays deltas; when old versions are pruned, their
/// deltas are *compacted* into a cumulative base delta so that newer
/// versions remain reconstructible.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VersionedJMTStorage<H: Hasher> {
    /// Cumulative state of every delta folded away by compaction.
    base: JMTDelta<H>,
    /// Highest version folded into `base` (0 = nothing folded yet).
    base_version: u64,
    /// Root hash per version.
    trees: BTreeMap<u64, H::Output>,
    /// Delta per version (deltas start at version 1).
    deltas: BTreeMap<u64, JMTDelta<H>>,
    /// Highest version created so far.
    current_version: u64,
}

impl<H: Hasher> VersionedJMTStorage<H> {
    /// Create empty storage starting before version 1.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: JMTDelta {
                inserted: BTreeMap::new(),
                deleted: BTreeSet::new(),
            },
            base_version: 0,
            trees: BTreeMap::new(),
            deltas: BTreeMap::new(),
            current_version: 0,
        }
    }

    /// Record `root` as the next version; returns its number.
    pub fn create_new_version(&mut self, root: H::Output) -> u64 {
        self.current_version += 1;
        self.trees.insert(self.current_version, root);
        self.current_version
    }

    /// Record `root` together with the `delta` that produced it; returns
    /// the version number.
    pub fn create_version_with_delta(&mut self, delta: JMTDelta<H>, root: H::Output) -> u64 {
        let version = self.create_new_version(root);
        self.deltas.insert(version, delta);
        version
    }

    /// Root hash recorded for `version`.
    #[must_use]
    pub fn get_root(&self, version: u64) -> Option<H::Output> {
        self.trees.get(&version).copied()
    }

    /// Delta recorded for `version`.
    #[must_use]
    pub fn get_delta(&self, version: u64) -> Option<&JMTDelta<H>> {
        self.deltas.get(&version)
    }

    /// Rebuild the tree state of `version` by replaying the compacted base
    /// and all deltas up to it.
    ///
    /// # Errors
    ///
    /// Returns [`JMTError::InvalidVersion`] if the version was never
    /// recorded, or if any delta needed to replay it is missing (versions
    /// created without a delta cannot be reconstructed).
    pub fn reconstruct_tree(&self, version: u64) -> Result<JellyfishMerkleTree<H>, JMTError> {
        let root = self
            .get_root(version)
            .ok_or(JMTError::InvalidVersion(version))?;
        let mut tree = JellyfishMerkleTree::new(version);
        for (key, hash) in &self.base.inserted {
            let commitment = H::commit_key(key);
            tree.leaves
                .insert(commitment.as_ref().to_vec(), (key.clone(), *hash));
        }
        for v in (self.base_version + 1)..=version {
            let delta = self.get_delta(v).ok_or(JMTError::InvalidVersion(version))?;
            for (key, hash) in &delta.inserted {
                let commitment = H::commit_key(key);
                tree.leaves
                    .insert(commitment.as_ref().to_vec(), (key.clone(), *hash));
            }
            for key in &delta.deleted {
                let commitment = H::commit_key(key);
                tree.leaves.remove(commitment.as_ref());
            }
        }
        tree.root = root;
        Ok(tree)
    }

    /// Drop the `trees.len() - keep_latest` oldest versions, folding their
    /// deltas into a cumulative base so the retained versions stay
    /// reconstructible. Versions at or below the compaction point can no
    /// longer be reconstructed individually.
    pub fn prune_with_compaction(&mut self, keep_latest: usize) {
        if self.trees.len() <= keep_latest {
            return;
        }
        let keep_from = self.current_version - keep_latest as u64 + 1;
        for version in (self.base_version + 1)..keep_from {
            if let Some(delta) = self.deltas.remove(&version) {
                for (key, hash) in delta.inserted {
                    self.base.deleted.remove(&key);
                    self.base.inserted.insert(key, hash);
                }
                for key in delta.deleted {
                    self.base.inserted.remove(&key);
                    self.base.deleted.insert(key);
                }
                self.base_version = version;
            }
            self.trees.remove(&version);
        }
        self.trees.retain(|&version, _| version >= keep_from);
        self.deltas.retain(|&version, _| version >= keep_from);
    }

    /// Highest version number created so far.
    #[must_use]
    pub const fn current_version(&self) -> u64 {
        self.current_version
    }

    /// Number of recorded versions.
    #[must_use]
    pub fn version_count(&self) -> usize {
        self.trees.len()
    }
}

impl<H: Hasher> Default for VersionedJMTStorage<H> {
    fn default() -> Self {
        Self::new()
    }
}

/// Pluggable persistence layer for [`JellyfishMerkleTree`] state.
///
/// Implement this over your storage engine (`RocksDB`, `Postgres`, ...) to make
/// trees and version history durable.
pub trait JMTDatabase<H: Hasher> {
    /// Read the root recorded for `version`.
    ///
    /// # Errors
    ///
    /// Implementation-defined storage failures.
    fn get_root(&self, version: u64) -> Result<Option<H::Output>, JMTError>;

    /// Record the root of `version`.
    ///
    /// # Errors
    ///
    /// Implementation-defined storage failures.
    fn set_root(&mut self, version: u64, root: H::Output) -> Result<(), JMTError>;

    /// Read the leaf hash committed for `key`.
    ///
    /// # Errors
    ///
    /// Implementation-defined storage failures.
    fn get_leaf(&self, key: &[u8]) -> Result<Option<H::Output>, JMTError>;

    /// Commit `hash` for `key`.
    ///
    /// # Errors
    ///
    /// Implementation-defined storage failures.
    fn set_leaf(&mut self, key: &[u8], hash: H::Output) -> Result<(), JMTError>;

    /// Remove `key`; returns whether it existed.
    ///
    /// # Errors
    ///
    /// Implementation-defined storage failures.
    fn delete_leaf(&mut self, key: &[u8]) -> Result<bool, JMTError>;

    /// Read the current version counter.
    ///
    /// # Errors
    ///
    /// Implementation-defined storage failures.
    fn get_version(&self) -> Result<u64, JMTError>;

    /// Overwrite the version counter.
    ///
    /// # Errors
    ///
    /// Implementation-defined storage failures.
    fn set_version(&mut self, version: u64) -> Result<(), JMTError>;
}

/// In-memory [`JMTDatabase`] suitable for tests and single-process use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryJMTDatabase<H: Hasher> {
    /// Root hashes per version.
    roots: BTreeMap<u64, H::Output>,
    /// Leaf hashes per key.
    leaves: BTreeMap<Vec<u8>, H::Output>,
    /// Version counter.
    current_version: u64,
}

impl<H: Hasher> MemoryJMTDatabase<H> {
    /// Create an empty database.
    #[must_use]
    pub fn new() -> Self {
        Self {
            roots: BTreeMap::new(),
            leaves: BTreeMap::new(),
            current_version: 0,
        }
    }
}

impl<H: Hasher> Default for MemoryJMTDatabase<H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<H: Hasher> JMTDatabase<H> for MemoryJMTDatabase<H> {
    fn get_root(&self, version: u64) -> Result<Option<H::Output>, JMTError> {
        Ok(self.roots.get(&version).copied())
    }

    fn set_root(&mut self, version: u64, root: H::Output) -> Result<(), JMTError> {
        self.roots.insert(version, root);
        Ok(())
    }

    fn get_leaf(&self, key: &[u8]) -> Result<Option<H::Output>, JMTError> {
        Ok(self.leaves.get(key).copied())
    }

    fn set_leaf(&mut self, key: &[u8], hash: H::Output) -> Result<(), JMTError> {
        self.leaves.insert(key.to_vec(), hash);
        Ok(())
    }

    fn delete_leaf(&mut self, key: &[u8]) -> Result<bool, JMTError> {
        Ok(self.leaves.remove(key).is_some())
    }

    fn get_version(&self) -> Result<u64, JMTError> {
        Ok(self.current_version)
    }

    fn set_version(&mut self, version: u64) -> Result<(), JMTError> {
        self.current_version = version;
        Ok(())
    }
}

#[cfg(test)]
#[path = "jmt_tests.rs"]
mod tests;
