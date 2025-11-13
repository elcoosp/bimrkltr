// src/lib.rs
use core::fmt;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::{Display, Formatter};

pub trait Hasher: Sized {
    type Output: Copy
        + Clone
        + std::fmt::Debug
        + PartialEq
        + Default
        + AsRef<[u8]>
        + Serialize
        + for<'de> Deserialize<'de>;

    fn hash_leaf(data: &[u8]) -> Self::Output;
    fn hash_node(left: &Self::Output, right: &Self::Output) -> Self::Output;

    // JMT optimizations
    fn empty_hash() -> Self::Output;
    fn hash_leaf_with_key(key: &[u8], data: &[u8]) -> Self::Output;
    fn hash_internal_node(children: &[Self::Output]) -> Self::Output;
    fn hash_with_version(version: u64, data: &[u8]) -> Self::Output;
    fn commit_key(key: &[u8]) -> Self::Output;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JellyfishMerkleTree<const HEIGHT: usize, H: Hasher> {
    root: H::Output,
    leaves: BTreeMap<Vec<u8>, H::Output>,
    version: u64,
    _hasher: std::marker::PhantomData<H>,
    node_cache: HashMap<Vec<u8>, H::Output>,
    cache_limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JMTProof<H: Hasher> {
    leaf: Option<(Vec<u8>, H::Output)>,
    siblings: Vec<H::Output>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JMTNonMembershipProof<H: Hasher> {
    key: Vec<u8>,
    predecessor: Option<(Vec<u8>, H::Output)>,
    successor: Option<(Vec<u8>, H::Output)>,
    root: H::Output,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JMTDelta<H: Hasher> {
    pub inserted: BTreeMap<Vec<u8>, H::Output>,
    pub deleted: BTreeSet<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct VersionedJMTStorage<H: Hasher> {
    trees: BTreeMap<u64, H::Output>,
    deltas: BTreeMap<u64, JMTDelta<H>>,
    current_version: u64,
    _hasher: std::marker::PhantomData<H>,
}

// Error types
#[derive(Debug, Clone)]
pub enum JMTError {
    KeyNotFound(Vec<u8>),
    InvalidVersion(u64),
    InvalidProof,
    TreeFull,
    InvalidKey,
    SerializationError,
}

impl std::fmt::Display for JMTError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JMTError::KeyNotFound(key) => write!(f, "Key not found: {:?}", key),
            JMTError::InvalidVersion(version) => write!(f, "Invalid version: {}", version),
            JMTError::InvalidProof => write!(f, "Invalid proof"),
            JMTError::TreeFull => write!(f, "Tree is full"),
            JMTError::InvalidKey => write!(f, "Invalid key"),
            JMTError::SerializationError => write!(f, "Serialization error"),
        }
    }
}

impl std::error::Error for JMTError {}

// Implementation for JellyfishMerkleTree
impl<const HEIGHT: usize, H: Hasher> JellyfishMerkleTree<HEIGHT, H> {
    pub fn new(version: u64) -> Self {
        Self {
            root: H::empty_hash(),
            leaves: BTreeMap::new(),
            version,
            _hasher: std::marker::PhantomData,
            node_cache: HashMap::new(),
            cache_limit: 1000,
        }
    }

    pub fn with_cache_limit(version: u64, cache_limit: usize) -> Self {
        Self {
            root: H::empty_hash(),
            leaves: BTreeMap::new(),
            version,
            _hasher: std::marker::PhantomData,
            node_cache: HashMap::new(),
            cache_limit,
        }
    }

    pub fn from_leaves<K, V>(version: u64, leaves: &[(K, V)]) -> Self
    where
        K: AsRef<[u8]>,
        V: AsRef<[u8]>,
    {
        let mut tree = Self::new(version);
        tree.batch_insert(leaves);
        tree
    }

    pub fn insert<K, V>(&mut self, key: K, data: V)
    where
        K: AsRef<[u8]>,
        V: AsRef<[u8]>,
    {
        let key_ref = key.as_ref();
        let data_ref = data.as_ref();
        let value_hash = H::hash_leaf_with_key(key_ref, data_ref);
        self.leaves.insert(key_ref.to_vec(), value_hash);
        self.update_root();
    }

    pub fn batch_insert<K, V>(&mut self, items: &[(K, V)])
    where
        K: AsRef<[u8]>,
        V: AsRef<[u8]>,
    {
        for (key, data) in items {
            let key_ref = key.as_ref();
            let data_ref = data.as_ref();
            let value_hash = H::hash_leaf_with_key(key_ref, data_ref);
            self.leaves.insert(key_ref.to_vec(), value_hash);
        }
        self.update_root();
    }

    pub fn update<K, V>(&mut self, key: K, data: V) -> Result<(), JMTError>
    where
        K: AsRef<[u8]>,
        V: AsRef<[u8]>,
    {
        let key_ref = key.as_ref();
        let key_vec = key_ref.to_vec();
        if self.leaves.contains_key(&key_vec) {
            let data_ref = data.as_ref();
            let value_hash = H::hash_leaf_with_key(key_ref, data_ref);
            self.leaves.insert(key_vec, value_hash);
            self.update_root();
            Ok(())
        } else {
            Err(JMTError::KeyNotFound(key_ref.to_vec()))
        }
    }

    pub fn remove<K: AsRef<[u8]>>(&mut self, key: K) -> bool {
        let key_vec = key.as_ref().to_vec();
        if self.leaves.remove(&key_vec).is_some() {
            self.update_root();
            true
        } else {
            false
        }
    }

    pub fn batch_remove<K>(&mut self, keys: &[K]) -> usize
    where
        K: AsRef<[u8]>,
    {
        let mut removed_count = 0;
        for key in keys {
            let key_vec = key.as_ref().to_vec();
            if self.leaves.remove(&key_vec).is_some() {
                removed_count += 1;
            }
        }

        if removed_count > 0 {
            self.update_root();
        }

        removed_count
    }

    pub fn get<K: AsRef<[u8]>>(&self, key: K) -> Result<H::Output, JMTError> {
        let key_vec = key.as_ref().to_vec();
        self.leaves
            .get(&key_vec)
            .copied()
            .ok_or_else(|| JMTError::KeyNotFound(key.as_ref().to_vec()))
    }

    pub fn contains<K: AsRef<[u8]>>(&self, key: K) -> bool {
        let key_vec = key.as_ref().to_vec();
        self.leaves.contains_key(&key_vec)
    }

    fn update_root(&mut self) {
        if self.leaves.is_empty() {
            self.root = H::empty_hash();
            return;
        }

        let mut current_level: Vec<(Vec<u8>, H::Output)> =
            self.leaves.iter().map(|(k, v)| (k.clone(), *v)).collect();

        let mut level = 0;
        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            let mut i = 0;

            while i < current_level.len() {
                if i + 1 < current_level.len() {
                    let (left_key, left_hash) = &current_level[i];
                    let (right_key, right_hash) = &current_level[i + 1];

                    let common_prefix = self.common_prefix_length(left_key, right_key, level);

                    if common_prefix > level {
                        let parent_hash = H::hash_internal_node(&[*left_hash, *right_hash]);
                        let parent_key = left_key[..common_prefix].to_vec();
                        next_level.push((parent_key, parent_hash));
                        i += 2;
                    } else {
                        next_level.push(current_level[i].clone());
                        i += 1;
                    }
                } else {
                    next_level.push(current_level[i].clone());
                    i += 1;
                }
            }

            current_level = next_level;
            level += 1;
        }

        self.root = current_level[0].1;
        self.clear_cache();
    }

    fn common_prefix_length(&self, key1: &[u8], key2: &[u8], level: usize) -> usize {
        let max_len = key1.len().min(key2.len());
        for i in level..max_len {
            if key1[i] != key2[i] {
                return i;
            }
        }
        max_len
    }

    pub fn prove<K: AsRef<[u8]>>(&self, key: K) -> JMTProof<H> {
        let key_slice = key.as_ref();
        let key_vec = key_slice.to_vec();
        let leaf_data = self.leaves.get(&key_vec).copied();
        let mut siblings = Vec::new();

        if let Some(_value_hash) = leaf_data {
            // Inclusion proof
            let current_key = key_slice;
            let mut remaining_leaves: Vec<(&Vec<u8>, &H::Output)> = self
                .leaves
                .iter()
                .filter(|(k, _)| k.as_slice() != key_slice)
                .collect();

            let mut level = 0;
            while !remaining_leaves.is_empty() {
                if let Some(sibling) = self.find_sibling(current_key, &remaining_leaves, level) {
                    siblings.push(*sibling.1);
                    remaining_leaves.retain(|(k, _)| k != &sibling.0);
                } else {
                    siblings.push(H::empty_hash());
                }
                level += 1;
            }
        } else {
            // Exclusion proof
            if let Some((closest_key, closest_hash)) = self.find_closest_leaf(key_slice) {
                siblings = self.generate_exclusion_proof(key_slice, closest_key, *closest_hash);
            }
        }

        JMTProof {
            leaf: leaf_data.map(|hash| (key_slice.to_vec(), hash)),
            siblings,
        }
    }

    pub fn prove_non_membership_with_range<K: AsRef<[u8]>>(
        &self,
        key: K,
    ) -> JMTNonMembershipProof<H> {
        let key_slice = key.as_ref();
        let (predecessor, successor) = self.find_adjacent_leaves(key_slice);

        JMTNonMembershipProof {
            key: key_slice.to_vec(),
            predecessor: predecessor.map(|(k, v)| (k.clone(), *v)),
            successor: successor.map(|(k, v)| (k.clone(), *v)),
            root: self.root,
        }
    }

    fn find_sibling<'a>(
        &self,
        key: &[u8],
        leaves: &[(&'a Vec<u8>, &'a H::Output)],
        level: usize,
    ) -> Option<(&'a Vec<u8>, &'a H::Output)> {
        leaves
            .iter()
            .find(|(k, _)| {
                if k.len() > level && key.len() > level {
                    k[..level] == key[..level] && k[level] != key[level]
                } else {
                    false
                }
            })
            .map(|(k, v)| (*k, *v))
    }

    fn find_closest_leaf(&self, key: &[u8]) -> Option<(&Vec<u8>, &H::Output)> {
        self.leaves.iter().min_by_key(|(k, _)| {
            let common_prefix = self.common_prefix_length(key, k, 0);
            std::cmp::Reverse(common_prefix)
        })
    }

    fn generate_exclusion_proof(
        &self,
        query_key: &[u8],
        closest_key: &[u8],
        closest_hash: H::Output,
    ) -> Vec<H::Output> {
        let common_prefix = self.common_prefix_length(query_key, closest_key, 0);
        let mut siblings = Vec::new();

        for level in common_prefix.. {
            if level >= query_key.len() || level >= closest_key.len() {
                break;
            }

            if query_key[level] != closest_key[level] {
                siblings.push(closest_hash);
                break;
            }
        }

        siblings
    }

    fn find_adjacent_leaves(
        &self,
        key: &[u8],
    ) -> (
        Option<(&Vec<u8>, &H::Output)>,
        Option<(&Vec<u8>, &H::Output)>,
    ) {
        use std::ops::Bound;

        let key_vec = key.to_vec();
        let mut predecessor = None;
        let mut successor = None;

        // Find predecessor (greatest key < query key)
        for (k, v) in self.leaves.range(..key_vec.clone()).rev().take(1) {
            predecessor = Some((k, v));
            break;
        }

        // Find successor (smallest key > query key)
        for (k, v) in self
            .leaves
            .range((Bound::Excluded(key_vec), Bound::Unbounded))
            .take(1)
        {
            successor = Some((k, v));
            break;
        }

        (predecessor, successor)
    }

    // Cache methods
    fn get_cached_node(&self, key: &[u8]) -> Option<H::Output> {
        self.node_cache.get(key).copied()
    }

    fn cache_node(&mut self, key: Vec<u8>, hash: H::Output) {
        if self.node_cache.len() >= self.cache_limit {
            if let Some(random_key) = self.node_cache.keys().next().cloned() {
                self.node_cache.remove(&random_key);
            }
        }
        self.node_cache.insert(key, hash);
    }

    fn clear_cache(&mut self) {
        self.node_cache.clear();
    }

    // Getters
    pub fn root(&self) -> H::Output {
        self.root
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }
}

// JMTProof implementation
impl<H: Hasher> JMTProof<H> {
    pub fn verify<K, V>(&self, root: H::Output, key: K, value: Option<V>) -> bool
    where
        K: AsRef<[u8]>,
        V: AsRef<[u8]>,
    {
        let key_ref = key.as_ref();
        match (&self.leaf, value) {
            (Some((leaf_key, leaf_hash)), Some(value_data)) => {
                // Inclusion proof verification
                if leaf_key != key_ref {
                    return false;
                }

                let computed_hash = H::hash_leaf_with_key(key_ref, value_data.as_ref());
                if computed_hash != *leaf_hash {
                    return false;
                }

                self.verify_siblings(*leaf_hash, root, key_ref)
            }
            (Some((leaf_key, leaf_hash)), None) => {
                // Exclusion proof with existing leaf
                if leaf_key == key_ref {
                    return false;
                }
                self.verify_siblings(*leaf_hash, root, leaf_key)
            }
            (None, None) => {
                // Exclusion proof with empty subtree
                self.verify_siblings(H::empty_hash(), root, key_ref)
            }
            (None, Some(_)) => false,
        }
    }

    pub fn verify_enhanced<K, V>(
        &self,
        root: H::Output,
        key: K,
        value: Option<V>,
        version: u64,
    ) -> bool
    where
        K: AsRef<[u8]>,
        V: AsRef<[u8]>,
    {
        let key_ref = key.as_ref();
        if !self.verify(root, key_ref, value.as_ref().map(|v| v.as_ref())) {
            return false;
        }

        // Additional security checks
        if self.siblings.len() > 256 {
            return false;
        }

        let expected_path_length = key_ref.len() * 8;
        if self.siblings.len() > expected_path_length {
            return false;
        }

        if let Some((leaf_key, leaf_hash)) = &self.leaf {
            if value.is_some() && leaf_key != key_ref {
                return false;
            }

            if let Some(data) = value {
                let leaf_data_hash = H::hash_leaf_with_key(key_ref, data.as_ref());
                let computed_hash = H::hash_with_version(version, leaf_data_hash.as_ref());
                if computed_hash != *leaf_hash {
                    return false;
                }
            }
        }

        true
    }

    fn verify_siblings(&self, mut current_hash: H::Output, root: H::Output, key: &[u8]) -> bool {
        let mut level = 0;

        for sibling in &self.siblings {
            let bit_index = level / 8;
            let bit_offset = 7 - (level % 8);

            if bit_index < key.len() {
                let byte = key[bit_index];
                let is_right = (byte >> bit_offset) & 1 == 1;

                current_hash = if is_right {
                    H::hash_internal_node(&[*sibling, current_hash])
                } else {
                    H::hash_internal_node(&[current_hash, *sibling])
                };
            } else {
                current_hash = H::hash_internal_node(&[current_hash, *sibling]);
            }

            level += 1;
        }

        current_hash == root
    }

    pub fn proof_size(&self) -> usize {
        std::mem::size_of::<Option<(Vec<u8>, H::Output)>>()
            + self.siblings.len() * std::mem::size_of::<H::Output>()
    }
}

// JMTNonMembershipProof implementation
impl<H: Hasher> JMTNonMembershipProof<H> {
    pub fn verify(&self, root: H::Output) -> bool {
        if self.root != root {
            return false;
        }

        match (&self.predecessor, &self.successor) {
            (Some((pred_key, _)), Some((succ_key, _))) => {
                pred_key < &self.key && &self.key < succ_key
            }
            (Some((pred_key, _)), None) => pred_key < &self.key,
            (None, Some((succ_key, _))) => &self.key < succ_key,
            (None, None) => true,
        }
    }
}

// VersionedJMTStorage implementation
impl<H: Hasher> VersionedJMTStorage<H> {
    pub fn new() -> Self {
        Self {
            trees: BTreeMap::new(),
            deltas: BTreeMap::new(),
            current_version: 0,
            _hasher: std::marker::PhantomData,
        }
    }

    pub fn create_new_version(&mut self, root: H::Output) -> u64 {
        self.current_version += 1;
        self.trees.insert(self.current_version, root);
        self.current_version
    }

    pub fn create_version_with_delta(&mut self, delta: JMTDelta<H>, root: H::Output) -> u64 {
        self.current_version += 1;
        self.trees.insert(self.current_version, root);
        self.deltas.insert(self.current_version, delta);
        self.current_version
    }

    pub fn get_root(&self, version: u64) -> Option<H::Output> {
        self.trees.get(&version).copied()
    }

    pub fn get_delta(&self, version: u64) -> Option<&JMTDelta<H>> {
        self.deltas.get(&version)
    }

    pub fn reconstruct_tree<const HEIGHT: usize>(
        &self,
        version: u64,
    ) -> Option<JellyfishMerkleTree<HEIGHT, H>> {
        if version > self.current_version {
            return None;
        }

        let mut tree = JellyfishMerkleTree::<HEIGHT, H>::new(0);

        for v in 1..=version {
            if let Some(delta) = self.deltas.get(&v) {
                for (key, value_hash) in &delta.inserted {
                    tree.leaves.insert(key.clone(), *value_hash);
                }

                for key in &delta.deleted {
                    tree.leaves.remove(key);
                }
            }
        }

        tree.root = self.get_root(version)?;
        tree.version = version;

        Some(tree)
    }

    pub fn prune_with_compaction(&mut self, keep_latest: usize) {
        if self.trees.len() <= keep_latest {
            return;
        }

        let versions_to_remove: Vec<u64> = self
            .trees
            .keys()
            .take(self.trees.len() - keep_latest)
            .cloned()
            .collect();

        for version in versions_to_remove {
            self.trees.remove(&version);
            self.deltas.remove(&version);
        }
    }

    pub fn current_version(&self) -> u64 {
        self.current_version
    }

    pub fn version_count(&self) -> usize {
        self.trees.len()
    }
}

// Database trait and implementation
pub trait JMTDatabase<H: Hasher> {
    fn get_root(&self, version: u64) -> Result<Option<H::Output>, JMTError>;
    fn set_root(&mut self, version: u64, root: H::Output) -> Result<(), JMTError>;
    fn get_leaf<K: AsRef<[u8]>>(&self, key: K) -> Result<Option<H::Output>, JMTError>;
    fn set_leaf<K: AsRef<[u8]>>(&mut self, key: K, hash: H::Output) -> Result<(), JMTError>;
    fn delete_leaf<K: AsRef<[u8]>>(&mut self, key: K) -> Result<bool, JMTError>;
    fn get_version(&self) -> Result<u64, JMTError>;
    fn set_version(&mut self, version: u64) -> Result<(), JMTError>;
}

#[derive(Debug)]
pub struct MemoryJMTDatabase<H: Hasher> {
    roots: HashMap<u64, H::Output>,
    leaves: BTreeMap<Vec<u8>, H::Output>,
    current_version: u64,
    _hasher: std::marker::PhantomData<H>,
}

impl<H: Hasher> MemoryJMTDatabase<H> {
    pub fn new() -> Self {
        Self {
            roots: HashMap::new(),
            leaves: BTreeMap::new(),
            current_version: 0,
            _hasher: std::marker::PhantomData,
        }
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

    fn get_leaf<K: AsRef<[u8]>>(&self, key: K) -> Result<Option<H::Output>, JMTError> {
        let key_vec = key.as_ref().to_vec();
        Ok(self.leaves.get(&key_vec).copied())
    }

    fn set_leaf<K: AsRef<[u8]>>(&mut self, key: K, hash: H::Output) -> Result<(), JMTError> {
        self.leaves.insert(key.as_ref().to_vec(), hash);
        Ok(())
    }

    fn delete_leaf<K: AsRef<[u8]>>(&mut self, key: K) -> Result<bool, JMTError> {
        let key_vec = key.as_ref().to_vec();
        Ok(self.leaves.remove(&key_vec).is_some())
    }

    fn get_version(&self) -> Result<u64, JMTError> {
        Ok(self.current_version)
    }

    fn set_version(&mut self, version: u64) -> Result<(), JMTError> {
        self.current_version = version;
        Ok(())
    }
}

// Example SHA256 Hasher Implementation with JMT optimizations
#[derive(Debug, Clone)]
pub struct Sha256Hasher;

impl Hasher for Sha256Hasher {
    type Output = [u8; 32];

    fn hash_leaf(data: &[u8]) -> Self::Output {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"leaf:");
        hasher.update(data);
        hasher.finalize().into()
    }

    fn hash_node(left: &Self::Output, right: &Self::Output) -> Self::Output {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"node:");
        hasher.update(left);
        hasher.update(right);
        hasher.finalize().into()
    }

    fn empty_hash() -> Self::Output {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"jmt_empty:");
        hasher.update([0u8; 32]);
        hasher.finalize().into()
    }

    fn hash_leaf_with_key(key: &[u8], data: &[u8]) -> Self::Output {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"leaf:");
        hasher.update(Self::commit_key(key));
        hasher.update(data);
        hasher.finalize().into()
    }

    fn hash_internal_node(children: &[Self::Output]) -> Self::Output {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"node:");
        hasher.update((children.len() as u64).to_le_bytes());
        for child in children {
            hasher.update(child);
        }
        hasher.finalize().into()
    }

    fn hash_with_version(version: u64, data: &[u8]) -> Self::Output {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"version:");
        hasher.update(version.to_le_bytes());
        hasher.update(data);
        hasher.finalize().into()
    }

    fn commit_key(key: &[u8]) -> Self::Output {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"key_commitment:");
        hasher.update(key);
        hasher.finalize().into()
    }
}

// Display implementations
impl<const HEIGHT: usize, H: Hasher> Display for JellyfishMerkleTree<HEIGHT, H>
where
    H::Output: AsRef<[u8]>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "🎯 Jellyfish Merkle Tree")?;
        writeln!(f, "├── Version: {}", self.version)?;
        writeln!(f, "├── Root: {}", hex::encode(self.root.as_ref()))?;
        writeln!(f, "├── Leaves: {}", self.leaf_count())?;
        writeln!(f, "├── Height: {}", HEIGHT)?;
        writeln!(f, "└── Cache size: {}", self.node_cache.len())?;
        Ok(())
    }
}

impl<H: Hasher> Display for JMTProof<H>
where
    H::Output: AsRef<[u8]>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "🔐 JMT Proof")?;
        writeln!(
            f,
            "├── Leaf: {}",
            if self.leaf.is_some() { "Some" } else { "None" }
        )?;
        writeln!(f, "├── Siblings: {}", self.siblings.len())?;
        writeln!(f, "└── Proof size: {} bytes", self.proof_size())?;

        if let Some((key, hash)) = &self.leaf {
            writeln!(f, "    Leaf Key: {}", hex::encode(key))?;
            writeln!(f, "    Leaf Hash: {}", hex::encode(hash.as_ref()))?;
        }

        Ok(())
    }
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jellyfish_merkle_tree_basic() {
        type JMT = JellyfishMerkleTree<256, Sha256Hasher>;

        let mut tree = JMT::new(1);

        // Test insertion
        tree.insert(b"key1", b"value1");
        tree.insert("key2", "value2"); // Using string slices

        assert_eq!(tree.leaf_count(), 2);
        assert!(tree.contains(b"key1"));
        assert!(tree.contains("key2")); // Using string slice
        assert!(!tree.contains(b"key3"));

        // Test retrieval
        let hash1 = tree.get(b"key1").unwrap();
        let expected_hash1 = Sha256Hasher::hash_leaf_with_key(b"key1", b"value1");
        assert_eq!(hash1, expected_hash1);

        // Test update
        tree.update(b"key1", b"new_value1").unwrap();
        let new_hash1 = tree.get(b"key1").unwrap();
        let expected_new_hash1 = Sha256Hasher::hash_leaf_with_key(b"key1", b"new_value1");
        assert_eq!(new_hash1, expected_new_hash1);

        // Test removal
        assert!(tree.remove("key2")); // Using string slice
        assert!(!tree.contains(b"key2"));
        assert_eq!(tree.leaf_count(), 1);
    }

    #[test]
    fn test_jmt_proof_inclusion() {
        type JMT = JellyfishMerkleTree<256, Sha256Hasher>;

        let mut tree = JMT::new(1);
        tree.insert(b"key1", b"value1");
        tree.insert(b"key2", b"value2");

        let proof = tree.prove(b"key1");
        assert!(proof.verify(tree.root(), b"key1", Some(b"value1")));
        assert!(!proof.verify(tree.root(), b"key1", Some(b"wrong_value")));
        assert!(!proof.verify(tree.root(), b"key2", Some(b"value2")));

        // Test with string slices
        assert!(proof.verify(tree.root(), "key1", Some("value1")));
    }

    #[test]
    fn test_jmt_proof_exclusion() {
        type JMT = JellyfishMerkleTree<256, Sha256Hasher>;

        let mut tree = JMT::new(1);
        tree.insert(b"key1", b"value1");
        tree.insert(b"key3", b"value3");

        let proof = tree.prove(b"key2");
        assert!(proof.verify(tree.root(), b"key2", None::<&[u8]>));

        // Test with string slice
        assert!(proof.verify(tree.root(), "key2", None::<&[u8]>));
    }

    #[test]
    fn test_jmt_non_membership_proof() {
        type JMT = JellyfishMerkleTree<256, Sha256Hasher>;

        let mut tree = JMT::new(1);
        tree.insert(b"key1", b"value1");
        tree.insert(b"key3", b"value3");

        let proof = tree.prove_non_membership_with_range(b"key2");
        assert!(proof.verify(tree.root()));

        let proof2 = tree.prove_non_membership_with_range("key0"); // Using string slice
        assert!(proof2.verify(tree.root()));

        let proof3 = tree.prove_non_membership_with_range(b"key4");
        assert!(proof3.verify(tree.root()));
    }

    #[test]
    fn test_batch_operations() {
        type JMT = JellyfishMerkleTree<256, Sha256Hasher>;

        let mut tree = JMT::new(1);

        let items = [
            (b"key1", b"value1"),
            (b"key2", b"value2"), // Using string slices
            (b"key3", b"value3"),
        ];

        tree.batch_insert(&items);
        assert_eq!(tree.leaf_count(), 3);

        let keys_to_remove = [b"key1", b"key3"]; // Mixed types
        let removed = tree.batch_remove(&keys_to_remove);
        assert_eq!(removed, 2);
        assert_eq!(tree.leaf_count(), 1);
        assert!(tree.contains(b"key2"));
    }

    #[test]
    fn test_versioned_storage() {
        type JMT = JellyfishMerkleTree<256, Sha256Hasher>;

        let mut storage = VersionedJMTStorage::<Sha256Hasher>::new();

        // Create version 1
        let mut tree1 = JMT::new(1);
        tree1.insert(b"key1", b"value1");
        storage.create_new_version(tree1.root());

        // Create version 2 with delta
        let mut tree2 = tree1;
        tree2.insert(b"key2", b"value2");

        let delta = JMTDelta {
            inserted: [(b"key2".to_vec(), tree2.get(b"key2").unwrap())]
                .into_iter()
                .collect(),
            deleted: BTreeSet::new(),
        };

        storage.create_version_with_delta(delta, tree2.root());

        assert_eq!(storage.current_version(), 2);
        assert_eq!(storage.version_count(), 2);

        // Test reconstruction
        let reconstructed = storage.reconstruct_tree::<256>(2).unwrap();
        assert_eq!(reconstructed.root(), tree2.root());
        assert_eq!(reconstructed.leaf_count(), 2);

        // Test pruning
        storage.prune_with_compaction(1);
        assert_eq!(storage.version_count(), 1);
    }

    #[test]
    fn test_memory_database() {
        let mut db = MemoryJMTDatabase::<Sha256Hasher>::new();

        assert!(db.set_root(1, [1u8; 32]).is_ok());
        assert!(db.set_leaf(b"key1", [2u8; 32]).is_ok());
        assert!(db.set_leaf("key2", [3u8; 32]).is_ok()); // Using string slice

        assert_eq!(db.get_root(1).unwrap(), Some([1u8; 32]));
        assert_eq!(db.get_leaf(b"key1").unwrap(), Some([2u8; 32]));
        assert_eq!(db.get_leaf("key2").unwrap(), Some([3u8; 32])); // Using string slice
        assert!(db.get_leaf(b"key3").unwrap().is_none());

        assert!(db.delete_leaf(b"key1").unwrap());
        assert!(db.delete_leaf("key2").unwrap()); // Using string slice
        assert!(!db.delete_leaf(b"key1").unwrap());
    }

    #[test]
    fn test_enhanced_verification() {
        type JMT = JellyfishMerkleTree<256, Sha256Hasher>;

        let mut tree = JMT::new(1);
        tree.insert(b"key1", b"value1");

        let proof = tree.prove(b"key1");
        assert!(proof.verify_enhanced(tree.root(), b"key1", Some(b"value1"), 1));

        // Test with string slices
        assert!(proof.verify_enhanced(tree.root(), "key1", Some("value1"), 1));

        // Test with wrong version
        assert!(!proof.verify_enhanced(tree.root(), b"key1", Some(b"value1"), 2));
    }

    #[test]
    fn test_empty_tree() {
        type JMT = JellyfishMerkleTree<256, Sha256Hasher>;

        let tree = JMT::new(1);
        assert!(tree.is_empty());
        assert_eq!(tree.root(), Sha256Hasher::empty_hash());
        assert_eq!(tree.leaf_count(), 0);

        let proof = tree.prove(b"key1");
        assert!(proof.verify(tree.root(), b"key1", None::<&[u8]>));

        // Test with string slice
        assert!(proof.verify(tree.root(), "key1", None::<&[u8]>));
    }

    #[test]
    fn test_cache_functionality() {
        type JMT = JellyfishMerkleTree<256, Sha256Hasher>;

        let mut tree = JMT::with_cache_limit(1, 2);
        tree.insert(b"key1", b"value1");
        tree.insert(b"key2", b"value2");

        // Cache should be cleared after update_root
        assert!(tree.node_cache.is_empty());
    }
}
