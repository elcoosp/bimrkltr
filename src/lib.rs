// src/lib.rs
use core::fmt;
use std::fmt::{Display, Formatter};

pub trait Hasher {
    type Output: Copy + PartialEq + Default + AsRef<[u8]>;

    fn hash_leaf(data: &[u8]) -> Self::Output;
    fn hash_node(left: &Self::Output, right: &Self::Output) -> Self::Output;
}

#[derive(Clone)]
pub struct MerkleTree<const HEIGHT: usize, H: Hasher> {
    nodes: Vec<H::Output>,
    _hasher: std::marker::PhantomData<H>,
}

#[derive(Clone)]
pub struct MerkleProof<const HEIGHT: usize, H: Hasher> {
    path: [Option<H::Output>; HEIGHT],
    leaf_index: usize,
    _hasher: std::marker::PhantomData<H>,
}

// Manual Debug implementation to avoid requiring H::Output: Debug
impl<const HEIGHT: usize, H: Hasher> fmt::Debug for MerkleTree<HEIGHT, H> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("MerkleTree")
            .field("height", &HEIGHT)
            .field("leaf_count", &self.leaf_count())
            .field("root", &"<hash>")
            .finish()
    }
}

impl<const HEIGHT: usize, H: Hasher> fmt::Debug for MerkleProof<HEIGHT, H> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("MerkleProof")
            .field("leaf_index", &self.leaf_index)
            .field("path_length", &self.path_length())
            .finish()
    }
}

// For bidirectional trees where deepest leaf is another tree's root
pub struct BidirectionalMerkleTree<const HEIGHT: usize, const MAX_CHILDREN: usize, H: Hasher> {
    main_tree: MerkleTree<HEIGHT, H>,
    child_trees: [Option<Box<MerkleTree<HEIGHT, H>>>; MAX_CHILDREN],
}

// Manual Debug implementation for BidirectionalMerkleTree
impl<const HEIGHT: usize, const MAX_CHILDREN: usize, H: Hasher> fmt::Debug
    for BidirectionalMerkleTree<HEIGHT, MAX_CHILDREN, H>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BidirectionalMerkleTree")
            .field("height", &HEIGHT)
            .field("max_children", &MAX_CHILDREN)
            .field("child_tree_count", &self.child_tree_count())
            .finish()
    }
}

impl<const HEIGHT: usize, H: Hasher> MerkleTree<HEIGHT, H> {
    const TOTAL_NODES: usize = if HEIGHT > 0 {
        (1 << (HEIGHT + 1)) - 1
    } else {
        1
    };
    const LEAF_COUNT: usize = if HEIGHT > 0 { 1 << HEIGHT } else { 1 };

    pub fn new() -> Self {
        Self {
            nodes: vec![H::Output::default(); Self::TOTAL_NODES],
            _hasher: std::marker::PhantomData,
        }
    }

    pub fn from_leaves(leaves: &[[u8; 32]]) -> Self {
        assert_eq!(leaves.len(), Self::LEAF_COUNT);

        let mut tree = Self::new();

        // Handle the case where HEIGHT is 0 (single node tree)
        if HEIGHT == 0 {
            tree.nodes[0] = H::hash_leaf(&leaves[0]);
            return tree;
        }

        // Fill leaves
        let leaf_offset = (1 << HEIGHT) - 1;
        for (i, leaf_data) in leaves.iter().enumerate() {
            tree.nodes[leaf_offset + i] = H::hash_leaf(leaf_data);
        }

        // Build tree bottom-up, starting from the level above leaves
        // We go from HEIGHT-1 down to 0 (but level 0 is the root and has no parent)
        for level in (0..HEIGHT).rev() {
            let level_start = (1 << level) - 1;
            let level_count = 1 << level;
            let level_end = level_start + level_count;

            // Only process if we have at least 2 nodes in this level
            if level_count >= 2 {
                for i in (level_start..level_end).step_by(2) {
                    // Ensure we don't go out of bounds
                    if i + 1 < level_end && i + 1 < tree.nodes.len() {
                        let left = &tree.nodes[i];
                        let right = &tree.nodes[i + 1];
                        let parent_index = (i - 1) / 2;
                        if parent_index < tree.nodes.len() {
                            tree.nodes[parent_index] = H::hash_node(left, right);
                        }
                    }
                }
            }
        }

        tree
    }

    pub fn root(&self) -> H::Output {
        self.nodes[0]
    }

    pub fn update_leaf(&mut self, index: usize, data: &[u8]) {
        if HEIGHT == 0 {
            // Single node tree
            self.nodes[0] = H::hash_leaf(data);
            return;
        }

        let leaf_offset = (1 << HEIGHT) - 1;
        let mut pos = leaf_offset + index;
        self.nodes[pos] = H::hash_leaf(data);

        // Update path to root
        while pos > 0 {
            let parent = (pos - 1) / 2;
            let left_child = parent * 2 + 1;
            let right_child = parent * 2 + 2;

            // Ensure we don't go out of bounds
            if right_child < self.nodes.len() {
                self.nodes[parent] =
                    H::hash_node(&self.nodes[left_child], &self.nodes[right_child]);
            }
            pos = parent;
        }
    }

    pub fn get_leaf(&self, index: usize) -> H::Output {
        if HEIGHT == 0 {
            return self.nodes[0];
        }
        let leaf_offset = (1 << HEIGHT) - 1;
        self.nodes[leaf_offset + index]
    }

    pub fn prove(&self, leaf_index: usize) -> MerkleProof<HEIGHT, H> {
        let mut path = [None; HEIGHT];

        if HEIGHT == 0 {
            return MerkleProof {
                path,
                leaf_index,
                _hasher: std::marker::PhantomData,
            };
        }

        let leaf_offset = (1 << HEIGHT) - 1;
        let mut pos = leaf_offset + leaf_index;

        for level in 0..HEIGHT {
            // Calculate sibling position
            let sibling = if pos % 2 == 1 { pos + 1 } else { pos - 1 };

            // Only store sibling if it's within bounds
            if sibling < self.nodes.len() {
                path[level] = Some(self.nodes[sibling]);
            }

            // Move to parent
            pos = (pos - 1) / 2;
        }

        MerkleProof {
            path,
            leaf_index,
            _hasher: std::marker::PhantomData,
        }
    }

    pub fn height(&self) -> usize {
        HEIGHT
    }

    pub fn leaf_count(&self) -> usize {
        Self::LEAF_COUNT
    }
}

impl<const HEIGHT: usize, H: Hasher> MerkleProof<HEIGHT, H> {
    pub fn verify(&self, leaf_data: &[u8], root: H::Output) -> bool {
        let mut hash = H::hash_leaf(leaf_data);
        let mut pos = self.leaf_index;

        for sibling_hash in self.path.iter() {
            if let Some(sibling) = sibling_hash {
                if pos % 2 == 0 {
                    // Current node is left child
                    hash = H::hash_node(&hash, sibling);
                } else {
                    // Current node is right child
                    hash = H::hash_node(sibling, &hash);
                }
                pos = pos / 2;
            } else {
                // No sibling at this level, can't continue
                return false;
            }
        }

        hash == root
    }

    pub fn path_length(&self) -> usize {
        self.path.iter().filter(|x| x.is_some()).count()
    }
}

impl<const HEIGHT: usize, const MAX_CHILDREN: usize, H: Hasher>
    BidirectionalMerkleTree<HEIGHT, MAX_CHILDREN, H>
{
    pub fn new() -> Self {
        Self {
            main_tree: MerkleTree::new(),
            child_trees: [const { None }; MAX_CHILDREN],
        }
    }

    pub fn link_child_tree(&mut self, leaf_index: usize, child_tree: MerkleTree<HEIGHT, H>) {
        // The deepest leaf becomes the root of the child tree
        let child_root = child_tree.root();

        // Convert hash to bytes for storage in the main tree leaf
        let root_bytes = child_root.as_ref().to_vec();
        self.main_tree.update_leaf(leaf_index, &root_bytes);
        self.child_trees[leaf_index] = Some(Box::new(child_tree));
    }

    pub fn get_child_root(&self, leaf_index: usize) -> Option<H::Output> {
        Some(self.child_trees.get(leaf_index)?.as_ref()?.as_ref().root())
    }

    pub fn verify_bidirectional_path(
        &self,
        main_leaf_index: usize,
        main_leaf_data: &[u8],
        child_leaf_index: usize,
        child_leaf_data: &[u8],
    ) -> bool {
        // First verify the main tree path
        let main_proof = self.main_tree.prove(main_leaf_index);
        if !main_proof.verify(main_leaf_data, self.main_tree.root()) {
            return false;
        }

        // Then verify the child tree exists and the path within it
        if let Some(child_tree) = &self.child_trees[main_leaf_index] {
            let child_proof = child_tree.prove(child_leaf_index);
            child_proof.verify(child_leaf_data, child_tree.root())
        } else {
            false
        }
    }

    pub fn main_tree(&self) -> &MerkleTree<HEIGHT, H> {
        &self.main_tree
    }

    pub fn child_tree_count(&self) -> usize {
        self.child_trees.iter().filter(|t| t.is_some()).count()
    }
}

// Beautiful Display implementations
impl<const HEIGHT: usize, H: Hasher> Display for MerkleTree<HEIGHT, H>
where
    H::Output: AsRef<[u8]>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "🌳 Merkle Tree (height: {})", HEIGHT)?;
        writeln!(f, "├── Root: {}", hex::encode(self.root().as_ref()))?;
        writeln!(f, "├── Leaves: {}", self.leaf_count())?;
        writeln!(f, "├── Total Nodes: {}", self.nodes.len())?;

        // Show first few leaves
        let show_leaves = 3.min(self.leaf_count());
        writeln!(f, "└── First {} leaves:", show_leaves)?;
        for i in 0..show_leaves {
            writeln!(f, "    {}: {}", i, hex::encode(self.get_leaf(i).as_ref()))?;
        }

        Ok(())
    }
}

impl<const HEIGHT: usize, H: Hasher> Display for MerkleProof<HEIGHT, H>
where
    H::Output: AsRef<[u8]>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "🔐 Merkle Proof")?;
        writeln!(f, "├── Leaf Index: {}", self.leaf_index)?;
        writeln!(f, "├── Path Length: {}", self.path_length())?;
        writeln!(f, "└── Sibling Hashes:")?;

        for (i, hash_opt) in self.path.iter().enumerate() {
            if let Some(hash) = hash_opt {
                writeln!(f, "    Level {}: {}", i, hex::encode(hash.as_ref()))?;
            }
        }

        Ok(())
    }
}

impl<const HEIGHT: usize, const MAX_CHILDREN: usize, H: Hasher> Display
    for BidirectionalMerkleTree<HEIGHT, MAX_CHILDREN, H>
where
    H::Output: AsRef<[u8]>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "🔄 Bidirectional Merkle Tree")?;
        writeln!(f, "├── Main Tree:")?;
        write!(f, "{}", self.main_tree)?;
        writeln!(
            f,
            "├── Child Trees: {}/{}",
            self.child_tree_count(),
            MAX_CHILDREN
        )?;

        // Show linked child trees
        for (i, child_opt) in self.child_trees.iter().enumerate() {
            if let Some(child) = child_opt {
                writeln!(
                    f,
                    "│   ├── Leaf {} → Child Root: {}",
                    i,
                    hex::encode(child.root().as_ref())
                )?;
            }
        }

        writeln!(f, "└── Configuration:")?;
        writeln!(f, "    Height: {}, Max Children: {}", HEIGHT, MAX_CHILDREN)?;

        Ok(())
    }
}

// Example SHA256 Hasher Implementation
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
}

// Simple test hasher for predictable results
#[derive(Debug, Clone)]
pub struct TestHasher;

impl Hasher for TestHasher {
    type Output = [u8; 32];

    fn hash_leaf(data: &[u8]) -> Self::Output {
        let mut result = [0u8; 32];
        let hash_val = data.iter().fold(0u8, |acc, &x| acc.wrapping_add(x));
        result[0] = hash_val;
        result[1] = hash_val.wrapping_mul(2);
        result
    }

    fn hash_node(left: &Self::Output, right: &Self::Output) -> Self::Output {
        let mut result = [0u8; 32];
        for i in 0..32 {
            result[i] = left[i].wrapping_add(right[i]);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_snapshot;

    // Helper function to create test data
    fn create_test_leaves<const N: usize>(start: u8) -> [[u8; 32]; N] {
        let mut leaves = [[0u8; 32]; N];
        for i in 0..N {
            leaves[i] = [start + i as u8; 32];
        }
        leaves
    }

    #[test]
    fn test_merkle_tree_construction() {
        type TestTree = MerkleTree<2, TestHasher>; // 4 leaves

        let leaves = create_test_leaves::<4>(1);
        let tree = TestTree::from_leaves(&leaves);

        assert_eq!(tree.leaf_count(), 4);
        assert_eq!(tree.height(), 2);
    }

    #[test]
    fn test_single_node_tree() {
        type SingleNodeTree = MerkleTree<0, TestHasher>; // 1 leaf (root)

        let leaves = [[1u8; 32]];
        let tree = SingleNodeTree::from_leaves(&leaves);

        assert_eq!(tree.leaf_count(), 1);
        assert_eq!(tree.height(), 0);
        assert_eq!(tree.root(), TestHasher::hash_leaf(&[1u8; 32]));
    }

    #[test]
    fn test_merkle_tree_update() {
        type TestTree = MerkleTree<2, TestHasher>; // 4 leaves

        let leaves = create_test_leaves::<4>(1);
        let mut tree = TestTree::from_leaves(&leaves);
        let original_root = tree.root();

        // Update a leaf
        tree.update_leaf(1, &[10u8; 32]);
        let new_root = tree.root();

        assert_ne!(original_root, new_root);

        // Verify the updated leaf
        let updated_leaf = tree.get_leaf(1);
        let expected_hash = TestHasher::hash_leaf(&[10u8; 32]);
        assert_eq!(updated_leaf, expected_hash);
    }

    #[test]
    fn test_merkle_proof() {
        type TestTree = MerkleTree<2, TestHasher>; // 4 leaves

        let leaves = [[1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32]];
        let tree = TestTree::from_leaves(&leaves);

        // Generate proof for leaf 1
        let proof = tree.prove(1);

        // Verify proof with correct data
        assert!(proof.verify(&[2u8; 32], tree.root()));

        // Invalid data should fail
        assert!(!proof.verify(&[99u8; 32], tree.root()));
    }

    #[test]
    fn test_merkle_proof_all_leaves() {
        type TestTree = MerkleTree<2, TestHasher>;

        let leaves = create_test_leaves::<4>(1);
        let tree = TestTree::from_leaves(&leaves);

        // Test proof for each leaf
        for i in 0..4 {
            let proof = tree.prove(i);
            assert!(proof.verify(&leaves[i], tree.root()));
        }
    }

    #[test]
    fn test_single_node_proof() {
        type SingleNodeTree = MerkleTree<0, TestHasher>;

        let leaves = [[1u8; 32]];
        let tree = SingleNodeTree::from_leaves(&leaves);

        let proof = tree.prove(0);
        assert_eq!(proof.path_length(), 0);
        assert!(proof.verify(&[1u8; 32], tree.root()));
    }

    #[test]
    fn test_merkle_tree_display() {
        type TestTree = MerkleTree<2, TestHasher>;

        let leaves = create_test_leaves::<4>(1);
        let tree = TestTree::from_leaves(&leaves);

        let display_output = format!("{}", tree);
        assert!(display_output.contains("Merkle Tree"));
        assert!(display_output.contains("Root:"));
        assert!(display_output.contains("Leaves: 4"));
    }

    #[test]
    fn test_bidirectional_tree() {
        type MainTree = MerkleTree<2, TestHasher>;
        type ChildTree = MerkleTree<2, TestHasher>;
        type BiTree = BidirectionalMerkleTree<2, 4, TestHasher>;

        let main_leaves = create_test_leaves::<4>(1);
        let child_leaves = create_test_leaves::<4>(5);

        let _main_tree = MainTree::from_leaves(&main_leaves);
        let child_tree = ChildTree::from_leaves(&child_leaves);

        let mut bi_tree = BiTree::new();
        bi_tree.link_child_tree(1, child_tree);

        assert_eq!(bi_tree.child_tree_count(), 1);

        // Should be able to get child root
        let child_root = bi_tree.get_child_root(1);
        assert!(child_root.is_some());
    }

    #[test]
    fn test_bidirectional_verification() {
        type MainTree = MerkleTree<2, TestHasher>;
        type ChildTree = MerkleTree<2, TestHasher>;
        type BiTree = BidirectionalMerkleTree<2, 4, TestHasher>;

        let main_leaves = create_test_leaves::<4>(1);
        let child_leaves = create_test_leaves::<4>(5);

        let main_tree = MainTree::from_leaves(&main_leaves);
        let child_tree = ChildTree::from_leaves(&child_leaves);
        let child_tree_clone = child_tree.clone();

        let mut bi_tree = BiTree::new();
        bi_tree.link_child_tree(0, child_tree);

        // Get the actual leaf data from the main tree after linking
        let main_leaf_data = bi_tree.main_tree().get_leaf(0);

        // Verify bidirectional path
        let result = bi_tree.verify_bidirectional_path(
            0,                                      // main leaf index
            main_leaf_data.as_ref(),                // main leaf data (from updated tree)
            1,                                      // child leaf index
            &child_tree_clone.get_leaf(1).as_ref(), // child leaf data
        );

        assert!(result);
    }

    #[test]
    fn test_sha256_hasher() {
        type Sha256Tree = MerkleTree<2, Sha256Hasher>;

        let leaves = [[1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32]];

        let tree = Sha256Tree::from_leaves(&leaves);
        let proof = tree.prove(0);

        assert!(proof.verify(&[1u8; 32], tree.root()));
    }

    #[test]
    fn test_edge_cases() {
        type SmallTree = MerkleTree<1, TestHasher>; // 2 leaves

        let leaves = [[1u8; 32], [2u8; 32]];
        let tree = SmallTree::from_leaves(&leaves);

        assert_eq!(tree.leaf_count(), 2);
        assert_eq!(tree.height(), 1);

        // Proof for height 1 tree
        let proof = tree.prove(0);
        assert!(proof.verify(&[1u8; 32], tree.root()));
    }

    #[test]
    fn test_proof_display() {
        type TestTree = MerkleTree<2, TestHasher>;

        let leaves = create_test_leaves::<4>(1);
        let tree = TestTree::from_leaves(&leaves);
        let proof = tree.prove(1);

        let display_output = format!("{}", proof);
        assert!(display_output.contains("Merkle Proof"));
        assert!(display_output.contains("Leaf Index: 1"));
    }

    #[test]
    fn test_bidirectional_display() {
        type BiTree = BidirectionalMerkleTree<2, 4, TestHasher>;

        let bi_tree = BiTree::new();
        let display_output = format!("{}", bi_tree);

        assert!(display_output.contains("Bidirectional Merkle Tree"));
        assert!(display_output.contains("Main Tree:"));
        assert!(display_output.contains("Child Trees: 0/4"));
    }

    #[test]
    fn test_multiple_child_trees() {
        type BiTree = BidirectionalMerkleTree<2, 4, TestHasher>;

        let mut bi_tree = BiTree::new();

        // Link multiple child trees
        for i in 0..3 {
            let child_leaves = create_test_leaves::<4>((i * 10) as u8);
            let child_tree = MerkleTree::from_leaves(&child_leaves);
            bi_tree.link_child_tree(i, child_tree);
        }

        assert_eq!(bi_tree.child_tree_count(), 3);

        // Verify we can access each child root
        for i in 0..3 {
            assert!(bi_tree.get_child_root(i).is_some());
        }

        // Unlinked leaf should return None
        assert!(bi_tree.get_child_root(3).is_none());
    }

    // Insta snapshot tests
    #[test]
    fn test_merkle_tree_snapshot() {
        type TestTree = MerkleTree<2, TestHasher>;

        let leaves = [[1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32]];
        let tree = TestTree::from_leaves(&leaves);

        assert_snapshot!("merkle_tree_display", format!("{}", tree));
    }

    #[test]
    fn test_merkle_proof_snapshot() {
        type TestTree = MerkleTree<2, TestHasher>;

        let leaves = create_test_leaves::<4>(1);
        let tree = TestTree::from_leaves(&leaves);
        let proof = tree.prove(1);

        assert_snapshot!("merkle_proof_display", format!("{}", proof));
    }

    #[test]
    fn test_bidirectional_tree_snapshot() {
        type BiTree = BidirectionalMerkleTree<2, 4, TestHasher>;

        let mut bi_tree = BiTree::new();

        // Add some child trees
        for i in 0..2 {
            let child_leaves = create_test_leaves::<4>((i + 1) as u8 * 10);
            let child_tree = MerkleTree::from_leaves(&child_leaves);
            bi_tree.link_child_tree(i, child_tree);
        }

        assert_snapshot!("bidirectional_tree_display", format!("{}", bi_tree));
    }

    #[test]
    fn test_empty_bidirectional_tree_snapshot() {
        type BiTree = BidirectionalMerkleTree<2, 4, TestHasher>;

        let bi_tree = BiTree::new();
        assert_snapshot!("empty_bidirectional_tree", format!("{}", bi_tree));
    }

    #[test]
    fn test_small_tree_snapshot() {
        type SmallTree = MerkleTree<1, TestHasher>;

        let leaves = [[1u8; 32], [2u8; 32]];
        let tree = SmallTree::from_leaves(&leaves);

        assert_snapshot!("small_tree_display", format!("{}", tree));
    }

    #[test]
    fn test_single_node_tree_snapshot() {
        type SingleNodeTree = MerkleTree<0, TestHasher>;

        let leaves = [[1u8; 32]];
        let tree = SingleNodeTree::from_leaves(&leaves);

        assert_snapshot!("single_node_tree_display", format!("{}", tree));
    }

    #[test]
    fn test_sha256_tree_snapshot() {
        type Sha256Tree = MerkleTree<2, Sha256Hasher>;

        let leaves = [
            b"leaf0".as_ref(),
            b"leaf1".as_ref(),
            b"leaf2".as_ref(),
            b"leaf3".as_ref(),
        ];

        // Convert to [u8; 32] arrays
        let leaf_arrays: Vec<[u8; 32]> = leaves
            .iter()
            .map(|&data| {
                let mut array = [0u8; 32];
                let bytes = data;
                array[..bytes.len().min(32)].copy_from_slice(&bytes[..bytes.len().min(32)]);
                array
            })
            .collect();

        let tree = Sha256Tree::from_leaves(&leaf_arrays);
        assert_snapshot!("sha256_tree_display", format!("{}", tree));
    }
}
