// src/lib.rs
use core::fmt;
use std::fmt::{Display, Formatter};

pub trait Hasher {
    type Output: Copy + Clone + std::fmt::Debug + PartialEq + Default + AsRef<[u8]>;

    fn hash_leaf(data: &[u8]) -> Self::Output;
    fn hash_node(left: &Self::Output, right: &Self::Output) -> Self::Output;
}

#[derive(Debug, Clone)]
pub struct MerkleTree<const HEIGHT: usize, H: Hasher> {
    nodes: Vec<H::Output>,
    _hasher: std::marker::PhantomData<H>,
}

#[derive(Debug, Clone)]
pub struct MerkleProof<const HEIGHT: usize, H: Hasher> {
    path: [Option<H::Output>; HEIGHT],
    leaf_index: usize,
    _hasher: std::marker::PhantomData<H>,
}

impl<const HEIGHT: usize, H: Hasher> MerkleTree<HEIGHT, H> {
    const TOTAL_NODES: usize = (1 << (HEIGHT + 1)) - 1;
    const LEAF_COUNT: usize = 1 << HEIGHT;

    pub fn new() -> Self {
        Self {
            nodes: vec![H::Output::default(); Self::TOTAL_NODES],
            _hasher: std::marker::PhantomData,
        }
    }

    pub fn from_leaves(leaves: &[[u8; 32]]) -> Self {
        assert_eq!(leaves.len(), Self::LEAF_COUNT);

        let mut tree = Self::new();

        // Fill leaves
        let leaf_offset = (1 << HEIGHT) - 1;
        for (i, leaf_data) in leaves.iter().enumerate() {
            tree.nodes[leaf_offset + i] = H::hash_leaf(leaf_data);
        }

        // Build tree bottom-up
        for level in (0..HEIGHT).rev() {
            let level_start = (1 << level) - 1;
            let level_end = (1 << (level + 1)) - 1;

            for i in (level_start..level_end).step_by(2) {
                let left = &tree.nodes[i];
                let right = &tree.nodes[i + 1];
                tree.nodes[(i - 1) / 2] = H::hash_node(left, right);
            }
        }

        tree
    }

    pub fn root(&self) -> H::Output {
        self.nodes[0]
    }

    pub fn update_leaf(&mut self, index: usize, data: &[u8]) {
        let mut pos = (1 << HEIGHT) - 1 + index;
        self.nodes[pos] = H::hash_leaf(data);

        // Update path to root
        while pos > 0 {
            let parent = (pos - 1) / 2;
            let left_child = parent * 2 + 1;
            let right_child = parent * 2 + 2;

            self.nodes[parent] = H::hash_node(&self.nodes[left_child], &self.nodes[right_child]);
            pos = parent;
        }
    }

    pub fn get_leaf(&self, index: usize) -> H::Output {
        self.nodes[(1 << HEIGHT) - 1 + index]
    }

    pub fn prove(&self, leaf_index: usize) -> MerkleProof<HEIGHT, H> {
        let mut path = [None; HEIGHT];
        let mut pos = (1 << HEIGHT) - 1 + leaf_index;

        for level in 0..HEIGHT {
            let sibling = if pos % 2 == 1 { pos + 1 } else { pos - 1 };
            path[level] = Some(self.nodes[sibling]);
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

        for sibling_hash in self.path.iter().flatten() {
            if pos % 2 == 0 {
                hash = H::hash_node(sibling_hash, &hash);
            } else {
                hash = H::hash_node(&hash, sibling_hash);
            }
            pos /= 2;
        }

        hash == root
    }

    pub fn path_length(&self) -> usize {
        self.path.iter().filter(|x| x.is_some()).count()
    }
}

// For bidirectional trees where deepest leaf is another tree's root
#[derive(Debug, Clone)]
pub struct BidirectionalMerkleTree<const HEIGHT: usize, const MAX_CHILDREN: usize, H: Hasher> {
    main_tree: MerkleTree<HEIGHT, H>,
    child_trees: [Option<Box<MerkleTree<HEIGHT, H>>>; MAX_CHILDREN],
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
        self.main_tree
            .update_leaf(leaf_index, &self.hash_to_bytes(child_root));
        self.child_trees[leaf_index] = Some(Box::new(child_tree));
    }

    pub fn get_child_root(&self, leaf_index: usize) -> Option<H::Output> {
        Some(self.child_trees.get(leaf_index)?.as_ref()?.root())
    }

    pub fn verify_bidirectional_path(
        &self,
        main_leaf_index: usize,
        main_leaf_data: &[u8],
        child_leaf_index: usize,
        child_leaf_data: &[u8],
    ) -> bool {
        // Verify path in main tree
        let main_proof = self.main_tree.prove(main_leaf_index);
        if !main_proof.verify(main_leaf_data, self.main_tree.root()) {
            return false;
        }

        // Verify the child tree exists and path within it
        if let Some(child_tree) = &self.child_trees[main_leaf_index] {
            let child_proof = child_tree.prove(child_leaf_index);
            child_proof.verify(child_leaf_data, child_tree.root())
        } else {
            false
        }
    }

    fn hash_to_bytes(&self, hash: H::Output) -> Vec<u8>
    where
        H::Output: AsRef<[u8]>,
    {
        hash.as_ref().to_vec()
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

    #[test]
    fn test_merkle_tree_construction() {
        type TestTree = MerkleTree<3, TestHasher>; // 8 leaves

        let leaves = [[1u8; 32]; 8];
        let tree = TestTree::from_leaves(&leaves);

        assert_eq!(tree.leaf_count(), 8);
        assert_eq!(tree.height(), 3);
    }

    #[test]
    fn test_merkle_tree_update() {
        type TestTree = MerkleTree<2, TestHasher>; // 4 leaves

        let leaves = [[1u8; 32]; 4];
        let mut tree = TestTree::from_leaves(&leaves);
        let original_root = tree.root();

        // Update a leaf
        tree.update_leaf(1, &[2u8; 32]);
        let new_root = tree.root();

        assert_ne!(original_root, new_root);

        // Verify the updated leaf
        let updated_leaf = tree.get_leaf(1);
        let expected_hash = TestHasher::hash_leaf(&[2u8; 32]);
        assert_eq!(updated_leaf, expected_hash);
    }

    #[test]
    fn test_merkle_proof() {
        type TestTree = MerkleTree<2, TestHasher>; // 4 leaves

        let leaves = [[1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32]];
        let tree = TestTree::from_leaves(&leaves);

        // Generate proof for leaf 1
        let proof = tree.prove(1);
        assert_eq!(proof.path_length(), 2);

        // Verify proof
        assert!(proof.verify(&[2u8; 32], tree.root()));

        // Invalid data should fail
        assert!(!proof.verify(&[99u8; 32], tree.root()));
    }

    #[test]
    fn test_merkle_tree_display() {
        type TestTree = MerkleTree<2, TestHasher>;

        let leaves = [[1u8; 32]; 4];
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

        let main_leaves = [[1u8; 32]; 4];
        let child_leaves = [[5u8; 32]; 4];

        let main_tree = MainTree::from_leaves(&main_leaves);
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

        let main_leaves = [[1u8; 32]; 4];
        let child_leaves = [[5u8; 32]; 4];

        let main_tree = MainTree::from_leaves(&main_leaves);
        let child_tree = ChildTree::from_leaves(&child_leaves);

        let mut bi_tree = BiTree::new();
        bi_tree.link_child_tree(0, child_tree.clone());

        // Verify bidirectional path
        let result = bi_tree.verify_bidirectional_path(
            0,                                // main leaf index
            &main_tree.get_leaf(0).as_ref(),  // main leaf data
            1,                                // child leaf index
            &child_tree.get_leaf(1).as_ref(), // child leaf data
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
        assert_eq!(proof.path_length(), 1);
        assert!(proof.verify(&[1u8; 32], tree.root()));
    }

    #[test]
    fn test_proof_display() {
        type TestTree = MerkleTree<2, TestHasher>;

        let leaves = [[1u8; 32]; 4];
        let tree = TestTree::from_leaves(&leaves);
        let proof = tree.prove(1);

        let display_output = format!("{}", proof);
        assert!(display_output.contains("Merkle Proof"));
        assert!(display_output.contains("Leaf Index: 1"));
        assert!(display_output.contains("Path Length: 2"));
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
            let child_leaves = [[i; 32]; 4];
            let child_tree = MerkleTree::from_leaves(&child_leaves);
            bi_tree.link_child_tree(i as usize, child_tree);
        }

        assert_eq!(bi_tree.child_tree_count(), 3);

        // Verify we can access each child root
        for i in 0..3 {
            assert!(bi_tree.get_child_root(i).is_some());
        }

        // Unlinked leaf should return None
        assert!(bi_tree.get_child_root(3).is_none());
    }
}
