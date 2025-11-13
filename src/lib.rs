// src/lib.rs
use core::marker::PhantomData;

pub trait Hasher {
    type Output: Copy + PartialEq + Default + AsRef<[u8]>;

    fn hash_leaf(data: &[u8]) -> Self::Output;
    fn hash_node(left: &Self::Output, right: &Self::Output) -> Self::Output;
}

pub struct MerkleTree<const HEIGHT: usize, H: Hasher> {
    nodes: Vec<H::Output>,
    _hasher: PhantomData<H>,
}

pub struct MerkleProof<const HEIGHT: usize, H: Hasher> {
    path: [Option<H::Output>; HEIGHT],
    leaf_index: usize,
    _hasher: PhantomData<H>,
}

impl<const HEIGHT: usize, H: Hasher> MerkleTree<HEIGHT, H> {
    const TOTAL_NODES: usize = (1 << (HEIGHT + 1)) - 1;
    const LEAF_COUNT: usize = 1 << HEIGHT;

    pub fn new() -> Self {
        Self {
            nodes: vec![H::Output::default(); Self::TOTAL_NODES],
            _hasher: PhantomData,
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
            _hasher: PhantomData,
        }
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
}
// For bidirectional trees where deepest leaf is another tree's root
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
}
