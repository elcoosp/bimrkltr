// src/lib.rs
use core::marker::PhantomData;

pub trait Hasher {
    type Output: Copy + PartialEq + Default;

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
