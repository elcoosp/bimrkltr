//! Error type for the sparse [`JellyfishMerkleTree`](crate::JellyfishMerkleTree).

use alloc::vec::Vec;
use core::fmt;

/// Errors returned by sparse tree operations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum JMTError {
    /// The requested key is not present in the tree.
    KeyNotFound(Vec<u8>),
    /// The requested version does not exist (or its history was pruned).
    InvalidVersion(u64),
    /// A different key is already committed to the same tree position; the
    /// payload carries the pre-existing key.
    KeyCollision(Vec<u8>),
}

impl fmt::Display for JMTError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeyNotFound(key) => write!(f, "key not found: {}", hex::encode(key)),
            Self::InvalidVersion(version) => write!(f, "invalid version: {version}"),
            Self::KeyCollision(key) => write!(
                f,
                "key commitment collides with existing key: {}",
                hex::encode(key)
            ),
        }
    }
}

impl core::error::Error for JMTError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_human_readable() {
        assert_eq!(
            JMTError::KeyNotFound(vec![1, 2, 3]).to_string(),
            "key not found: 010203"
        );
        assert_eq!(
            JMTError::InvalidVersion(7).to_string(),
            "invalid version: 7"
        );
        assert_eq!(
            JMTError::KeyCollision(vec![9]).to_string(),
            "key commitment collides with existing key: 09"
        );
    }

    #[test]
    fn implements_std_error() {
        fn assert_error<E: core::error::Error>(_: &E) {}
        assert_error(&JMTError::InvalidVersion(1));
    }
}
