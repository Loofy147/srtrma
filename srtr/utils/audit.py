import hashlib
import json
import logging

logger = logging.getLogger("SRTR.Audit")

class StateAuditor:
    """
    Cryptographic validator for state parity during hot-swaps.
    Ensures zero state-loss by hashing state snapshots.
    """
    @staticmethod
    def compute_state_hash(state):
        """
        Generates a SHA-256 hash of a serializable state dictionary.
        """
        try:
            # Ensure deterministic serialization
            state_str = json.dumps(state, sort_keys=True)
            return hashlib.sha256(state_str.encode('utf-8')).hexdigest()
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to hash state: {e}")
            return None

    @staticmethod
    def verify_parity(pre_hash, post_hash):
        """
        Verifies that two state hashes are identical.
        """
        if pre_hash is None or post_hash is None:
            return False
        return pre_hash == post_hash
