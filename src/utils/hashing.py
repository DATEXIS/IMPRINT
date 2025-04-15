"""
Hashing utilities for generating consistent IDs.

This module provides functions for generating consistent hash IDs from
parameter combinations, useful for caching experiment results or creating
unique identifiers for configurations.
"""

import hashlib
import json


def consistent_id(combination):
    """
    Generate a consistent integer ID from a dictionary of parameters.

    This function converts a dictionary to a JSON string and then hashes it
    using MD5 to create a deterministic integer ID.

    Args:
        combination: Dictionary of parameters to hash

    Returns:
        int: Consistent integer ID based on the input dictionary

    Example:
        >>> params = {"learning_rate": 0.01, "batch_size": 32}
        >>> consistent_id(params)
        123456789  # Some integer hash
    """
    # Create a single string from the dict
    combined = json.dumps(combination, sort_keys=True)

    # Generate a hash using md5 and convert to an integer
    return int(hashlib.md5(combined.encode()).hexdigest(), 16)
