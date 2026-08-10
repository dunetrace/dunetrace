"""
API-key hashing. The secret must never be derivable from what is stored.

Run: PYTHONPATH=packages/schemas-py python -m pytest packages/schemas-py/tests/test_keys.py -v
"""

from __future__ import annotations

import unittest

from dunetrace_schemas.keys import (
    KEY_PREFIX_LENGTH,
    generate_api_key,
    hash_api_key,
    key_prefix,
    keys_equal,
)


class TestKeyGeneration(unittest.TestCase):
    def test_keys_are_prefixed_and_unique(self):
        a, b = generate_api_key(), generate_api_key()
        self.assertTrue(a.startswith("dt_"))
        self.assertNotEqual(a, b)

    def test_keys_carry_enough_entropy_to_resist_brute_force(self):
        # 32 random bytes url-safe encoded; the point of using plain SHA-256
        # rather than a slow KDF is that there is nothing to guess.
        self.assertGreaterEqual(len(generate_api_key()), 40)


class TestHashing(unittest.TestCase):
    def test_hash_is_stable(self):
        key = generate_api_key()
        self.assertEqual(hash_api_key(key), hash_api_key(key))

    def test_hash_does_not_contain_the_key(self):
        key = generate_api_key()
        digest = hash_api_key(key)
        self.assertNotIn(key, digest)
        self.assertNotIn(key[3:20], digest)

    def test_different_keys_hash_differently(self):
        self.assertNotEqual(hash_api_key(generate_api_key()), hash_api_key(generate_api_key()))

    def test_hash_is_hex_sha256(self):
        self.assertEqual(len(hash_api_key("x")), 64)


class TestPrefix(unittest.TestCase):
    def test_prefix_is_short_enough_to_be_useless_as_a_secret(self):
        key = generate_api_key()
        prefix = key_prefix(key)
        self.assertEqual(len(prefix), KEY_PREFIX_LENGTH)
        self.assertTrue(key.startswith(prefix))
        self.assertLess(len(prefix), len(key) / 2)


class TestComparison(unittest.TestCase):
    def test_equal_and_unequal(self):
        digest = hash_api_key("a")
        self.assertTrue(keys_equal(digest, digest))
        self.assertFalse(keys_equal(digest, hash_api_key("b")))


if __name__ == "__main__":
    unittest.main()
