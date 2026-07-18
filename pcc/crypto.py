"""Cryptographic boundary for the prototype.

All plaintext enters and leaves through AES-SIV. The carrier layer only handles
the resulting opaque byte string and never interprets plaintext.
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.ciphers.aead import AESSIV
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .errors import AuthenticationError, InvalidArgument

TAG_SIZE = 16
MAX_PLAINTEXT = 1 << 20


@dataclass(frozen=True)
class DerivedKeys:
    encryption: bytes
    mapping: bytes
    stream: bytes
    authentication: bytes
    nonce: bytes
    shuffle: bytes


def derive_keys(passphrase: str | bytes, salt: bytes) -> DerivedKeys:
    """Derive independent encryption and carrier-mapping keys.

    Scrypt is used here because it is available in the installed cryptography
    package without adding another runtime dependency. The pack salt is public.
    """

    if isinstance(passphrase, str):
        passphrase = passphrase.encode("utf-8")
    if len(passphrase) < 12:
        raise InvalidArgument("passphrase must contain at least 12 bytes")
    if len(salt) < 16:
        raise InvalidArgument("pack salt must contain at least 16 bytes")
    master = Scrypt(salt=salt, length=32, n=2**15, r=8, p=1).derive(passphrase)

    def subkey(label: bytes) -> bytes:
        return HKDF(
            algorithm=hashes.SHA256(),
            length=64 if label == b"message-encryption" else 32,
            salt=None,
            info=b"pcc/v1/" + label,
        ).derive(master)

    return DerivedKeys(
        encryption=subkey(b"message-encryption"),
        mapping=subkey(b"carrier-mapping"),
        stream=subkey(b"stream-encryption"),
        authentication=subkey(b"message-authentication"),
        nonce=subkey(b"message-nonce"),
        shuffle=subkey(b"message-shuffle"),
    )


def pack_id_bytes(pack_id: str) -> bytes:
    value = bytes.fromhex(pack_id)
    if len(value) != 32:
        raise ValueError("pack id must be a SHA-256 digest")
    return value


def associated_data(pack_id: str, sequence: int) -> bytes:
    if sequence < 0 or sequence >= 2**64:
        raise ValueError("sequence is outside the uint64 range")
    return b"pcc/v1/message\x00" + pack_id_bytes(pack_id) + struct.pack(">Q", sequence)


def encrypt(keys: DerivedKeys, pack_id: str, sequence: int, plaintext: bytes) -> bytes:
    if len(plaintext) > MAX_PLAINTEXT:
        raise ValueError("plaintext is too large")
    return AESSIV(keys.encryption).encrypt(plaintext, [associated_data(pack_id, sequence)])


def decrypt(keys: DerivedKeys, pack_id: str, sequence: int, sealed: bytes) -> bytes:
    if len(sealed) < TAG_SIZE:
        raise AuthenticationError("authenticated message failed")
    try:
        return AESSIV(keys.encryption).decrypt(sealed, [associated_data(pack_id, sequence)])
    except InvalidTag as exc:
        raise AuthenticationError("authenticated message failed") from exc


def keyed_digest(key: bytes, label: bytes) -> bytes:
    return hmac.new(key, label, hashlib.sha256).digest()


def v2_context(carrier_id: str, profile: str, sequence: int) -> bytes:
    if sequence < 0 or sequence >= 2**64:
        raise ValueError("sequence is outside the uint64 range")
    return (
        b"pcc/v2/message\x00"
        + pack_id_bytes(carrier_id)
        + profile.encode("ascii")
        + b"\x00"
        + struct.pack(">Q", sequence)
    )


def stream_crypt(keys: DerivedKeys, carrier_id: str, profile: str, sequence: int, data: bytes) -> bytes:
    context = v2_context(carrier_id, profile, sequence)
    nonce = keyed_digest(keys.nonce, b"pcc/v2/chacha20-nonce\x00" + context)[:16]
    cryptor = Cipher(algorithms.ChaCha20(keys.stream, nonce), mode=None).encryptor()
    return cryptor.update(data) + cryptor.finalize()
