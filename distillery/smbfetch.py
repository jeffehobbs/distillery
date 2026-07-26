"""Fetch tracks from an SMB share without mounting it.

`library.py` prefers a mounted share (`mount_smbfs` on macOS). On a Linux box
without passwordless sudo there is no way to mount one, so this reads files straight
off the share with `smbprotocol` — pure Python, no root, no kernel mount.

One connection per album, closed when the context manager exits.

The Unicode retry matters: a collection indexed on macOS records decomposed (NFD)
paths, while the share generally wants composed (NFC). An accented filename will
otherwise 404 with STATUS_OBJECT_PATH_NOT_FOUND even though it is plainly there.
"""
import unicodedata
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import config

_READ_CHUNK = 1 << 20      # 1 MiB per SMB2 read


def available():
    try:
        import smbprotocol  # noqa: F401
        return True
    except ImportError:
        return False


def _credentials():
    s = config.secrets()
    user, pw = s.get("SMB_USER"), s.get("SMB_PASSWORD")
    if not user or not pw:
        raise RuntimeError(
            f"no SMB credentials — put SMB_USER / SMB_PASSWORD in "
            f"{config.SECRETS_FILE} (chmod 600), or point "
            f"DISTILLERY_SMB_CREDENTIALS at a username=/password= file.")
    host = s.get("SMB_HOST", config.SMB_HOST)
    share = s.get("SMB_SHARE", config.SMB_SHARE)
    return user, pw, host, share


@contextmanager
def session():
    """Yield a connected tree for the configured share."""
    from smbprotocol.connection import Connection
    from smbprotocol.session import Session
    from smbprotocol.tree import TreeConnect

    user, pw, host, share = _credentials()
    conn = Connection(uuid.uuid4(), host, 445)
    conn.connect(timeout=20)
    try:
        sess = Session(conn, user, pw)
        sess.connect()
        tree = TreeConnect(sess, rf"\\{host}\{share}")
        tree.connect()
        yield tree
    finally:
        conn.disconnect()


def fetch(tree, rel_path, dest_path):
    """Download one file, retrying the other Unicode normalizations."""
    from smbprotocol.exceptions import ObjectNameNotFound, ObjectPathNotFound

    last = None
    tried = set()
    for form in (None, "NFC", "NFD"):
        p = rel_path if form is None else unicodedata.normalize(form, rel_path)
        if p in tried:
            continue
        tried.add(p)
        try:
            return _fetch_one(tree, p, Path(dest_path))
        except (ObjectNameNotFound, ObjectPathNotFound) as e:
            last = e
    raise last


def _fetch_one(tree, rel_path, dest_path):
    from smbprotocol.open import (
        CreateDisposition, CreateOptions, FileAttributes,
        FilePipePrinterAccessMask, ImpersonationLevel, Open, ShareAccess,
    )
    fh = Open(tree, rel_path.replace("/", "\\"))
    fh.create(
        ImpersonationLevel.Impersonation,
        FilePipePrinterAccessMask.GENERIC_READ,
        FileAttributes.FILE_ATTRIBUTE_NORMAL,
        ShareAccess.FILE_SHARE_READ,
        CreateDisposition.FILE_OPEN,
        CreateOptions.FILE_NON_DIRECTORY_FILE,
    )
    try:
        size = fh.end_of_file
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest_path.with_suffix(dest_path.suffix + ".part")
        with open(tmp, "wb") as out:
            offset = 0
            while offset < size:
                chunk = fh.read(offset, min(_READ_CHUNK, size - offset))
                if not chunk:
                    break
                out.write(chunk)
                offset += len(chunk)
        tmp.replace(dest_path)      # atomic: a partial download is never mistaken
                                    # for a complete cached file
        return dest_path
    finally:
        fh.close()


def list_dir(tree, rel_path=""):
    """Directory entry names under `rel_path` on the share."""
    from smbprotocol.open import (
        CreateDisposition, CreateOptions, DirectoryAccessMask, FileAttributes,
        FileInformationClass, ImpersonationLevel, Open, ShareAccess,
    )
    fh = Open(tree, rel_path.replace("/", "\\") or "")
    fh.create(
        ImpersonationLevel.Impersonation,
        DirectoryAccessMask.FILE_LIST_DIRECTORY,
        FileAttributes.FILE_ATTRIBUTE_DIRECTORY,
        ShareAccess.FILE_SHARE_READ,
        CreateDisposition.FILE_OPEN,
        CreateOptions.FILE_DIRECTORY_FILE,
    )
    try:
        out = []
        for entry in fh.query_directory("*", FileInformationClass.FILE_NAMES_INFORMATION):
            name = entry["file_name"].get_value().decode("utf-16-le")
            if name not in (".", ".."):
                out.append(name)
        return out
    finally:
        fh.close()
