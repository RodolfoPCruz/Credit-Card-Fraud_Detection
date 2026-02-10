import hashlib

def compute_hash(file_path: str, chunk_size: int = 8192) -> str:
    """
    Computes SHA-256 hash of a file.

    Parameters
    ----------
    file_path : str
        Path to the file.
    chunk_size : int
        Number of bytes read per iteration.

    Returns
    -------
    str
        Hexadecimal SHA-256 hash.
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)

    return sha256.hexdigest()
