import datetime
import glob
import hashlib
import os
import subprocess
import sys
import uuid
import warnings


def deterministic_uuid(seed: str) -> uuid.UUID:
    """
    Generate a UUID using a deterministic hashing of a seed string.

    Args:
        seed: str, a string seed

    Returns:
        UUID

    """
    if not isinstance(seed, str):
        raise TypeError(f"Expected seed type `str`, got {type(seed)}.")
    m = hashlib.md5()
    m.update(seed.encode('utf-8'))
    new_uuid = uuid.UUID(m.hexdigest())
    return new_uuid


class BaseContent:
    """
    Represents an entity with UUID and content path.
    """

    def __init__(self, *, seed: str):
        sync_content()
        self.content_path = self._content_path()
        self.seed = seed
        self.id = str(deterministic_uuid(seed))

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other: 'BaseContent') -> bool:
        """
        Enables equality comparison of content by id.

        Args:
            other: a base content object

        Returns:
            True if the ids are equal, False otherwise.
        """
        if other is None:
            return False
        if not isinstance(other, BaseContent):
            raise TypeError(f"Expected 'BaseContent' inheritor, got {type(other)}")
        return self.id == other.id

    @classmethod
    def _content_path(cls):
        return os.path.split(os.path.abspath(sys.modules[cls.__module__].__file__))[:-1]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.id[:6]})"

    def __lt__(self, other: 'BaseContent') -> bool:
        """
        Enables sorting of content by id.

        Args:
            other: a base content object

        Returns:
            True if the id of the current object is less than the id of the other object, False otherwise.
        """
        if not isinstance(other, BaseContent):
            raise TypeError(f"Expected 'BaseContent' inheritor, got {type(other)}")
        return self.id < other.id


_BASE_CONTENT_SYNC = False


def sync_content():
    if os.environ.get("SKIP_CONTENT_SYNC", "0") == "1":
        return

    global _BASE_CONTENT_SYNC
    if not _BASE_CONTENT_SYNC:
        _BASE_CONTENT_SYNC = True

        content_prefix_path = os.path.split(os.path.abspath(__file__))[:-1]

        cert_file = os.path.join(*content_prefix_path, '.sync_cert')
        print(f"Searching for sync certificate: {cert_file}")
        if os.path.exists(cert_file):
            with open(cert_file, 'r') as f:
                # get the last line
                last_line = f.readlines()[-1]
                # Parse from isot
                last_sync = datetime.datetime.fromisoformat(last_line.strip())
                if datetime.datetime.now() - last_sync < datetime.timedelta(days=1):
                    return
        print("Syncing assets. This happens at most once per day, and may take a few minutes.")

        def remove_content_prefix(content_path):
            prefix_path = os.path.join(*content_prefix_path)
            # remove
            if not content_path.startswith(prefix_path):
                raise RuntimeError("Prefix content path not found in the content path")
            content_path = content_path[len(prefix_path):]
            return content_path

        lf_paths = glob.glob(os.path.join(*content_prefix_path, "**", ".large_files"), recursive=True)
        FMCAL_FTP_ADDRESS = os.environ.get("FMCAL_FTP_ADDRESS", "mario:/safepool/fmcal_data")

        for lf_path in lf_paths:
            # lf_path = os.path.join(*self.content_path, ".large_files")
            if os.path.exists(lf_path):
                with open(lf_path, 'r') as f:
                    for line in f:
                        # line examples:
                        # Tau-sources.txt
                        # fits_models/*.fits
                        destination_path = os.path.dirname(lf_path)
                        asset_root_path = remove_content_prefix(destination_path)
                        ftp_url = f"{FMCAL_FTP_ADDRESS}{asset_root_path}/{line.strip()}"
                        cmd = ['rsync', '-a', '--partial', ftp_url,
                               f"{destination_path}/"]
                        subprocess.run(['echo'] + cmd)
                        completed_process = subprocess.run(cmd)
                        if completed_process.returncode != 0:
                            warnings.warn(f"rsync failed with return code {completed_process.returncode}")

        with open(cert_file, 'a') as f:
            f.write(f"{datetime.datetime.now().isoformat()}\n")
