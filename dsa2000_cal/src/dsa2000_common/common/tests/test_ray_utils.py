import os
import time

from dsa2000_common.common.ray_utils import MemoryLogger


def test_memory_logger(tmp_path):
    log_file = str(tmp_path / "memory_usage.log")
    with MemoryLogger(log_file=log_file) as mem_logger:
        time.sleep(5)
        # Memory usage should be logged every second for 5 seconds

    # Check that the log file was created
    assert os.path.exists(log_file)

    # Read the log file
    with open(log_file, 'r') as f:
        lines = f.readlines()

    print(lines)

    # Check that the log file contains the expected number of lines
    assert len(lines) == 6
