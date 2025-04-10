import signal
from contextlib import contextmanager


@contextmanager
def ignore_interrupt():
    """
    Context manager to ignore SIGINT (Ctrl+C) signals.

    Returns:
        a context manager that ignores SIGINT signals.
    """
    # Save the original SIGINT handler.
    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        # Restore the original SIGINT handler.
        signal.signal(signal.SIGINT, original_handler)
