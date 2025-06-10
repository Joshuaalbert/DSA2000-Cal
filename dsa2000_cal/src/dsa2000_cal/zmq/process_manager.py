import multiprocessing
import signal
import sys
import time
from typing import List

import zmq

from dsa2000_cal.zmq.zmq_actor import ZMQActor
from dsa2000_common.common.logging import dsa_logger

STARTUP_TIMEOUT_S = 30


class ProcessManager:
    """
    Starts, supervises, and cleanly shuts down a list of ZMQActor processes.
    """

    def __init__(self, actors: List[ZMQActor], ctrl_addr: str, ack_rep_addr: str, shutdown_timeout: float = 1.0):
        self.actors = actors
        self.procs: List[multiprocessing.Process] = []
        self.shutdown_timeout = shutdown_timeout
        self.ack_rep_addr = ack_rep_addr

        self.ctx = zmq.Context()
        self.ctl = self.ctx.socket(zmq.PUB)
        self.ctl.bind(ctrl_addr)

        # Bind signals before starting any processes
        signal.signal(signal.SIGINT, self._cleanup)
        signal.signal(signal.SIGTERM, self._cleanup)

    def start_all(self):
        """Spawn a Process for each actor's run() method."""
        spawn_ctx = multiprocessing.get_context("forkserver")
        ack_rep = self.ctx.socket(zmq.REP)
        ack_rep.bind(self.ack_rep_addr)
        poller = zmq.Poller()
        poller.register(ack_rep, zmq.POLLIN)
        try:
            num_acks = 0
            for actor in self.actors:
                p = spawn_ctx.Process(target=actor.start, daemon=False)
                p.start()
                self.procs.append(p)
                socks = dict(poller.poll(STARTUP_TIMEOUT_S * 1000))
                if ack_rep in socks:
                    _ = ack_rep.recv()
                    ack_rep.send(b"")
                    num_acks += 1
                    continue
                dsa_logger.warning(f"Timeout waiting for actor startup acknowledgments. "
                                  f"Received {num_acks} out of {len(self.procs)} expected acks.")
            dsa_logger.info(f"Started {len(self.procs)} actor processes.")
        finally:
            poller.unregister(ack_rep)
            ack_rep.close(linger=0)
        if num_acks < len(self.procs):
            dsa_logger.error(f"Only {num_acks} out of {len(self.procs)} processes ack'd startup.")
            self.stop_all()
            sys.exit(1)
        else:
            dsa_logger.info(f"All {num_acks} actor processes acknowledged startup successfully.")

    def wait_all(self):
        """Block until all child processes exit."""
        for p in self.procs:
            p.join()

    def stop_all(self):
        """Terminate and join all live child processes."""
        # Publish shutdown signal to all actors
        dsa_logger.info("Gracefully terminating all actor processes...")
        self.ctl.send(b"TERMINATE")
        time.sleep(3)  # Give actors time to process the signal
        self.ctl.close(linger=0)
        self.ctx.term()

        # Our strategy for graceful shutdown is to send a control socket 'TERMINATE' message, which each actor should
        # handle appropriately. If they do not exit, within 3 seconds, we will attempt to forcefully terminate them by
        # calling terminate() on each process. This sends the SIGTERM signal to the process, which should be caught and
        # a forceful shutdown initiated, allowing the actor to clean up its resources. If they still do not exit, we
        # will call kill() on each process, which sends SIGKILL and does not allow the process to clean up. This is a
        # last resort to ensure all processes are terminated.

        force_term_count = 0
        for a, p in zip(self.actors, self.procs):
            if p.is_alive():
                dsa_logger.info(f"Attempting to forcefully terminate {p.pid} actor process ({a.__class__.__name__}).")
                p.terminate()  # sends SIGTERM
                force_term_count += 1
        if force_term_count > 0:
            dsa_logger.info(f"{force_term_count} of {len(self.procs)} actors failed to gracefully shutdown...")
        force_kill_count = 0
        for a, p in zip(self.actors, self.procs):
            p.join(timeout=self.shutdown_timeout)
            if p.is_alive():
                dsa_logger.warning(f"Process {p.pid} ({a.__class__.__name__}) did not exit cleanly, killing it.")
                p.kill()
                force_kill_count += 1
        if force_kill_count > 0:
            dsa_logger.info(
                f"{force_kill_count} of {len(self.procs)} actors failed to forceful termination... Killed them.")

    def _cleanup(self, signum, frame):
        """
        Signal handler: shut down children and exit.
        """
        dsa_logger.info(f"Process Manager received signal {signum}, shutting down...")
        self.stop_all()
        sys.exit(0)


def create_random_control_address() -> str:
    """
    Create a random control address for the process manager.
    This is a placeholder function; in practice, you would generate a unique address.
    """
    import random
    fd = random.randint(10000, 99999)  # Random file descriptor-like number
    return f"ipc:///tmp/ctl_pub_{fd}.ipc"


def create_random_ack_address() -> str:
    """
    Create a random control address for the process manager.
    This is a placeholder function; in practice, you would generate a unique address.
    """
    import random
    fd = random.randint(10000, 99999)  # Random file descriptor-like number
    return f"ipc:///tmp/ack_rep_{fd}.ipc"
