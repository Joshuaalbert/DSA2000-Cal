import contextlib
import pickle
import uuid
from abc import ABCMeta, abstractmethod, ABC
from collections import OrderedDict
from typing import TypeVar, Generic

import zmq
import zmq.asyncio

from dsa2000_cal.zmq.zmq_actor import ZMQActor


class MethodNotFoundError(Exception):
    """Raised when an RPC method name is not found on the target object."""
    pass


# -----------------------------------------------------------------------------
# Server‐side: handle_request_loop with idempotency + pickle-error guarding
# -----------------------------------------------------------------------------


K = TypeVar('K')
V = TypeVar('V')


class LRUDict(Generic[K, V]):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict[K, V] = OrderedDict()

    def clear(self):
        self.cache.clear()

    def __getitem__(self, item: K) -> V:
        return self.get(item)

    def __setitem__(self, key: K, value: V):
        return self.put(key, value)

    def __contains__(self, item: K):
        return item in self.cache

    def __len__(self):
        return len(self.cache)

    def __iter__(self):
        return iter(self.cache)

    def keys(self):
        return self.cache.keys()

    def items(self):
        return self.cache.items()

    def values(self):
        return self.cache.values()

    def get(self, key: K) -> V:
        # Move the key to the end to mark it as recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: K, value: V):
        if key in self.cache:
            # Update the value and move to the end
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # Remove the oldest item
                self.cache.popitem(last=False)
            self.cache[key] = value


class RPCActor(ZMQActor, ABC):
    """
    Provides a three-frame‐in, two-frame‐out request/reply loop on a REP socket:
      [method, pickle(args), pickle(kwargs)] → [b"ok", pickle(result)] or [b"error", pickle(exception)]
    """

    def __init__(self, ctl_pub_addr: str, ack_rep_addr: str, rep_addr: str):
        ZMQActor.__init__(self, ctl_pub_addr=ctl_pub_addr, ack_rep_addr=ack_rep_addr)
        self.rep_addr = rep_addr

    @contextlib.contextmanager
    @abstractmethod
    def yield_service(self):
        """
        Context manager that yields a service object to handle RPC requests, e.g. a database connection.
        """
        pass

    def run(self):
        rep = self.new_socket(zmq.REP, bind=self.rep_addr)
        ctl = self.new_socket(zmq.SUB, connect=self.ctl_pub_addr)
        ctl.setsockopt_string(zmq.SUBSCRIBE, "")
        self.ack_startup()
        with self.yield_service() as db:
            self.handle_request_loop(ctl, rep, db)

    @staticmethod
    def handle_request_loop(ctl: zmq.Socket, rep: zmq.Socket, target, cache_size: int = 1024):
        cache = LRUDict(capacity=cache_size)  # req_id -> (status: bytes, payload: bytes)
        poller = zmq.Poller()
        poller.register(ctl, zmq.POLLIN)
        poller.register(rep, zmq.POLLIN)
        try:
            while True:
                socks = dict(poller.poll())
                if ctl in socks:
                    msg = ctl.recv()
                    if msg == b"TERMINATE":
                        break
                if rep in socks:
                    try:
                        frames = rep.recv_multipart()
                    except (zmq.error.ContextTerminated, zmq.error.ZMQError):
                        break

                    # expect [ method, req_id, args_pickle, kwargs_pickle ]
                    if len(frames) != 4:
                        rep.send_multipart(
                            [b"", b"error", pickle.dumps(ValueError(f"Expected 4 frames, got {len(frames)}"))])
                        continue

                    method_name = frames[0].decode("utf-8")
                    req_id = frames[1]
                    args_blob, kwargs_blob = frames[2], frames[3]

                    if method_name == "ping":
                        rep.send_multipart([req_id, b'ok', pickle.dumps(b'')])
                        continue

                    # If we’ve already processed this req_id, re‐send the cached reply
                    if req_id in cache:
                        status, payload = cache[req_id]
                        rep.send_multipart([req_id, status, payload])
                        continue

                    # decode args/kwargs
                    try:
                        args = pickle.loads(args_blob)
                        kwargs = pickle.loads(kwargs_blob)
                    except Exception as e:
                        err = RuntimeError(f"Unpickle error: {str(e)}")
                        payload = pickle.dumps(err)
                        cache[req_id] = (b"error", payload)
                        rep.send_multipart([req_id, b"error", payload])
                        continue

                    # look up method
                    if not hasattr(target, method_name):
                        mnf = MethodNotFoundError(method_name)
                        payload = pickle.dumps(mnf)
                        cache[req_id] = (b"error", payload)
                        rep.send_multipart([req_id, b"error", payload])
                        continue

                    # invoke and send
                    try:
                        result = getattr(target, method_name)(*args, **kwargs)
                        try:
                            payload = pickle.dumps(result)
                            status = b"ok"
                        except pickle.PicklingError as e:
                            # pickling the result failed
                            payload = pickle.dumps(e)
                            status = b"error"

                    except Exception as e:
                        # user method raised
                        try:
                            payload = pickle.dumps(e)
                            status = b"error"
                        except pickle.PicklingError as e:
                            # pickling the exception failed
                            payload = pickle.dumps(RuntimeError(f"Exception unpicklable: {str(e)}"))
                            status = b"error"

                    # cache & send
                    cache[req_id] = (status, payload)
                    rep.send_multipart([req_id, status, payload])
        finally:
            poller.unregister(ctl)
            poller.unregister(rep)


class ZMQRPCClient:
    """
    A thin REQ‐side wrapper that lets you call any RPC method by attr access.
    Sends 3 frames and expects 2‐frame replies exactly as RPCMixin.
    """

    def __init__(self, req_socket: zmq.Socket, timeout_ms: int | None = 5000):
        self.socket = req_socket
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.timeout_ms = timeout_ms

    def close(self):
        """
        Close the underlying socket.
        """
        self.poller.unregister(self.socket)

    def _call(self, method_name: str, *args, **kwargs):
        req_id = uuid.uuid4().bytes  # 16-byte unique ID

        # prepare frames: [method, req_id, args, kwargs]
        try:
            args_blob = pickle.dumps(args)
            kwargs_blob = pickle.dumps(kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to pickle arguments: {str(e)}")

        frames = [
            method_name.encode("utf-8"),
            req_id,
            args_blob,
            kwargs_blob,
        ]
        self.socket.send_multipart(frames)

        # wait for reply or timeout
        socks = dict(self.poller.poll(self.timeout_ms))
        if socks.get(self.socket) != zmq.POLLIN:
            raise TimeoutError(f"RPC '{method_name}' timed out after {self.timeout_ms}ms")

        resp = self.socket.recv_multipart()
        # expect [req_id, status, payload]
        if len(resp) != 3 or resp[0] != req_id:
            raise RuntimeError("Malformed or unexpected RPC response")

        status = resp[1].decode("utf-8")
        payload = resp[2]
        data = pickle.loads(payload)

        if status == "ok":
            return data
        elif status == "error":
            # payload is an exception instance
            raise data
        else:
            raise RuntimeError(f"Unknown status '{status}' in RPC response")

    def __getattr__(self, method_name: str):
        # Only invoke methods that are not already defined in ABC. In normal use this would lead to method not found error,
        # but there could be a legit reason to RPC a method that is not defined in the ABC.
        def proxy(*args, **kwargs):
            return self._call(method_name, *args, **kwargs)

        return proxy

    def heartbeat(self):
        """
        Shortcut to call a lightweight 'ping' RPC on the server.
        Raises TimeoutError if server is dead/unresponsive.
        """
        return self._call("ping")


class RPCClientMeta(ABCMeta):

    def __new__(mcls, name, bases, namespace):
        # 1) clone the namespace dict so we can mutate it
        ns = dict(namespace)

        # 2) collect all abstract method names from any ABC base
        abstracts = set()
        for base in bases:
            abstracts |= getattr(base, "__abstractmethods__", set())

        # 3) for each missing abstract, inject a concrete proxy into ns
        for meth in abstracts:
            if meth in ns:
                raise RuntimeError(f"Cannot override abstract method {meth} in {name} as it is already defined.")

            proxy = mcls.make_proxy(meth)

            proxy.__name__ = meth
            proxy.__doc__ = f"auto-RPC stub for {meth}"
            # explicitly declare it non-abstract
            proxy.__isabstractmethod__ = False

            ns[meth] = proxy

        # 4) now let ABCMeta build the class—with our proxies already in place
        return super().__new__(mcls, name, bases, ns)

    @classmethod
    def make_proxy(mcls, m):
        def proxy(self, *args, **kwargs):
            return self._call(m, *args, **kwargs)

        return proxy
