import dataclasses
from typing import NamedTuple

import astropy.units as au
import jax
import jax.numpy as jnp

from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.interfaces.shared_data import SharedData
from dsa2000_cal.interfaces.socket_interface import SocketInterface


class CalibrationState(NamedTuple):
    prev_cadence_params: jax.Array


class CalibrationData(NamedTuple):
    state: CalibrationState  # Initial calibration parameters
    vis: jax.Array  # [rows, num_freqs, 4] the visibility data
    weights: jax.Array  # [rows, num_freqs, 4] the weights
    flags: jax.Array  # [rows, num_freqs, 4] the flags


@dataclasses.dataclass(eq=False)
class CalibrationProcess:
    num_iterations: int
    freqs: au.Quantity

    def __post_init__(self):
        if not self.freqs.unit.is_equivalent(au.Hz):
            raise ValueError("Frequency unit must be Hz")
        self.freqs_jax = quantity_to_jnp(self.freqs)

    def shapes_and_dtypes(self) -> CalibrationData:
        return CalibrationData(
            times=jax.ShapeDtypeStruct(shape=(1,), dtype=jnp.float32),

        )

    def run(self, port: int):
        print(f"Exposing calibration on port {port}")
        shared_data = SharedData(create=True, shape=(10,), dtype='i')
        print("Shared memory created with name:", shared_data.shm_name)
        try:
            with SocketInterface(port=port) as server:
                server.listen()
                print("Server listening...")

                client_socket, addr = server.accept()
                print(f"Connected to {addr}")

                client_socket.send(b"ARRAY_READY")
                print("Notified client that array is ready")

                client_socket.send(shared_data.shm_name.encode())  # Send the shm_name as a string

                while True:
                    message = client_socket.recv(1024)
                    print("Received from client:", message)

                    if message == b"SERVER_PROCESSING":
                        print(f'Array contents: {shared_data[:]}')

                        shared_data[:] = shared_data[:] + 1

                        print("New array contents:", shared_data[:])

                        client_socket.send(b"CLIENT_PROCESSING")
                        print("Notified client.")

                    if message == b"COMPLETED":
                        print("Client is done.")
                        break
        finally:
            shared_data.close()
            shared_data.unlink()
