# server.py

from dsa2000_cal.interfaces.shared_data import SharedData
from dsa2000_cal.interfaces.socket_interface import SocketInterface


def run_server(port: int):
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


if __name__ == "__main__":
    port = 12345
    run_server(port)
