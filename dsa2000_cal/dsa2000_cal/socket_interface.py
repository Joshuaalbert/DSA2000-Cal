import socket
from typing import Tuple, Optional


class SocketInterface:
    """
    A singleton class for managing socket communication.

    This class provides a context manager for socket operations,
    ensuring proper management of resources. It supports both server
    and client socket operations including listening, accepting, sending,
    and receiving data.

    Attributes:
        host (str): The hostname or IP address to connect or bind the socket.
        port (int): The port number to connect or bind the socket.
        socket (Optional[socket.socket]): The socket instance for communication.

    Example:
        As a server:
            with SocketInterface(port=12345) as server:
                server.listen()
                client_socket, addr = server.accept()
                message = server.recv()
                server.send("Response")

        As a client:
            with SocketInterface(host='localhost', port=12345) as client:
                client.connect()
                client.send("Request")
                response = client.recv()
    """

    def __init__(self, host: str = 'localhost', port: int = 12345) -> None:
        """
        Initialize the SocketInterface with host and port.

        Args:
            host (str): The hostname or IP address.
            port (int): The port number.
        """
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None

    def __enter__(self) -> 'SocketInterface':
        """Context manager entry, creates a new socket."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error as e:
            print(f"Socket creation failed: {e}")
            raise
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[object]) -> None:
        """Context manager exit, closes the socket."""
        if self.socket:
            self.socket.close()

    def connect(self) -> None:
        """
        Connects the client socket to a remote address. The socket must be bound to an address and listening for
        connections. The format of address depends on the address family. The arguments passed to connect() depend
        on the address family and the socket type being used.
        """
        try:
            self.socket.connect((self.host, self.port))
        except socket.error as e:
            print(f"Connection failed: {e}")
            raise
        except OverflowError as e:
            print(f"Port number or address format error: {e}")
            raise

    def listen(self, backlog: int = 1) -> None:
        """
        Configures the server socket to listen for incoming connections. The socket must be bound to an address
        and listening for connections. The backlog argument specifies the maximum number of queued connections
        and should be at least 0; the maximum value is system-dependent (usually 5), the minimum value is forced
        to 0. If a connection request arrives when the queue is full, the client may receive an error with an
        indication of ECONNREFUSED or, if the underlying protocol supports retransmission, the request may be ignored
        so that a later reattempt at connection succeeds.

        Args:
            backlog (int): The maximum number of queued connections.
        """
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(backlog)
        except socket.error as e:
            print(f"Listening failed: {e}")
            raise
        except OverflowError as e:
            print(f"Port number or address format error: {e}")
            raise

    def accept(self) -> Tuple[socket.socket, Tuple[str, int]]:
        """
        Accepts a connection from a client. The socket must be bound to an address and listening for connections.
        The return value is a pair (conn, address) where conn is a new socket object usable to send and receive data
        on the connection, and address is the address bound to the socket on the other end of the connection.

        Returns:
            Tuple[socket.socket, Tuple[str, int]]: The client's socket and address.
        """
        try:
            return self.socket.accept()
        except socket.error as e:
            print(f"Accepting connection failed: {e}")
            raise
