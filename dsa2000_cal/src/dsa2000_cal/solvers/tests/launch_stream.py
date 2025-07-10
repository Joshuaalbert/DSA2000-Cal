import numpy as np
import zmq

from dsa2000_cal.streaming_calibrator import StreamData
from dsa2000_cal.zmq.zmq_actor import ZMQActor

# class StreamData(NamedTuple):
#     vis_obs: FloatArray  # [T, B, C, 2, 2] visibility observations
#     weights: FloatArray  # [T, B, C, 2, 2] weights for the observations
#     vis_model: FloatArray  # [D, T, B, C, 2, 2] model visibilities

class MockDataStream(ZMQActor):
    def __init__(self, ctl_pub_addr: str, ack_rep_addr: str, rep_addr: str):
        super().__init__(ctl_pub_addr, ack_rep_addr)
        self.rep_addr = rep_addr

    def run(self):
        req = self.new_socket(zmq.REQ, connect=self.rep_addr)

        D = 2
        T = 2
        A = 10
        B = A * (A-1) // 2
        C = 2


        while True:
            # Produce fake data
            vis_obs = np.random.normal(size=(T, B, C, 2, 2)).astype(np.float32) + 1j * np.random.normal(size=(T, B, C, 2, 2)).astype(np.float32)
            weights = np.random.uniform(0, 1, size=(T, B, C, 2, 2)).astype(np.float32)
            vis_model = np.random.normal(size=(D, T, B, C, 2, 2)).astype(np.float32) + 1j * np.random.normal(size=(D, T, B, C, 2, 2)).astype(np.float32)
            data = StreamData(
                vis_obs=vis_obs,  # [T, B, C, 2, 2]
                weights=weights,  # [T, B, C, 2, 2]
                vis_model=vis_model  # [D, T, B, C, 2, 2]
            )


def main():
    ...