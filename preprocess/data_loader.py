"""
Because data from different sources can come in different formats (most look like they are in bags though)
I figured I would make this file to load data from the different formats and put it into a common format we are
expecting later on in the pipeline.
"""
from pathlib import Path
import os
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
import numpy as np

CURRENT_DIR = Path(__file__).parent
test_ros_bag = os.path.expanduser("~/Downloads/MH_01_easy.bag")
typestore = get_typestore(Stores.ROS1_NOETIC) # not sure which version was used but this is working

class DataLoader:
    """
    for now this only implements anything with ros bags, but if needed it shouldnt be too hard to change
    """
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist")

        if not file_path.endswith(".bag"):
            raise ValueError(f"I only have rosbags working so no thank you")

        self.bag_path = file_path
        self.cam0 = []
        self.cam1 = []
        self.imu = []

        self._read_all()
        self.t0 = min(
            self.cam0[0][0],
            self.cam1[0][0],
            self.imu[0][0]
        )


    def _read_all(self):
        with Reader(self.bag_path) as bag:
            i=0
            for conn, t, raw in bag.messages():
                if i == 800:
                    break

                if conn.topic == "/cam0/image_raw":
                    msg = typestore.deserialize_ros1(raw, conn.msgtype)
                    self.cam0.append((t, msg))

                elif conn.topic == "/cam1/image_raw":
                    msg = typestore.deserialize_ros1(raw, conn.msgtype)
                    self.cam1.append((t, msg))

                elif conn.topic == "/imu0":
                    msg = typestore.deserialize_ros1(raw, conn.msgtype)
                    self.imu.append((t, msg))
                i += 1

    def get_stereo(self):
        # data looks to be perfectly synchronized so no difference between t0 and t1
        for (t0, m0), (t1, m1) in zip(self.cam0, self.cam1):
            t = (t0 - self.t0) * 1e-9
            # convert from ros to numpy
            left = np.frombuffer(m0.data, dtype=np.uint8).reshape(m0.height, m0.width)
            right = np.frombuffer(m1.data, dtype=np.uint8).reshape(m1.height, m1.width)
            yield t, left, right

    def get_imu(self):
        for t0, msg in self.imu:
            t = (t0 - self.t0) * 1e-9
            yield t, msg

    def display_topics(self):
        with Reader(self._file_path) as bag:
            for conneciton in bag.connections:
                print(f"Topic: {conneciton.topic} -> {conneciton.msgtype}")

if __name__ == "__main__":
    bag = os.path.expanduser("~/Downloads/cam_checkerboard.bag")
    data_loader = DataLoader(bag)
    print(data_loader.display_topics())
    # for data in data_loader.get_topic("/imu0"):
        # timestamp, imu = data
        # print(timestamp, data)
    # for data in data_loader.get_imu_data():
    #     timestamp, imu = data
    #     print(timestamp, data)