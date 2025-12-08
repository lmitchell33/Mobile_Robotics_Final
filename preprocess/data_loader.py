"""
Because data from different sources can come in different formats (most look like they are in bags though)
I figured I would make this file to load data from the different formats and put it into a common format we are
expecting later on in the pipeline.
"""
from pathlib import Path
import os
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

CURRENT_DIR = Path(__file__).parent
test_ros_bag = os.path.expanduser("~/Downloads/MH_01_easy.bag")
typestore = get_typestore(Stores.ROS1_NOETIC) # not sure which version was used but this is working

class DataLoader:
    """
    for now this only implements anything with ros bags, but if needed it shouldnt be too hard to change. I also think that bc some 
    of these datasets are use I am going to have to yield the output data so the gc can work its magic (so we dont run out of memory)
    """
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist")

        if not file_path.endswith(".bag"):
            raise ValueError(f"I only have rosbags working so no thank you")

        self._file_path = file_path

    def get_imu_data(self):
        yield from self.get_topic("/imu0")

    def get_image_data(self):
        yield from self.get_topic("/cam0/image_raw")

    def get_topic(self, topic_name: str):
        with Reader(self._file_path) as bag:
            for connection, timestamp, rawdata in bag.messages():
                if connection.topic == topic_name:
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                    yield timestamp, msg

    def display_topics(self):
        with Reader(self._file_path) as bag:
            for conneciton in bag.connections:
                print(f"Topic: {conneciton.topic} -> {conneciton.msgtype}")

if __name__ == "__main__":
    data_loader = DataLoader(test_ros_bag)
    # print(data_loader.display_topics())
    # for data in data_loader.get_topic("/imu0"):
        # timestamp, imu = data
        # print(timestamp, data)
    for data in data_loader.get_imu_data():
        timestamp, imu = data
        print(timestamp, data)