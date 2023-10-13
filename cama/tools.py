import ffmpeg
import numpy as np
import json


def load_json(filename):
    with open(filename, "r") as f:
        contents = json.load(f)
    return contents


class VideoGenerator:
    def __init__(self, output_video_path, output_shape=(2880, 1080)):
        self.writer = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{output_shape[0]}x{output_shape[1]}')
            .output(output_video_path, pix_fmt='yuv420p', vcodec='libx264', r=10, loglevel='quiet')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def concate_image(self, image_dict):
        top_row = np.concatenate([image_dict["camera_front_left"], image_dict["camera_front"], image_dict["camera_front_right"]], axis=1)
        bottom_row = np.concatenate([image_dict["camera_rear_left"], image_dict["camera_rear"], image_dict["camera_rear_right"]], axis=1)
        return np.concatenate([top_row, bottom_row], axis=0)

    def add_frame(self, image):
        self.writer.stdin.write(
            image
            .astype(np.uint8)
            .tobytes()
        )

    def add_frame_from_dict(self, image_dict):
        image = self.concate_image(image_dict)
        self.add_frame(image)

    def __del__(self):
        self.writer.stdin.close()
        self.writer.wait()
