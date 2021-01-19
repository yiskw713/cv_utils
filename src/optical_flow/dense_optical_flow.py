import dataclasses
import os
from typing import List

import cv2
import numpy as np


@dataclasses.dataclass
class DenseOpticalFlow(object):
    """Base class for dense optical flow."""

    # how often the process is done
    frame_interval: int = 1

    def _calc_optical_flow(
        self, prev_frame: np.ndarray, cur_frame: np.ndarray
    ) -> np.ndarray:
        """calculate optical flow
        Args:
            prev_frame: previous frame (H, W). Gray
            cur_frame: current frame (H, W). Gray
        Return:
            flow: optical flow (H, W, 2)
        """
        # overwrite this method in each child class
        # this is dummy calculation
        h, w = cur_frame.shape
        flow = cv2.absdiff(cur_frame, prev_frame)
        flow = flow.reshape(h, w, 1)
        flow = np.concatenate((flow, flow), axis=2).astype(np.float32)
        return flow

    def _colorize_flow(self, flow: np.ndarray) -> np.ndarray:
        """
        Args:
            flow: optical flow (H, W, 2)
        Return:
            rgb_flow: colorized optical flow (H, W, 3)
        """
        h, w, _ = flow.shape
        hsv_flow = np.zeros((h, w, 3), np.uint8)
        hsv_flow[..., 1] = 255

        # convert 2D array to magnitude and angular
        # mag -> (H, W)
        # ang -> (H, W)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv_flow[..., 0] = ang * 180 / np.pi / 2
        hsv_flow[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        rgb_flow = cv2.cvtColor(hsv_flow, cv2.COLOR_HSV2BGR)
        return rgb_flow

    def _save_optical_flow(
        self, flow: np.ndarray, timestamp: int, save_dir: str
    ) -> None:
        save_path = os.path.join(save_dir, f"flow_{timestamp:05}.npy")
        np.save(save_path, flow)

    def process(
        self,
        frames: List[np.ndarray],
        save_dir: str,
    ) -> None:
        # TODO:　動画のフレーム数と一致させるにはどうする？
        prev_frame = frames[0]
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        n_frames = len(frames)

        for timestamp in range(self.frame_interval, n_frames, self.frame_interval):
            cur_frame = frames[timestamp]
            cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

            flow = self._calc_optical_flow(prev_frame, cur_frame)

            self._save_optical_flow(flow, timestamp, save_dir)

            prev_frame = cur_frame.copy()

    def demo(self) -> None:
        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()
        prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        while cap.isOpened():
            ret, frame = cap.read()
            cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            flow = self._calc_optical_flow(prev_frame, cur_frame)

            rgb_flow = self._colorize_flow(flow)

            cv2.imshow("optical_flow", rgb_flow)

            k = cv2.waitKey(1)
            if k == ord("q"):
                break

            prev_frame = cur_frame.copy()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    processor = DenseOpticalFlow()
    processor.demo()
