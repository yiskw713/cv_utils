import dataclasses
import os
from typing import List

import cv2
import numpy as np


@dataclasses.dataclass
class GunnarFarnebackOpticalFlow(object):
    """Calculate Gunnar Farneback optical flow
    reference:
    https://docs.opencv.org/3.4/dc/d6b/group__video__track.html
    """

    # parameters for gunnar farneback optical flow

    # param for image pyramid scale
    # see http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html
    pyr_scale: float = 0.5

    levels: int = 3
    window_size: int = 15
    n_iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0

    # how often the process is done
    interval: int = 1

    def _calc_optical_flow(
        self, prev_frame: np.ndarray, cur_frame: np.ndarray
    ) -> np.ndarray:
        """calculate optical flow
        Args:
            prev_frame: previous frame (H, W, 3)
            cur_frame: current frame (H, W, 3)
        Return:
            flow: optical flow (H, W, 2)
        """
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame,
            cur_frame,
            None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.window_size,
            iterations=self.n_iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=self.flags,
        )
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

        for timestamp in range(self.interval, n_frames, self.interval):
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
    flow_processor = GunnarFarnebackOpticalFlow()
    flow_processor.demo()
