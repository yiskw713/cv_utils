import cv2
import numpy as np
from dense_optical_flow import DenseOpticalFlow


class GunnarFarnebackOpticalFlow(DenseOpticalFlow):
    """Calculate Gunnar Farneback optical flow
    reference:
    https://docs.opencv.org/3.4/dc/d6b/group__video__track.html
    """

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        window_size: int = 15,
        n_iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        flags: int = 0,
    ) -> None:
        super().__init__()

        # parameters for gunnar farneback optical flow

        # param for image pyramid scale
        # see http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html
        self.pyr_scale = pyr_scale

        self.levels = levels
        self.window_size = window_size
        self.n_iterations = n_iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags

    def _calc_optical_flow(
        self, prev_frame: np.ndarray, cur_frame: np.ndarray
    ) -> np.ndarray:
        """calculate optical flow
        Args:
            prev_frame: previous frame (H, W) Gray scale
            cur_frame: current frame (H, W) Gray scale
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


if __name__ == "__main__":
    flow_processor = GunnarFarnebackOpticalFlow()
    flow_processor.demo()
