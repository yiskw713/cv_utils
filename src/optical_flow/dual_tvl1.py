import cv2
import numpy as np
from dense_optical_flow import DenseOpticalFlow


class DualTVL1OpticalFlow(DenseOpticalFlow):
    """Dual TVL1 OpticalFlow
    reference:
    https://docs.opencv.org/3.4/dc/d47/classcv_1_1DualTVL1OpticalFlow.html
    """

    def __init__(self) -> None:
        super().__init__()
        self.processor = cv2.createOptFlow_DualTVL1()

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
        # TODO: this process takes a lot of time. Add a code using gpus.
        flow = self.processor.calc(
            prev_frame,
            cur_frame,
            None,
        )
        return flow


if __name__ == "__main__":
    flow_processor = DualTVL1OpticalFlow()
    flow_processor.demo()
