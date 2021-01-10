import dataclasses
from typing import Tuple

import cv2
import numpy as np


@dataclasses.dataclass
class LucasKanadeOpticalFlow(object):
    """Calculate Lucas-Kanade optical flow
    reference:
    http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    """

    # parameters for Shi-Tomasi corner detection
    max_corners: int = 100
    quality_level: float = 0.3
    min_distance: int = 7
    block_size: int = 7

    # parameters for Lucas-Kanade optical flow
    window_size: Tuple[int, int] = (15, 15)
    max_level: int = 2
    criteria: Tuple[int, int, int] = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03,
    )

    def _setup_demo(self, height: int, width: int) -> None:
        # create radom colors for visualization
        self.colors = np.random.randint(0, 255, (self.max_corners, 3))

        # image for visualization
        self.flow_mask = np.zeros((height, width, 3), np.uint8)

    def _calc_optical_flow(
        self,
        prev_gray: np.ndarray,
        cur_gray: np.ndarray,
        prev_feature_positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # feature_next -> new position of input features.
        # status -> whether the corresponding features has been found or not.
        cur_feature_positions, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            cur_gray,
            prev_feature_positions,
            None,
            winSize=self.window_size,
            maxLevel=self.max_level,
            criteria=self.criteria,
        )

        if cur_feature_positions is not None:
            prev_feature_positions = prev_feature_positions[status == 1]
            cur_feature_positions = cur_feature_positions[status == 1]

        return prev_feature_positions, cur_feature_positions

    def _draw_optical_flow_line(
        self,
        frame: np.ndarray,
        prev_feature_positions: np.ndarray,
        cur_feature_positions: np.ndarray,
    ) -> None:
        # visualize optical flow
        for i, (cur_point, prev_point) in enumerate(
            zip(cur_feature_positions, prev_feature_positions)
        ):
            # get coordianate
            prev_x, prev_y = prev_point.ravel()
            cur_x, cur_y = cur_point.ravel()

            # line between first_point and cur_point
            cv2.line(
                self.flow_mask,
                (cur_x, cur_y),
                (prev_x, prev_y),
                self.colors[i].tolist(),
                2,
            )

            cv2.circle(frame, (cur_x, cur_y), 5, self.colors[i].tolist(), -1)

        return frame

    def demo(self) -> None:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        H, W, _ = frame.shape

        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # corner detection
        prev_feature_positions = cv2.goodFeaturesToTrack(
            prev_gray,
            mask=None,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
        )

        self._setup_demo(H, W)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            prev_feature_positions, cur_feature_positions = self._calc_optical_flow(
                prev_gray, cur_gray, prev_feature_positions
            )

            frame = self._draw_optical_flow_line(
                frame, prev_feature_positions, cur_feature_positions
            )

            frame = cv2.add(frame, self.flow_mask)

            cv2.imshow("result", frame)

            k = cv2.waitKey(1)
            if k == ord("q"):
                break

            prev_gray = cur_gray.copy()
            prev_feature_positions = cur_feature_positions.reshape(-1, 1, 2)

        cv2.destroyAllWindows()
        cap.release()


if __name__ == "__main__":
    flow_processor = LucasKanadeOpticalFlow()
    flow_processor.demo()
