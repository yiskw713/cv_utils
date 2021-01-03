"""
Ref: https://qiita.com/hitomatagi/items/a4ecf7babdbe710208ae
cv2.motempl.updateMotionHistory works with opencv 3.4.11.45.
The later version does not seem to support this function.
"""

import dataclasses
import os
from typing import List

import cv2
import numpy as np


@dataclasses.dataclass
class MotionHistoryImageProcessor(object):
    """Calculate motion history image proposed in James W.

    Davis and Aaron F. Bobick
    'The Representation and Recognition of Action Using Temporal Templates'
    in CVPR 1997
    http://alumni.media.mit.edu/~jdavis/Publications/publications_402.pdf
    """

    # how long the motion is kept
    duration: int = 30

    # threshold for diff image
    threshold: int = 32

    # how often the process is done.
    interval: int = 2

    def _init_motion_history(self, height: int, width: int) -> None:
        # motion history image
        self.motion_history = np.zeros((height, width), np.float32)

    def _save_motion_history(
        self, normalized_mh: np.ndarray, timestamp: int, save_dir: str
    ) -> None:
        save_path = os.path.join(save_dir, f"motion_history_{timestamp:05}.png")
        cv2.imwrite(save_path, normalized_mh)

    def _normalize_motion_history(self, timestamp: int) -> np.ndarray:
        # see http://opencv.jp/opencv-2svn/py/video_motion_analysis_and_object_tracking.html#updatemotionhistory
        normalized_mh = np.uint8(
            np.clip(
                (self.motion_history - (timestamp - self.duration)) / self.duration,
                0,
                1,
            )
            * 255
        )
        return normalized_mh

    def _update_motion_history(
        self, cur_frame: np.ndarray, prev_frame: np.ndarray, timestamp: int
    ) -> None:
        bgr_diff = cv2.absdiff(cur_frame, prev_frame)
        gray_diff = cv2.cvtColor(bgr_diff, cv2.COLOR_BGR2GRAY)
        ret, fgmask = cv2.threshold(gray_diff, self.threshold, 1, cv2.THRESH_BINARY)

        # update motion history
        cv2.motempl.updateMotionHistory(
            fgmask, self.motion_history, timestamp, self.duration
        )

    def calc_motion_history(
        self,
        frames: List[np.ndarray],
        save_dir: str,
    ) -> None:
        height, width, _ = frames[0].shape
        self._init_motion_history(height, width)
        prev_frame = frames[0].copy()
        n_frames = len(frames)

        for timestamp in range(self.interval, n_frames, self.interval):
            cur_frame = frames[timestamp]

            self._update_motion_history(cur_frame, prev_frame, timestamp)
            normalized_mh = self._normalize_motion_history(timestamp)

            prev_frame = cur_frame.copy()

            self._save_motion_history(normalized_mh, timestamp, save_dir)

    def realtime_demo(self):
        cap = cv2.VideoCapture(0)
        ret, cur_frame = cap.read()

        height, width, _ = cur_frame.shape
        timestamp = 0
        self._init_motion_history(height, width)

        prev_frame = cur_frame.copy()

        while True:
            ret, cur_frame = cap.read()
            timestamp += 1

            cv2.imshow("raw videos", cur_frame)

            if timestamp % self.interval != 0:
                continue

            self._update_motion_history(cur_frame, prev_frame, timestamp)
            normalized_mh = self._normalize_motion_history(timestamp)

            cv2.imshow("motion history images", normalized_mh)

            prev_frame = cur_frame.copy()

            # keyboard input
            k = cv2.waitKey(1)

            # end
            if k == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    mh_processor = MotionHistoryImageProcessor()
    mh_processor.realtime_demo()


if __name__ == "__main__":
    main()
