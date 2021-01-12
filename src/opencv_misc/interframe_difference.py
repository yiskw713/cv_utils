import dataclasses

import cv2
import numpy as np


@dataclasses.dataclass
class InterframeDifferenceProcessor(object):
    threshold: int = 30

    def _update_background(self, gray_frame: np.ndarray) -> None:
        self.background = gray_frame.copy()

    def _calc_interframe_difference(self, target_gray: np.ndarray) -> np.ndarray:
        # calculate difference
        mask = cv2.absdiff(target_gray, self.background)

        mask[mask < self.threshold] = 0
        mask[mask >= self.threshold] = 255

        return mask

    def demo(self) -> None:
        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._update_background(gray)

        while cap.isOpened():
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            mask = self._calc_interframe_difference(gray)

            # visualization
            cv2.imshow("frame", frame)
            cv2.imshow("mask", mask)

            k = cv2.waitKey(1)
            if k == ord("q"):
                # quit
                break
            elif k == ord("s"):
                # update background
                self._update_background(gray)


if __name__ == "__main__":
    processor = InterframeDifferenceProcessor()
    processor.demo()
