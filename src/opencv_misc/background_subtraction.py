import dataclasses
from typing import ClassVar, Tuple

import cv2
import numpy as np


@dataclasses.dataclass
class BackgrounSubtractor(object):
    # for Morphological Transformations to remove noise
    kernel_type: int = cv2.MORPH_ELLIPSE
    kernel_size: Tuple[int, int] = (3, 3)

    subtraction_type: str = "MOG"
    subtraction_candidates: ClassVar[Tuple[str, ...]] = (
        "CNT",
        "GMG",
        "GSOC",
        "LSBP",
        "MOG",
    )

    def __post_init__(self) -> None:
        self.kernel = cv2.getStructuringElement(self.kernel_type, self.kernel_size)
        self.subtractor = self._get_subtractor()

    def _get_subtractor(self):
        if self.subtraction_type not in self.subtraction_candidates:
            raise ValueError

        subtractor_class = getattr(
            cv2.bgsegm, f"createBackgroundSubtractor{self.subtraction_type}"
        )

        # instantiation
        return subtractor_class()

    def _remove_noise(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        return mask

    def demo(self) -> None:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()

            # background subtraction
            fgmask = self.subtractor.apply(frame)

            fgmask = self._remove_noise(fgmask)

            # visualization
            cv2.imshow("video", frame)
            cv2.imshow("frame", fgmask)

            k = cv2.waitKey(1)
            if k == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    subtractor = BackgrounSubtractor()
    subtractor.demo()
