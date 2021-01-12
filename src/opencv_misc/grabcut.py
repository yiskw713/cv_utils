import dataclasses
from typing import Tuple

import cv2
import numpy as np


@dataclasses.dataclass
class GrabCut(object):
    n_iteration: int = 5
    mode: int = cv2.GC_INIT_WITH_RECT

    def _setup(self, height, width) -> None:
        # mask stores 0, 1, 2 or 3, each of which means
        # background, foreground, probably background and probably foreground
        self.mask = np.zeros((height, width), np.uint8)

        # TODO: what does size (1, 65) mean?
        self.bg_model = np.zeros((1, 65), np.float64)
        self.fg_model = np.zeros((1, 65), np.float64)

    def demo(self, img_path: str, foreground_rect: Tuple[int, int, int, int]) -> None:
        """
        Args:
            img_path: the path to an image you want to proecess.
            background_rect: the coordinates of a rectangle which includes
                             the foreground object in the format (x,y,w,h)
        """
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        self._setup(h, w)

        cv2.grabCut(
            img,
            self.mask,
            foreground_rect,
            self.bg_model,
            self.fg_model,
            self.n_iteration,
            self.mode,
        )

        # remove background
        result_mask = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype(
            np.uint8
        )
        img = img * result_mask[..., None]

        cv2.imshow("result", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    processor = GrabCut()
    # image size is (700, 700)
    processor.demo("./imgs/cat.jpg", (100, 50, 400, 600))
