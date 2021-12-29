import numpy as np
from PIL import Image
import random
from typing import List


class ImageScramble(object):
    """Image Scrambling using Pillow Image.
    Reference:
    https://stackoverflow.com/questions/17777760/mixing-pixels-of-an-image-manually-using-python
    """

    @staticmethod
    def _set_seed_from_image(img: np.ndarray) -> None:
        """画像からハッシュ値を計算し，シードを設定する．
        ハッシュ値は，画像スクランブルの前後で変わってはいけない．
        今回はRGBそれぞれの画素値の平均からハッシュ値を算出した．

        Args:
            img (np.ndarray): input image
        """
        # use RGB mean value to calculate hash value
        average = np.mean(img.reshape(-1, 3), axis=0)
        average = tuple(average)
        random.seed(hash(average))

    @staticmethod
    def _get_pixels(img: np.ndarray) -> np.ndarray:
        """Get the list of pixel values

        Args:
            img (np.ndarray): (h, w, 3)

        Returns:
            np.ndarray: (h*w, 3)
        """
        pixels = img.reshape(-1, 3)
        return pixels

    @staticmethod
    def _restore_image(
        pixels: np.ndarray, original_width: int, original_height: int
    ) -> np.ndarray:
        """Restore an image from the list of pixel values

        Args:
            pixels (np.ndarray): (h*w, 3)
            original_width (int): the width of an original image
            original_height (int): the height of an original image

        Returns:
            np.ndarray: (h, w, 3)
        """
        img = pixels.reshape((original_height, original_width, 3))
        return img

    @staticmethod
    def _scramble_index(pixels: np.ndarray) -> List[int]:
        """Scramble indicies of pixels.

        Args:
            pixels (np.ndarray): (h*w, 3)

        Returns:
            List[int]: shuffled index list
        """
        idx = list(range(len(pixels)))
        random.shuffle(idx)
        return idx

    def _scramble_pixels(self, img: np.ndarray) -> np.ndarray:
        """Scramble pixels

        Args:
            img (np.ndarray): (h, w, 3)

        Returns:
            np.ndarray: (h*w, 3)
        """
        pixels = self._get_pixels(img)
        idx = self._scramble_index(pixels)

        scrambled_pixels = pixels[idx]
        return scrambled_pixels

    def scramble(self, img: Image.Image) -> Image.Image:
        """Do image scrambling.

        Args:
            img (Image.Image): input image

        Returns:
            Image.Image: scrambled image
        """
        w, h = img.size

        img = np.asarray(img)

        self._set_seed_from_image(img)
        scrambled_pixels = self._scramble_pixels(img)
        img = self._restore_image(scrambled_pixels, w, h)
        return Image.fromarray(img)

    def _unscramble_pixels(self, img: np.ndarray) -> np.ndarray:
        """Unscramble pixels

        Args:
            img (np.ndarray): (h, w, 3)

        Returns:
            np.ndarray: (h*w, 3)
        """
        pixels = self._get_pixels(img)

        idx = self._scramble_index(pixels)
        # argsortを使って元画像のピクセルのインデックスを取得する
        idx = np.argsort(idx)

        original_pixels = pixels[idx]
        return original_pixels

    def unscramble(self, img: Image.Image) -> Image.Image:
        """Do unscramgling image

        Args:
            img (Image.Image): input scrambled image.

        Returns:
            Image.Image: unscrambled image.
        """
        w, h = img.size

        img = np.asarray(img)

        self._set_seed_from_image(img)
        original_pixels = self._unscramble_pixels(img)
        img = self._restore_image(original_pixels, w, h)
        return Image.fromarray(img)


if __name__ == "__main__":
    image_path = "../imgs/2007_000032.jpg"
    img = Image.open(image_path)

    processor = ImageScramble()
    scrambled_img = processor.scramble(img)
    scrambled_img.save("scrambled.jpg")
    unscrambled_img = processor.unscramble(scrambled_img)
    unscrambled_img.save("unscrambled.jpg")
