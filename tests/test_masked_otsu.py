import cv2
import numpy as np
import pytest

from src.masked_otsu import OtsuThreshold


class TestMixupCrossEntropy:
    @pytest.fixture()
    def processor(self) -> OtsuThreshold:
        return OtsuThreshold()

    @pytest.fixture()
    def image(self) -> np.ndarray:
        return cv2.imread("./imgs/2007_000032.jpg")

    def test_convert_gray_image(
        self, processor: OtsuThreshold, image: np.ndarray
    ) -> None:
        gray = processor._convert_gray_image(image)
        assert gray.ndim == 2

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = processor._convert_gray_image(image)
        assert gray.ndim == 2

    def test_check_mask(self, processor: OtsuThreshold) -> None:
        target = np.zeros((20, 30)).astype(np.uint8)
        target[0:10, 0:10] = 255

        mask = np.zeros((20, 30)).astype(np.uint8)
        mask[0:10, 0:10] = 1
        res = processor._check_mask(mask)
        assert np.all(res == target)

        mask = np.zeros((20, 30)).astype(np.uint8)
        mask[0:10, 0:10] = 255
        res = processor._check_mask(mask)
        assert np.all(res == target)

        mask = np.zeros((20, 30)).astype(np.uint8)
        mask[0:10, 0:10] = 124
        mask[10:20, 10:20] = 200
        with pytest.raises(ValueError):
            processor._check_mask(mask)

        mask = np.zeros((20, 30, 3)).astype(np.uint8)
        with pytest.raises(ValueError):
            processor._check_mask(mask)

    def test_mask_gray_image(self, processor: OtsuThreshold, image: np.ndarray) -> None:
        gray = processor._convert_gray_image(image)

        mask = np.zeros_like(gray).astype(np.uint8)
        mask[0:20, 0:20] = 255

        # 1D arary
        res = processor._mask_gray_image(gray, mask)
        assert 20 * 20 == len(res)

    def test_calc_hist(self, processor: OtsuThreshold) -> None:
        gray = np.zeros((100, 100)).astype(np.uint8)
        for i in range(1, 100):
            gray[i : i + 1] = i

        hist = processor._calc_hist(gray)
        for i in range(100):
            assert hist[i] == 100

    def test_find_max_separation_th(
        self, processor: OtsuThreshold, image: np.ndarray
    ) -> None:
        gray = processor._convert_gray_image(image)
        res = processor.process(gray)

        _, cv2_res = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        assert np.all(res == cv2_res)
