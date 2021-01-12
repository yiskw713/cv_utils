import numpy as np
import pytest

from src.opencv_misc.interframe_difference import InterframeDifferenceProcessor


class TestInterframeDifferenceProcessor:
    @pytest.fixture()
    def processor(self) -> InterframeDifferenceProcessor:
        return InterframeDifferenceProcessor(threshold=30)

    def test_update_background(self, processor: InterframeDifferenceProcessor):
        bg = np.zeros((10, 10), np.uint8)
        processor._update_background(bg)

        assert np.all(bg == processor.background)

    def test_calc_interframe_difference(
        self, processor: InterframeDifferenceProcessor
    ) -> None:
        bg = np.zeros((10, 10), np.uint8)
        processor._update_background(bg)

        # target gray image
        target = bg.copy()
        target[1, 1] = 30
        target[2, 2] = 29

        # expected mask
        expected = np.zeros((10, 10), np.uint8)
        expected[1, 1] = 255

        mask = processor._calc_interframe_difference(target)
        assert np.all(mask == expected)
