import cv2
import numpy as np
import pytest

from src.opencv_misc.background_subtraction import BackgrounSubtractor


class TestBackgroundSubtractors:
    def test_get_subtractor(self) -> None:
        with pytest.raises(ValueError):
            BackgrounSubtractor(subtraction_type="dummy")

        for cand in BackgrounSubtractor.subtraction_candidates:
            processor = BackgrounSubtractor(subtraction_type=cand)

            expected_class = getattr(cv2, f"bgsegm_BackgroundSubtractor{cand}")
            assert type(processor.subtractor) is expected_class

    def test_remove_noise(self) -> None:
        processor = BackgrounSubtractor()

        mask = np.zeros((3, 3), np.uint8)
        mask[1, 1] = 255
        mask = processor._remove_noise(mask)

        assert np.all(mask == np.zeros((3, 3), np.uint8))
