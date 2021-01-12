import numpy as np
import pytest

from src.opencv_misc.grabcut import GrabCut


class TestGrabCut:
    def test_setup(self) -> None:
        processor = GrabCut()

        with pytest.raises(AttributeError):
            processor.mask

        with pytest.raises(AttributeError):
            processor.bg_model

        with pytest.raises(AttributeError):
            processor.fg_model

        processor._setup(100, 100)
        assert processor.mask.shape == (100, 100)
        assert processor.mask.dtype == np.uint8
        assert processor.bg_model.dtype == np.float64
        assert processor.fg_model.dtype == np.float64
