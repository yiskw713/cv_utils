import os

import numpy as np
import pytest

from src.optical_flow.gunnar_farneback import GunnarFarnebackOpticalFlow


class TestGunnarFarnebackOpticalFlow:
    @pytest.fixture()
    def processor(self) -> GunnarFarnebackOpticalFlow:
        return GunnarFarnebackOpticalFlow()

    def test_process(self, processor: GunnarFarnebackOpticalFlow) -> None:
        frames = [np.random.randint(0, 256, (224, 224, 3), np.uint8) for _ in range(50)]
        save_dir = "tests/tmp/"

        processor.process(frames, save_dir)
        for i in range(1, 50):
            flow_path = os.path.join(save_dir, f"flow_{i:05}.npy")
            assert os.path.exists(flow_path)

            flow = np.load(flow_path)
            assert flow.shape == (224, 224, 2)
            assert np.any(np.where(flow != 0.0))

            os.remove(flow_path)
