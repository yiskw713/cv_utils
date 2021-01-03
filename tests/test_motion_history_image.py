import os

import cv2
import numpy as np
import pytest

from src.motion_history_image import MotionHistoryImageProcessor


class TestMixupCrossEntropy:
    @pytest.fixture()
    def processor(self) -> MotionHistoryImageProcessor:
        return MotionHistoryImageProcessor()

    def test_init_motion_history(self, processor: MotionHistoryImageProcessor) -> None:
        processor._init_motion_history(224, 224)
        h, w = processor.motion_history.shape
        dtype = processor.motion_history.dtype

        assert (h, w) == (224, 224)
        assert dtype == np.float32
        assert np.all(processor.motion_history == 0.0)

    def test_save_motion_history(self, processor: MotionHistoryImageProcessor) -> None:
        processor._init_motion_history(224, 224)

        normalized_mh = np.random.randint(0, 256, size=(224, 224), dtype=np.uint8)
        processor._save_motion_history(normalized_mh, 1, "tests/tmp")

        file_path = "tests/tmp/motion_history_00001.png"
        assert os.path.exists(file_path)

        saved_mh = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        assert np.all(saved_mh == normalized_mh)

        # delete unnecessary file
        os.remove(file_path)

    def test_normalize_motion_history(
        self, processor: MotionHistoryImageProcessor
    ) -> None:
        processor.motion_history = np.array([[0, 3], [6, 12]], np.float32)
        normalized_mh = processor._normalize_motion_history(30)

        expected_mh = np.array([[0, 0.1], [0.2, 0.4]]) * 255
        expected_mh = expected_mh.astype(np.uint8)
        assert np.all(normalized_mh == expected_mh)

    def test_update_motion_history(
        self, processor: MotionHistoryImageProcessor
    ) -> None:
        processor._init_motion_history(2, 2)

        # prev_frame.shape -> (2, 2, 3)
        prev_frame = np.array([[0, 0], [0, 0]], dtype=np.uint8)[:, :, None]
        prev_frame = np.repeat(prev_frame, 3, axis=2)

        cur_frame = np.array([[255, 0], [0, 0]], dtype=np.uint8)[:, :, None]
        cur_frame = np.repeat(cur_frame, 3, axis=2)

        processor._update_motion_history(cur_frame, prev_frame, 1)

        motion_history = processor.motion_history
        prev_frame = cur_frame.copy()

        assert np.all(
            motion_history == np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        )

        cur_frame = np.array([[255, 255], [0, 0]], dtype=np.uint8)[:, :, None]
        cur_frame = np.repeat(cur_frame, 3, axis=2)

        processor._update_motion_history(cur_frame, prev_frame, 2)

        assert np.all(
            motion_history == np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32)
        )

    def test_calc_motion_history(self, processor: MotionHistoryImageProcessor) -> None:
        frames = [
            np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        save_dir = "tests/tmp"

        processor.calc_motion_history(frames, save_dir)

        # interval is set to 2
        for i in range(2, 10, 2):
            file_path = os.path.join(save_dir, f"motion_history_{i:05}.png")
            assert os.path.exists(file_path)

            # delete unnecessary file
            os.remove(file_path)
