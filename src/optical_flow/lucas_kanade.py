import cv2
import numpy as np
import dataclasses
from typing import Tuple


@dataclasses.dataclass(frozen=True)
class LucasKanadeOpticalFlow(object):
    """Calculate Lucas-Kanade optical flow
    reference:
    http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    """

    # parameters for Shi-Tomasi corner detection
    max_corners: int = 100
    quality_level: float = 0.3
    min_distance: int = 7
    block_size: int = 7

    # parameters for Lucas-Kanade optical flow
    window_size: Tuple[int, int] = (15, 15)
    max_level: int = 2
    criteria: Tuple[int, int, int] = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03,
    )

    def demo(self) -> None:
        cap = cv2.VideoCapture(0)
        # 最初のフレーム読み込み
        first_flag, first = cap.read()

        # グレースケールに変換
        gray_first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        feature_first = cv2.goodFeaturesToTrack(
            gray_first,
            mask=None,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
        )

        # create radom colors for visualization
        colors = np.random.randint(0, 255, (self.max_corners, 3))

        # フロー書き出し用の画像作成
        flow_mask = np.zeros_like(first)

        while True:
            # 動画のフレーム取得
            ret, frame = cap.read()

            # 動画のフレームが無くなったら強制終了
            if not ret:
                break

            # グレースケールに変換
            gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # オプティカルフロー検出
            # feature_next : gray_nextの特徴点の座標を保持
            feature_next, status, err = cv2.calcOpticalFlowPyrLK(
                gray_first,
                gray_next,
                feature_first,
                None,
                winSize=self.window_size,
                maxLevel=self.max_level,
                criteria=self.criteria,
            )

            # 特徴点の移動を検出できた場合
            if feature_next is not None:
                # オプティカルフローを検出した特徴点を選別（0：検出せず、1：検出した）
                good_first = feature_first[status == 1]
                good_next = feature_next[status == 1]

            # オプティカルフローを描画
            for i, (next_point, first_point) in enumerate(zip(good_next, good_first)):

                # 前フレームの座標獲得
                first_x, first_y = first_point.ravel()

                # 後フレームの座標獲得
                next_x, next_y = next_point.ravel()

                # 前フレームと後フレームを繋ぐ線を描画
                flow_mask = cv2.line(
                    flow_mask,
                    (next_x, next_y),
                    (first_x, first_y),
                    colors[i].tolist(),
                    2,
                )

                # 現在の特徴点のところに丸（大きな点）を描画
                frame = cv2.circle(frame, (next_x, next_y), 5, colors[i].tolist(), -1)

            output = cv2.add(frame, flow_mask)

            # ウィンドウに結果を表示
            cv2.imshow("window", output)

            # 終了オプション
            k = cv2.waitKey(1)
            if k == ord("q"):
                break

            # 次のフレーム、ポイントの準備
            gray_first = gray_next.copy()
            feature_first = good_next.reshape(-1, 1, 2)

        # 終了処理
        cv2.destroyAllWindows()
        cap.release()


if __name__ == "__main__":
    flow_processor = LucasKanadeOpticalFlow()
    flow_processor.demo()
