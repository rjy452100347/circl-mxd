import cv2
import numpy as np

from src.utils.common import load_image


def test_load_image_supports_chinese_windows_path(tmp_path):
    directory = tmp_path / "测试地图"
    directory.mkdir()
    path = directory / "地图.png"
    expected = np.zeros((8, 12, 3), dtype=np.uint8)
    expected[:, :] = (17, 83, 211)
    success, encoded = cv2.imencode(".png", expected)
    assert success
    encoded.tofile(path)

    loaded = load_image(str(path))

    assert loaded.shape == expected.shape
    assert np.array_equal(loaded, expected)
