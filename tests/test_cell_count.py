from pathlib import Path
import unittest

import cv2
import numpy as np

from app.inference import (
    _count_visible_cells,
    _count_visible_cells_by_contours,
    _count_visible_cells_by_grid,
    _count_visible_cells_with_method,
    _leaf_based_germination_fallback,
    _resolve_cell_count,
    _tiny_green_germination_fallback,
)


ROOT = Path(__file__).resolve().parents[1]


def load_image(relative_path: str):
    image_path = ROOT / relative_path
    if not image_path.exists():
        raise AssertionError(f"Imagem de regressao nao encontrada: {relative_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise AssertionError(f"OpenCV nao conseguiu abrir: {relative_path}")
    return image


class CellCountRegressionTest(unittest.TestCase):
    def test_purple_grid_crop_uses_grid_instead_of_merged_soil_contours(self):
        image = load_image("tests/fixtures/images/purple-grid-24-cells.jpeg")

        self.assertEqual(_count_visible_cells_by_contours(image), 11)
        self.assertEqual(_count_visible_cells_by_grid(image), 24)
        self.assertEqual(_count_visible_cells(image, {"issue": "led_purple"}), 24)

    def test_good_lighting_keeps_legacy_contour_count(self):
        image = load_image("tests/fixtures/images/good-light-12-cells.jpeg")

        self.assertEqual(_count_visible_cells_by_contours(image), 12)
        self.assertEqual(_count_visible_cells_by_grid(image), 12)
        self.assertEqual(_count_visible_cells(image), 12)
        self.assertEqual(_count_visible_cells(image, {"issue": None}), 12)

    def test_side_cropped_purple_tray_does_not_count_border_slivers(self):
        image = load_image("tests/fixtures/images/side-crop-purple-12-cells.jpeg")

        self.assertEqual(_count_visible_cells_by_contours(image), 14)
        self.assertEqual(_count_visible_cells_by_grid(image), 12)
        self.assertEqual(_count_visible_cells(image, {"issue": "led_purple"}), 12)
        self.assertEqual(_count_visible_cells_with_method(image, {"issue": "led_purple"}), (12, "grid"))

    def test_grid_count_can_equal_germinated_count(self):
        self.assertEqual(_resolve_cell_count(12, 12, None, raw_method="grid"), (12, "detected_visible"))
        self.assertEqual(_resolve_cell_count(4, 4, None, raw_method="contours")[1], "fallback_default")

    def test_leaf_fallback_ignores_low_confidence_edge_leaf(self):
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        image[100:240, 450:499] = (0, 180, 0)
        boxes = [("Folha", 0.60, (450, 100, 499, 240))]

        result = _leaf_based_germination_fallback(image, boxes, 500, 500)

        self.assertEqual([box[0] for box in result], ["Folha"])

    def test_tiny_green_fallback_finds_small_seedling_inside_empty_grid_cell(self):
        image = load_image("tests/fixtures/images/purple-grid-mini-seedling.jpeg")

        result = _tiny_green_germination_fallback(image, [])
        germ_boxes = [bbox for cls_name, _, bbox in result if cls_name == "Germinacao"]

        self.assertEqual(len(germ_boxes), 1)
        x1, y1, x2, y2 = germ_boxes[0]
        self.assertTrue(780 <= x1 <= 820)
        self.assertTrue(140 <= y1 <= 180)
        self.assertTrue(x2 <= 860)
        self.assertTrue(y2 <= 210)


if __name__ == "__main__":
    unittest.main()
