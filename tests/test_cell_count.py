from pathlib import Path
import unittest

import cv2
import numpy as np

from app.inference import (
    _count_visible_cells,
    _count_visible_cells_by_contours,
    _count_visible_cells_by_grid,
    _count_visible_cells_with_method,
    _assess_image_quality,
    _enhance_for_yolo,
    _leaf_based_germination_fallback,
    _naturalize_magenta_for_display,
    _plant_signal_ratio,
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

    def test_good_lighting_side_crop_prefers_grid_when_contours_pick_extra_regions(self):
        image = load_image("tests/fixtures/images/good-light-side-crop-12-cells.jpeg")

        self.assertIsNone(_assess_image_quality(image)["issue"])
        self.assertEqual(_count_visible_cells_by_contours(image), 13)
        self.assertEqual(_count_visible_cells_by_grid(image), 12)
        self.assertEqual(_count_visible_cells_with_method(image), (12, "grid"))

    def test_magenta_reconstruction_restores_plant_signal_for_inference(self):
        image = load_image("tests/fixtures/images/magenta-full-tray.jpeg")
        quality = _assess_image_quality(image)
        self.assertEqual(quality["issue"], "led_magenta")

        x1, y1, x2, y2 = (631, 1156, 716, 1245)
        original_crop = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        enhanced = _enhance_for_yolo(image, quality)
        enhanced_crop = cv2.cvtColor(enhanced[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)

        self.assertEqual(_plant_signal_ratio(original_crop), 0.0)
        self.assertGreaterEqual(_plant_signal_ratio(enhanced_crop), 0.40)

    def test_magenta_display_naturalization_removes_magenta_cast(self):
        image = load_image("tests/fixtures/images/magenta-full-tray.jpeg")
        quality = _assess_image_quality(image)
        enhanced = _enhance_for_yolo(image, quality)

        display = _naturalize_magenta_for_display(
            image,
            enhanced,
            [(631, 1156, 716, 1245, 0.8)],
        )
        original_mean = image.reshape(-1, 3).mean(axis=0)
        display_mean = display.reshape(-1, 3).mean(axis=0)
        plant_crop = display[1156:1245, 631:716]
        plant_mean = plant_crop.reshape(-1, 3).mean(axis=0)

        self.assertLess(display_mean[2] - display_mean[1], original_mean[2] - original_mean[1])
        self.assertGreater(plant_mean[1], plant_mean[2])
        self.assertGreater(plant_mean[1], plant_mean[0])

    def test_magenta_display_paints_green_beyond_yolo_boxes(self):
        """Garante que folhas detectadas via HSV (fora dos germ_boxes) também aparecem verdes."""
        image = load_image("tests/fixtures/images/magenta-full-tray.jpeg")
        quality = _assess_image_quality(image)
        enhanced = _enhance_for_yolo(image, quality)
        # Sem germ_boxes (simula YOLO falhando totalmente)
        display = _naturalize_magenta_for_display(image, enhanced, [])
        from app.inference import _plant_mask_from_hsv
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        plant_mask = _plant_mask_from_hsv(hsv, include_led_shadow=True)
        if cv2.countNonZero(plant_mask) < 500:
            self.skipTest("HSV nao detecta planta nessa fixture")
        plant_pixels = display[plant_mask > 0]
        # Onde HSV detecta planta, display.G deve ser claramente maior que R e B
        self.assertGreater(plant_pixels[:, 1].mean(), plant_pixels[:, 2].mean() + 10)
        self.assertGreater(plant_pixels[:, 1].mean(), plant_pixels[:, 0].mean() + 10)

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

    def test_grid_occupation_finds_mature_overlapping_leaves(self):
        from app.inference import _grid_occupation_germination_fallback
        image = load_image("tests/fixtures/images/magenta-full-tray.jpeg")
        result = _grid_occupation_germination_fallback(image, [])
        germ = [b for cls, _, b in result if cls == "Germinacao"]
        self.assertGreater(len(germ), 0, "grid occupation deveria detectar pelo menos uma planta madura")


if __name__ == "__main__":
    unittest.main()
