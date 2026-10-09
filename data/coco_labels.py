"""
COCO Annotation Parser and Category Mapping for Selectivity Checks.

Maps natural scene stimuli to high-level semantic categories (faces/persons,
bodies, scenes/places, animals, vehicles, food) to test whether category-selective
visual areas (FFA, EBA, PPA, etc.) demonstrate expected category response profiles.
"""

from typing import Dict, List, Optional, Set
import json
import os
import numpy as np

# Core category groupings mapped to visual ROIs
SUPER_CATEGORIES = {
    "person_face": ["person"],
    "body": ["person"],
    "place_scene": ["building", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench"],
    "animal": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "vehicle": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    "food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]
}

# Target category-to-ROI expectations
CATEGORY_TO_EXPECTED_ROI = {
    "person_face": ["FFA", "OFA"],
    "body": ["EBA"],
    "place_scene": ["PPA", "RSC", "OPA"],
}


class COCOCategoryManager:
    """Manages COCO annotations and categorizes images for selectivity analyses."""

    def __init__(self, image_to_categories: Optional[Dict[str, List[str]]] = None):
        """
        Args:
            image_to_categories: Map of image filename or ID -> list of category names.
        """
        self.image_to_categories: Dict[str, List[str]] = image_to_categories or {}

    @classmethod
    def from_coco_instances_json(cls, json_path: str) -> "COCOCategoryManager":
        """Loads categories from a standard COCO instances_val2017.json or train2017.json."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"COCO annotations not found at {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cat_id_to_name = {c["id"]: c["name"] for c in data.get("categories", [])}
        img_id_to_file = {i["id"]: i["file_name"] for i in data.get("images", [])}

        img_to_cats: Dict[str, List[str]] = {fname: [] for fname in img_id_to_file.values()}
        for ann in data.get("annotations", []):
            img_id = ann.get("image_id")
            cat_id = ann.get("category_id")
            if img_id in img_id_to_file and cat_id in cat_id_to_name:
                fname = img_id_to_file[img_id]
                cname = cat_id_to_name[cat_id]
                if cname not in img_to_cats[fname]:
                    img_to_cats[fname].append(cname)

        return cls(img_to_cats)

    def get_category_mask(
        self, image_list: List[str], target_category: str
    ) -> np.ndarray:
        """
        Returns a boolean mask of shape (N,) indicating whether each image
        contains the target category or belongs to its supercategory.
        """
        mask = np.zeros(len(image_list), dtype=bool)
        target_cats: Set[str] = set()

        if target_category in SUPER_CATEGORIES:
            target_cats = set(SUPER_CATEGORIES[target_category])
        else:
            target_cats = {target_category}

        for i, img_name in enumerate(image_list):
            cats = self.image_to_categories.get(img_name, [])
            # Also test base filename if full path provided
            base_name = os.path.basename(img_name)
            cats_base = self.image_to_categories.get(base_name, [])
            all_present = set(cats).union(set(cats_base))
            if any(c in target_cats for c in all_present):
                mask[i] = True

        return mask

    def get_all_group_masks(self, image_list: List[str]) -> Dict[str, np.ndarray]:
        """Returns boolean mask for each key supercategory."""
        return {
            group: self.get_category_mask(image_list, group)
            for group in SUPER_CATEGORIES.keys()
        }
