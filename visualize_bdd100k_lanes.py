#!/usr/bin/env python3
"""
BDD100K Lane Segmentation Visualizer
Displays original images alongside color-coded lane segmentation masks
"""

import cv2
import numpy as np
import os
import sys
import glob
from pathlib import Path


# Lane category names (bits 0-2)
LANE_CATEGORIES = {
    0: "Crosswalk",
    1: "Double Other",
    2: "Double White",
    3: "Double Yellow",
    4: "Road Curb",
    5: "Single Other",
    6: "Single White",
    7: "Single Yellow"
}

# Color map for each lane category (BGR format for OpenCV)
LANE_COLORS = {
    0: (255, 200, 0),    # Crosswalk - Cyan
    1: (128, 128, 128),  # Double Other - Gray
    2: (255, 255, 255),  # Double White - White
    3: (0, 255, 255),    # Double Yellow - Yellow
    4: (0, 165, 255),    # Road Curb - Orange
    5: (128, 0, 128),    # Single Other - Purple
    6: (200, 200, 200),  # Single White - Light Gray
    7: (0, 200, 255)     # Single Yellow - Gold
}


def decode_lane_mask(mask):
    """
    Decode BDD100K lane mask pixel encoding

    Bit layout:
    - Bits 0-2: Lane category (0-7)
    - Bit 3: Lane style (0=solid, 1=dashed)
    - Bit 4: Lane direction (0=parallel, 1=vertical)
    - Bit 5: Background flag (0=lane, 1=background)
    - 255: Ignore pixel

    Args:
        mask: Grayscale image (H, W) with encoded lane information

    Returns:
        category: Lane category (0-7)
        style: Lane style (0=solid, 1=dashed)
        direction: Lane direction (0=parallel, 1=vertical)
        is_background: Background flag
    """
    # Extract category (bits 0-2)
    category = mask & 0b00000111

    # Extract style (bit 3)
    style = (mask >> 3) & 0b1

    # Extract direction (bit 4)
    direction = (mask >> 4) & 0b1

    # Extract background flag (bit 5)
    is_background = (mask >> 5) & 0b1

    return category, style, direction, is_background


def create_colored_mask(mask):
    """
    Create a color-coded visualization of the lane segmentation mask

    Args:
        mask: Grayscale lane mask (H, W)

    Returns:
        colored_mask: RGB image (H, W, 3) with color-coded lanes
    """
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Decode the mask
    category, style, direction, is_background = decode_lane_mask(mask)

    # Apply colors based on category
    for cat_id, color in LANE_COLORS.items():
        # Create mask for this category (exclude background and ignore pixels)
        lane_pixels = (category == cat_id) & (is_background == 0) & (mask != 255)
        colored_mask[lane_pixels] = color

    return colored_mask


def create_legend(height=720, width=300):
    """
    Create a color legend showing lane categories

    Args:
        height: Legend height
        width: Legend width

    Returns:
        legend: Image with color legend
    """
    legend = np.zeros((height, width, 3), dtype=np.uint8)

    # Title
    cv2.putText(legend, "Lane Categories", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw each category
    y_offset = 70
    rect_height = 40
    rect_width = 60

    for cat_id in range(8):
        # Draw colored rectangle
        color = LANE_COLORS[cat_id]
        cv2.rectangle(legend,
                     (10, y_offset),
                     (10 + rect_width, y_offset + rect_height),
                     color, -1)

        # Draw label
        label = LANE_CATEGORIES[cat_id]
        cv2.putText(legend, label,
                   (80, y_offset + 27),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y_offset += rect_height + 10

    return legend


def visualize_bdd100k_lane(image_path, mask_path, show_legend=True):
    """
    Visualize BDD100K image and lane segmentation mask side-by-side

    Args:
        image_path: Path to original image
        mask_path: Path to lane segmentation mask
        show_legend: Whether to show color legend
    """
    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not load mask from {mask_path}")
        return

    # Create colored segmentation mask
    colored_mask = create_colored_mask(mask)

    # Add labels to images
    image_labeled = image.copy()
    cv2.putText(image_labeled, "Original Image", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    mask_labeled = colored_mask.copy()
    cv2.putText(mask_labeled, "Lane Segmentation", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Combine side-by-side
    if show_legend:
        legend = create_legend(height=image.shape[0])
        combined = np.hstack([image_labeled, mask_labeled, legend])
    else:
        combined = np.hstack([image_labeled, mask_labeled])

    # Display
    window_name = f"BDD100K Lane Visualization - {os.path.basename(image_path)}"
    cv2.imshow(window_name, combined)

    print(f"\nDisplaying: {os.path.basename(image_path)}")
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print("\nPress any key to close the window...")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_image_mask_pairs(dataset_path, split='train', max_samples=None):
    """
    Find matching image and mask pairs in the dataset

    Args:
        dataset_path: Root path to BDD100K dataset
        split: 'train' or 'val'
        max_samples: Maximum number of samples to return (None for all)

    Returns:
        List of (image_path, mask_path) tuples
    """
    image_dir = os.path.join(dataset_path, 'images', '100k', split)
    mask_dir = os.path.join(dataset_path, 'labels', 'lane', 'masks', split)

    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        return []

    if not os.path.exists(mask_dir):
        print(f"Error: Mask directory not found: {mask_dir}")
        return []

    # Get all image files
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))

    pairs = []
    for image_path in image_files:
        # Construct expected mask path
        image_name = os.path.basename(image_path)
        mask_name = image_name.replace('.jpg', '.png')
        mask_path = os.path.join(mask_dir, mask_name)

        # Check if mask exists
        if os.path.exists(mask_path):
            pairs.append((image_path, mask_path))

        if max_samples and len(pairs) >= max_samples:
            break

    return pairs


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_bdd100k_lanes.py <dataset_path> [split] [index]")
        print("\nArguments:")
        print("  dataset_path : Path to BDD100K dataset root directory")
        print("  split        : 'train' or 'val' (default: 'train')")
        print("  index        : Image index to visualize (default: 0)")
        print("\nExample:")
        print("  python visualize_bdd100k_lanes.py ./bdd100k train 0")
        sys.exit(1)

    dataset_path = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else 'train'
    index = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    print(f"Loading BDD100K dataset from: {dataset_path}")
    print(f"Split: {split}")

    # Find image-mask pairs
    pairs = find_image_mask_pairs(dataset_path, split)

    if not pairs:
        print("Error: No image-mask pairs found!")
        print("\nExpected directory structure:")
        print("  dataset_path/")
        print("    images/100k/train/  (or val/)")
        print("    labels/lane/masks/train/  (or val/)")
        sys.exit(1)

    print(f"Found {len(pairs)} image-mask pairs")

    if index >= len(pairs):
        print(f"Error: Index {index} out of range (max: {len(pairs) - 1})")
        sys.exit(1)

    # Visualize the selected pair
    image_path, mask_path = pairs[index]
    visualize_bdd100k_lane(image_path, mask_path, show_legend=True)


if __name__ == "__main__":
    main()
