"""Shared visualization functions for YOLO Pool."""

import cv2

# COCO skeleton connections (1-indexed pairs)
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # legs
    [6, 12], [7, 13],   # torso-legs
    [6, 7],              # shoulders
    [6, 8], [7, 9],     # upper arms
    [8, 10], [9, 11],   # lower arms
    [2, 3], [1, 2], [1, 3],  # face
    [2, 4], [3, 5],     # ears
    [4, 6], [5, 7]      # ear-shoulder
]

# Left/right keypoint indices for coloring
LEFT_INDICES = {1, 3, 5, 7, 9, 11, 13, 15}
RIGHT_INDICES = {2, 4, 6, 8, 10, 12, 14, 16}

# Class colors (BGR)
CLASS_COLORS = {
    'adult': (0, 200, 255),   # cyan
    'child': (255, 147, 0),   # orange
}


def draw_skeleton(image, keypoints):
    """Draw pose skeleton on image using COCO 17 keypoint format."""
    kpt_color = (0, 255, 255)     # yellow (center)
    left_color = (255, 128, 0)    # orange (left side)
    right_color = (51, 153, 255)  # blue (right side)

    # Draw keypoint dots
    for idx, kpt in enumerate(keypoints):
        x, y, conf = kpt
        if conf > 0.5:
            if idx in LEFT_INDICES:
                color = left_color
            elif idx in RIGHT_INDICES:
                color = right_color
            else:
                color = kpt_color
            cv2.circle(image, (int(x), int(y)), 3, color, -1)
            cv2.circle(image, (int(x), int(y)), 4, (0, 0, 0), 1)

    # Draw skeleton lines
    for idx1, idx2 in SKELETON:
        idx1 -= 1  # convert to 0-indexed
        idx2 -= 1
        if idx1 < len(keypoints) and idx2 < len(keypoints):
            kpt1, kpt2 = keypoints[idx1], keypoints[idx2]
            if kpt1[2] > 0.5 and kpt2[2] > 0.5:
                pt1 = (int(kpt1[0]), int(kpt1[1]))
                pt2 = (int(kpt2[0]), int(kpt2[1]))
                cv2.line(image, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)


def draw_bbox(image, box, class_name, conf):
    """Draw styled bounding box with label for a detected person.

    Args:
        image: Frame to draw on (modified in-place).
        box: (x, y, w, h) center-format bounding box.
        class_name: 'adult' or 'child'.
        conf: Detection confidence.
    """
    x, y, w, h = box
    x1, y1 = int(x - w / 2), int(y - h / 2)
    x2, y2 = int(x + w / 2), int(y + h / 2)

    box_color = CLASS_COLORS.get(class_name, (147, 20, 255))

    # Semi-transparent bbox fill
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
    cv2.addWeighted(overlay, 0.15, image, 0.85, 0, image)

    # Bbox border
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 1, cv2.LINE_AA)

    # Label
    label = f"{class_name} {conf:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

    pad_x, pad_y = 6, 4
    lx1 = x1
    ly1 = y1 - th - pad_y * 2
    lx2 = x1 + tw + pad_x * 2
    ly2 = y1

    # Rounded label background
    r = 5
    overlay2 = image.copy()
    cv2.rectangle(overlay2, (lx1 + r, ly1), (lx2 - r, ly2), box_color, -1)
    cv2.rectangle(overlay2, (lx1, ly1 + r), (lx2, ly2 - r), box_color, -1)
    cv2.circle(overlay2, (lx1 + r, ly1 + r), r, box_color, -1)
    cv2.circle(overlay2, (lx2 - r, ly1 + r), r, box_color, -1)
    cv2.circle(overlay2, (lx1 + r, ly2 - r), r, box_color, -1)
    cv2.circle(overlay2, (lx2 - r, ly2 - r), r, box_color, -1)
    cv2.addWeighted(overlay2, 0.85, image, 0.15, 0, image)

    # Label text with shadow
    tx = lx1 + pad_x
    ty = ly2 - pad_y
    cv2.putText(image, label, (tx + 1, ty + 1), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(image, label, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_status(image, box, status, color):
    """Draw status text below bounding box."""
    x, y, w, h = box
    x1 = int(x - w / 2)
    y2 = int(y + h / 2)

    if status and status != "Active":
        font = cv2.FONT_HERSHEY_SIMPLEX
        (sw, sh), _ = cv2.getTextSize(status, font, 0.4, 1)
        sx = x1
        sy = y2 + sh + 6
        cv2.putText(image, status, (sx + 1, sy + 1), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, status, (sx, sy), font, 0.4, color, 1, cv2.LINE_AA)
