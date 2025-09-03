import cv2
import numpy as np
import math


def segment_foot(image):
    """
    Segment the foot from the image using Otsu threshold + contour detection.
    Returns binary mask and rotated image with foot oriented vertically.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, 0
    
    # Find the largest contour (should be the foot)
    foot_contour = max(contours, key=cv2.contourArea)
    
    # Create mask
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [foot_contour], 255)
    
    # Get bounding rectangle for rotation
    rect = cv2.minAreaRect(foot_contour)
    angle = rect[2]
    
    # Adjust angle to make foot vertical
    if rect[1][0] < rect[1][1]:  # width < height
        angle += 90
    
    # Rotate image to make foot vertical
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (w, h))
    
    # Find rotated contour
    contours_rotated, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_rotated:
        rotated_contour = max(contours_rotated, key=cv2.contourArea)
        
        # Check if heel is at top - if so, flip vertically
        x, y, w_box, h_box = cv2.boundingRect(rotated_contour)
        
        # Simple heuristic: check if the top part is wider than bottom (heel wider than toes)
        top_y = y + h_box // 4
        bottom_y = y + 3 * h_box // 4
        
        top_width = np.sum(rotated_mask[top_y, x:x+w_box] > 0)
        bottom_width = np.sum(rotated_mask[bottom_y, x:x+w_box] > 0)
        
        if top_width > bottom_width * 1.2:  # Heel at top
            rotated_image = cv2.flip(rotated_image, 0)
            rotated_mask = cv2.flip(rotated_mask, 0)
    
    return rotated_mask, rotated_image, angle


def measure_foot(mask):
    """
    Measure foot dimensions from the binary mask.
    Returns measurements in pixels.
    """
    if mask is None:
        return None
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    foot_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(foot_contour)
    
    # Foot length is the bounding box height
    foot_length = h
    
    # Measure widths at different points
    # Forefoot at 25% from top (toes area)
    forefoot_y = y + int(0.25 * h)
    forefoot_width = np.sum(mask[forefoot_y, x:x+w] > 0)
    
    # Midfoot at 50%
    midfoot_y = y + int(0.50 * h)
    midfoot_width = np.sum(mask[midfoot_y, x:x+w] > 0)
    
    # Heel at 85% from top
    heel_y = y + int(0.85 * h)
    heel_width = np.sum(mask[heel_y, x:x+w] > 0)
    
    return {
        'length': foot_length,
        'forefoot_width': forefoot_width,
        'midfoot_width': midfoot_width,
        'heel_width': heel_width,
        'bounding_box': (x, y, w, h)
    }


def analyze_arch_type(measurements):
    """
    Classify arch type based on midfoot to forefoot width ratio.
    """
    if not measurements or measurements['forefoot_width'] == 0:
        return 'unknown'
    
    ratio = measurements['midfoot_width'] / measurements['forefoot_width']
    
    if ratio >= 0.80:
        return 'flat'
    elif ratio >= 0.60:
        return 'normal'
    else:
        return 'high'


def classify_width(measurements):
    """
    Classify foot width based on forefoot width to length ratio.
    """
    if not measurements or measurements['length'] == 0:
        return 'unknown'
    
    ratio = measurements['forefoot_width'] / measurements['length']
    
    if ratio >= 0.42:
        return 'wide'
    elif ratio >= 0.34:
        return 'regular'
    else:
        return 'narrow'


def detect_credit_card(image):
    """
    Detect credit card in image for scale calibration.
    Returns pixels per mm if card detected, None otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Look for quadrilateral (4 corners)
        if len(approx) == 4:
            # Check if it's roughly rectangular
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h
            
            # Credit card aspect ratio is approximately 1.586 (85.6/54)
            if 1.4 < aspect_ratio < 1.8 and cv2.contourArea(contour) > 5000:
                # Calculate pixels per mm
                card_width_mm = 85.6
                card_height_mm = 54.0
                
                pixels_per_mm_w = w / card_width_mm
                pixels_per_mm_h = h / card_height_mm
                
                # Return average
                return (pixels_per_mm_w + pixels_per_mm_h) / 2
    
    return None


def convert_to_shoe_sizes(length_mm):
    """
    Convert foot length in mm to approximate US shoe sizes.
    """
    # Approximate conversion tables
    men_sizes = [
        (228, 6.0), (234, 6.5), (240, 7.0), (246, 7.5), (251, 8.0),
        (257, 8.5), (263, 9.0), (269, 9.5), (275, 10.0), (280, 10.5),
        (286, 11.0), (292, 11.5), (298, 12.0), (303, 12.5), (309, 13.0)
    ]
    
    women_sizes = [
        (201, 5.0), (207, 5.5), (213, 6.0), (219, 6.5), (225, 7.0),
        (230, 7.5), (236, 8.0), (242, 8.5), (248, 9.0), (254, 9.5),
        (260, 10.0), (266, 10.5), (271, 11.0), (277, 11.5), (283, 12.0)
    ]
    
    def find_closest_size(length, size_table):
        for i, (mm, size) in enumerate(size_table):
            if length <= mm:
                return size
        return size_table[-1][1]  # Return largest size if beyond table
    
    men_size = find_closest_size(length_mm, men_sizes)
    women_size = find_closest_size(length_mm, women_sizes)
    
    return men_size, women_size


def create_measurement_overlay(image, measurements, pixels_per_mm=None):
    """
    Create overlay image with measurement lines.
    """
    overlay = image.copy()
    
    if not measurements:
        return overlay
    
    x, y, w, h = measurements['bounding_box']
    
    # Green color for lines
    color = (0, 255, 0)
    thickness = 2
    
    # Draw measurement lines at 25%, 50%, 85% of foot length
    forefoot_y = y + int(0.25 * h)
    midfoot_y = y + int(0.50 * h)
    heel_y = y + int(0.85 * h)
    
    cv2.line(overlay, (x, forefoot_y), (x + w, forefoot_y), color, thickness)
    cv2.line(overlay, (x, midfoot_y), (x + w, midfoot_y), color, thickness)
    cv2.line(overlay, (x, heel_y), (x + w, heel_y), color, thickness)
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_color = (0, 255, 0)
    
    cv2.putText(overlay, "Forefoot", (x + w + 5, forefoot_y), font, font_scale, text_color, 1)
    cv2.putText(overlay, "Midfoot", (x + w + 5, midfoot_y), font, font_scale, text_color, 1)
    cv2.putText(overlay, "Heel", (x + w + 5, heel_y), font, font_scale, text_color, 1)
    
    return overlay


def process_foot_image(image, manual_length_cm=None):
    """
    Main processing function that combines all CV operations.
    """
    # Segment foot
    mask, rotated_image, rotation_angle = segment_foot(image)
    
    if mask is None:
        return None
    
    # Measure foot
    measurements = measure_foot(mask)
    
    if not measurements:
        return None
    
    # Analyze foot characteristics
    arch_type = analyze_arch_type(measurements)
    width_category = classify_width(measurements)
    
    # Scale calibration
    pixels_per_mm = None
    length_mm = None
    
    if manual_length_cm:
        pixels_per_mm = measurements['length'] / (manual_length_cm * 10)
        length_mm = manual_length_cm * 10
    
    # Convert to shoe sizes if we have real measurements
    men_size = women_size = None
    if length_mm:
        men_size, women_size = convert_to_shoe_sizes(length_mm)
    
    # Create overlay image
    overlay = create_measurement_overlay(rotated_image, measurements, pixels_per_mm)
    
    return {
        'overlay_image': overlay,
        'measurements': measurements,
        'arch_type': arch_type,
        'width_category': width_category,
        'length_mm': length_mm,
        'men_size': men_size,
        'women_size': women_size,
        'pixels_per_mm': pixels_per_mm
    }