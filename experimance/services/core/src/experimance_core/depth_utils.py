import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np


logger = logging.getLogger(__name__)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

def mask_bright_area(image: np.ndarray) -> np.ndarray:
    """Create a mask for the bright area in the center of the image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    height, width = gray_image.shape
    center = (width // 2, height // 2)
    
    # For tight depth ranges, we need adaptive thresholding
    # Calculate image statistics to determine appropriate threshold
    non_zero_pixels = gray_image[gray_image > 0]
    if len(non_zero_pixels) == 0:
        # No depth data, return circular fallback
        logger.debug("No depth data found, using circular fallback mask")
        fallback_mask = np.zeros((height, width), dtype=np.uint8)
        radius = int(min(width, height) * 0.35)
        cv2.circle(fallback_mask, center, radius, (255,), -1)
        return fallback_mask

    # Use adaptive threshold based on image statistics
    non_zero_pixels = non_zero_pixels.astype(np.float32)  # Ensure correct dtype for mean/std
    mean_val = np.mean(non_zero_pixels)
    std_val = np.std(non_zero_pixels)
    
    # For tight depth ranges, use a higher threshold relative to the data range
    # If std is very low (tight range), use mean - 0.5*std as threshold
    # If std is higher (wide range), use fixed threshold
    if std_val < 20:  # Tight depth range
        threshold = max(10, int(mean_val - 0.5 * std_val))
        logger.debug(f"Tight depth range detected: mean={mean_val:.1f}, std={std_val:.1f}, threshold={threshold:.1f}")
    else:  # Normal depth range
        threshold = 30
    
    _, thresholded = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    
    # Use floodFill to find contiguous bright area from center
    flood_mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(thresholded, flood_mask, center, (255,), loDiff=(20,), upDiff=(20,), flags=cv2.FLOODFILL_MASK_ONLY)
    
    # Crop the flood fill mask
    cropped_mask = flood_mask[1:-1, 1:-1]
    
    # Find and fill contours
    contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cropped_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    
    # Check if flood fill found a reasonable area (for sand bowl detection)
    mask_area = cv2.countNonZero(cropped_mask)
    total_area = height * width
    mask_ratio = mask_area / total_area
    
    logger.debug(f"Mask area ratio: {mask_ratio:.3f} ({mask_area}/{total_area} pixels)")
    
    # If flood fill found too much (>80%) or too little (<5%), create a fallback circular mask
    if mask_ratio < 0.05 or mask_ratio > 0.8:
        logger.debug(f"Flood fill area ratio {mask_ratio:.3f} out of range, using fallback circular mask")
        fallback_mask = np.zeros((height, width), dtype=np.uint8)
        # Create circular mask sized for typical sand bowl (about 70% of smaller dimension)
        radius = int(min(width, height) * 0.35)
        cv2.circle(fallback_mask, center, radius, (255,), -1)
        return fallback_mask
    
    return cropped_mask


def simple_obstruction_detect(image: np.ndarray, size: Tuple[int, int] = (32, 32), pixel_threshold: int = 1, debug_save: bool = False, debug_prefix: str = "hand_detect") -> Optional[bool]:
    """
    Detect obstruction (hands) in the depth image.

    This function checks if the center of a downscaled copy of the image has a significant number of black pixels
    within a circular area, indicating an obstruction like a hand.
    Args:
        image: Input depth image as a numpy array.
        size: Size to resize the image for processing (default is 32x32).
        pixel_threshold: Minimum number of black pixels required to consider it an obstruction.
        debug_save: If True, save debug images to disk
        debug_prefix: Prefix for debug image filenames
    If the image is blank or has no significant content, returns None.
    
    Returns:
        True if obstruction detected, False if not, None if test fails
    """
    resized = cv2.resize(image, size)
    
    # Save debug image before processing if requested
    if debug_save:
        import os
        debug_dir = "debug_hand_detection"
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)  # millisecond timestamp
    
    if is_blank_frame(resized):
        return None
    
    thickness_multiplier = 0.3
    circle_radius = int(size[0] * (1.0 + thickness_multiplier)) // 2
    circle_center = (size[0] // 2, size[1] // 2)
    
    # Mask everything outside the circle as white
    cv2.circle(resized, circle_center, circle_radius, (255, 255, 255), thickness=int(size[0] * thickness_multiplier))
    
    # Count black pixels inside the circle
    not_black_pixels = cv2.countNonZero(resized)
    black_pixels = size[0] * size[1] - not_black_pixels
    
    result = black_pixels > pixel_threshold
    
    # Add debug logging
    if debug_save:
        print(f"🔍 Hand detection: {black_pixels} black pixels (threshold: {pixel_threshold}) → {result}")
        print(f"    Image stats: min={resized.min()}, max={resized.max()}, mean={resized.mean():.1f}")
    
    # Save debug image after processing if requested
    if debug_save:
        # Create a visualization showing the circle and detection result
        debug_vis = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR) if len(resized.shape) == 2 else resized.copy()
        color = (0, 255, 0) if result else (0, 0, 255)  # Green if hand detected, red if not
        cv2.circle(debug_vis, circle_center, circle_radius, color, 2)
        cv2.putText(debug_vis, f"Hand: {result} ({black_pixels}px)", (2, size[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        cv2.imwrite(f"{debug_dir}/{debug_prefix}_processed_{timestamp}.png", resized)
    
    return result


def is_blank_frame(image: np.ndarray, threshold: float = 1.0) -> bool:
    """Detect blank/empty frames."""
    if np.std(image) < threshold:
        logger.debug("Detected blank frame (std dev below threshold)")
        return True
    return False


def detect_difference(image1: Optional[np.ndarray], image2: np.ndarray, threshold: int = 60) -> Tuple[float, np.ndarray]:
    """
    Calculate the amount of difference between two images.
    
    Returns:
        Tuple of (difference_score, frame_to_use_for_next_comparison)
    """
    if image1 is None:
        return threshold + 1, image2
    
    if is_blank_frame(image2):
        return 0, image1
    
    # Calculate absolute difference
    diff = cv2.absdiff(np.asanyarray(image1), np.asanyarray(image2))
    _, binary_diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to remove noise
    cleaned_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_CLOSE, kernel)
    cleaned_diff = cv2.morphologyEx(cleaned_diff, cv2.MORPH_OPEN, kernel)
    
    # Count non-zero pixels
    difference_score = cv2.countNonZero(cleaned_diff)
    
    return difference_score, image2


def calculate_change_score(current_frame: np.ndarray, previous_frame: np.ndarray, threshold: int) -> float:
    """
    Calculate change score between two frames.
    
    Returns:
        Change score as a float between 0.0 and 1.0
    """
    try:
        # Calculate absolute difference
        diff = cv2.absdiff(current_frame, previous_frame)
        
        # Count pixels above threshold
        changed_pixels = np.sum(diff > threshold)
        total_pixels = diff.size
        
        if total_pixels == 0:
            return 0.0
        
        return float(changed_pixels / total_pixels)
        
    except Exception as e:
        logger.warning(f"Change score calculation failed: {e}")
        return 0.0


def analyze_depth_values(depth_image: np.ndarray, config_min_depth: float, config_max_depth: float, 
                        exclude_edges: bool = True, percentile_filter: float = 5.0) -> dict:
    """
    Analyze the raw depth image to determine actual depth range and statistics.
    
    This function assumes the depth image is colorized/processed where:
    - Higher pixel values (brighter) represent closer depths
    - Lower pixel values (darker) represent farther depths
    - The mapping is linear within the configured depth range
    
    Args:
        depth_image: Processed depth image (grayscale, 0-255)
        config_min_depth: Configured minimum depth in meters
        config_max_depth: Configured maximum depth in meters
        exclude_edges: Whether to exclude edge pixels to avoid artifacts
        percentile_filter: Exclude pixels below this percentile and above (100-this) percentile
    
    Returns:
        Dictionary with depth analysis results:
        - actual_min_depth: Estimated closest depth in meters
        - actual_max_depth: Estimated farthest depth in meters
        - depth_range_cm: Actual depth range in centimeters
        - pixel_stats: Dictionary with pixel value statistics
        - center_depth: Estimated depth at center of image
        - mapping_info: Information about the depth-to-pixel mapping
    """
    logger = logging.getLogger(__name__)
    
    # Apply filtering to exclude edge artifacts and outliers
    working_image = depth_image.copy()

    # Exclude edge pixels if requested
    if exclude_edges:
        edge_size = 5  # Exclude 5-pixel border
        working_image[:edge_size, :] = 0
        working_image[-edge_size:, :] = 0
        working_image[:, :edge_size] = 0
        working_image[:, -edge_size:] = 0
        logger.debug(f"Excluded {edge_size}-pixel edge border")
    
    # Get non-zero pixels (ignore black/invalid areas)
    non_zero_pixels = working_image[working_image > 0]
    
    if len(non_zero_pixels) == 0:
        logger.warning("No valid depth data found in image")
        return {
            'actual_min_depth': None,
            'actual_max_depth': None,
            'depth_range_cm': None,
            'pixel_stats': {'min': 0, 'max': 0, 'mean': 0, 'std': 0},
            'center_depth': None,
            'mapping_info': {'valid_data': False}
        }

    # Apply percentile filtering to remove only dark outliers (edge artifacts)
    # but preserve bright pixels (sand peaks)
    if percentile_filter > 0:
        # Only filter out the darkest pixels (likely edge artifacts)
        low_thresh = np.percentile(non_zero_pixels, percentile_filter)
        # Keep all bright pixels - don't filter the high end
        filtered_pixels = non_zero_pixels[non_zero_pixels >= low_thresh]
        logger.debug(f"Low percentile filtering ({percentile_filter}%): {len(non_zero_pixels)} -> {len(filtered_pixels)} pixels")
        logger.debug(f"Filtered range: {low_thresh:.0f} - {np.max(filtered_pixels):.0f} (preserved bright pixels)")
    else:
        filtered_pixels = non_zero_pixels
    
    if len(filtered_pixels) == 0:
        logger.warning("No pixels remaining after percentile filtering")
        return {
            'actual_min_depth': None,
            'actual_max_depth': None,
            'depth_range_cm': None,
            'pixel_stats': {'min': 0, 'max': 0, 'mean': 0, 'std': 0},
            'center_depth': None,
            'mapping_info': {'valid_data': False}
        }

    # Calculate pixel statistics on filtered data
    pixel_min = float(np.min(filtered_pixels))
    pixel_max = float(np.max(filtered_pixels))
    pixel_mean = float(np.mean(filtered_pixels))
    pixel_std = float(np.std(filtered_pixels))
    
    # Calculate depth range
    depth_range_meters = config_max_depth - config_min_depth
    
    # Map pixel values to actual depths based on RealSense colorizer behavior
    # RealSense colorizer mapping (discovered empirically):
    # - Pixel value 255 (or ~232 max observed) = config_min_depth (closest)
    # - Pixel value 1 = config_max_depth (farthest)  
    # - Pixel value 0 = invalid/no data
    # - The colorizer uses linear interpolation between min and max depth
    
    # Use the theoretical colorizer range (1-255) to estimate actual depths
    colorizer_range = 254  # 255 - 1 (valid pixel range)
    
    # Map the observed pixel values back to actual depths using the colorizer's mapping
    # Formula: depth = min_depth + ((255 - pixel_value) / 254) * depth_range
    print(f"Pixel min: {pixel_min}, Pixel max: {pixel_max}")
    actual_min_depth = config_min_depth + ((255 - pixel_max) / colorizer_range) * depth_range_meters
    actual_max_depth = config_min_depth + ((255 - pixel_min) / colorizer_range) * depth_range_meters
    
    logger.debug(f"Pixel range: {pixel_min:.0f} - {pixel_max:.0f}")
    logger.debug(f"Mapped to depths: {actual_min_depth:.3f}m - {actual_max_depth:.3f}m")
    logger.debug(f"Height difference: {(actual_max_depth - actual_min_depth) * 100:.1f}cm")
    
    # Calculate center depth
    height, width = depth_image.shape
    center_y, center_x = height // 2, width // 2
    center_region_size = 20
    
    y1 = max(0, center_y - center_region_size // 2)
    y2 = min(height, center_y + center_region_size // 2)
    x1 = max(0, center_x - center_region_size // 2)
    x2 = min(width, center_x + center_region_size // 2)
    
    center_region = depth_image[y1:y2, x1:x2]
    center_non_zero = center_region[center_region > 0]
    
    if len(center_non_zero) > 0:
        center_pixel_mean = float(np.mean(center_non_zero))
        # Use the same colorizer mapping: depth = min_depth + ((255 - pixel_value) / 254) * depth_range
        colorizer_range = 254  # 255 - 1 (valid pixel range)
        center_depth = config_min_depth + ((255 - center_pixel_mean) / colorizer_range) * depth_range_meters
        logger.debug(f"Center pixel value: {center_pixel_mean:.1f}, mapped to depth: {center_depth:.3f}m")
    else:
        center_depth = None
    
    # Calculate actual depth range in centimeters
    if actual_min_depth is not None and actual_max_depth is not None:
        depth_range_cm = abs(actual_max_depth - actual_min_depth) * 100
    else:
        depth_range_cm = None
    
    # Create mapping info
    pixel_range = pixel_max - pixel_min
    mapping_info = {
        'valid_data': True,
        'config_range_cm': depth_range_meters * 100,
        'pixel_range': pixel_range if pixel_max != pixel_min else 0,
        'pixels_per_cm': pixel_range / (depth_range_meters * 100) if depth_range_meters > 0 and pixel_max != pixel_min else 0,
        'depth_resolution_mm': (depth_range_meters * 1000) / pixel_range if pixel_max != pixel_min else 0,
        'excluded_edges': exclude_edges,
        'percentile_filter': percentile_filter
    }
    
    return {
        'actual_min_depth': actual_min_depth,
        'actual_max_depth': actual_max_depth,
        'depth_range_cm': depth_range_cm,
        'pixel_stats': {
            'min': pixel_min,
            'max': pixel_max,
            'mean': pixel_mean,
            'std': pixel_std,
            'filtered_count': len(filtered_pixels),
            'total_valid': len(non_zero_pixels)
        },
        'center_depth': center_depth,
        'mapping_info': mapping_info
    }


def log_depth_analysis(depth_image: np.ndarray, config_min_depth: float, config_max_depth: float, frame_number: Optional[int] = None) -> dict:
    """
    Analyze and log depth information for debugging.
    
    Args:
        depth_image: Processed depth image
        config_min_depth: Configured minimum depth
        config_max_depth: Configured maximum depth  
        frame_number: Optional frame number for logging
        
    Returns:
        Analysis results dictionary
    """
    analysis = analyze_depth_values(depth_image, config_min_depth, config_max_depth)
    
    frame_prefix = f"Frame {frame_number}: " if frame_number is not None else ""
    
    if analysis['mapping_info']['valid_data']:
        logger.info(f"{frame_prefix}Depth Analysis:")
        logger.info(f"  Config range: {config_min_depth:.3f}m - {config_max_depth:.3f}m ({analysis['mapping_info']['config_range_cm']:.1f}cm)")
        logger.info(f"  Actual range: {analysis['actual_min_depth']:.3f}m - {analysis['actual_max_depth']:.3f}m ({analysis['depth_range_cm']:.1f}cm)")
        logger.info(f"  Pixel range: {analysis['pixel_stats']['min']:.0f} - {analysis['pixel_stats']['max']:.0f} (mean: {analysis['pixel_stats']['mean']:.1f})")
        
        if analysis['center_depth'] is not None:
            logger.info(f"  Center depth: {analysis['center_depth']:.3f}m ({analysis['center_depth']*100:.1f}cm)")
            
            # Check if center is closer than configured minimum
            if analysis['center_depth'] < config_min_depth:
                logger.warning(f"  ⚠ Center depth ({analysis['center_depth']:.3f}m) is CLOSER than min_depth ({config_min_depth:.3f}m)")
            elif analysis['center_depth'] > config_max_depth:
                logger.warning(f"  ⚠ Center depth ({analysis['center_depth']:.3f}m) is FARTHER than max_depth ({config_max_depth:.3f}m)")
            else:
                logger.info(f"  ✓ Center depth is within configured range")
        
        logger.info(f"  Depth resolution: {analysis['mapping_info']['depth_resolution_mm']:.2f}mm per pixel level")
    else:
        logger.warning(f"{frame_prefix}No valid depth data found")
    
    return analysis


def find_depth_peak(depth_image: np.ndarray, min_distance: int = 20) -> dict:
    """
    Find the location of the depth peak (closest point) in the image.
    
    Args:
        depth_image: Processed depth image (grayscale, 0-255)
        min_distance: Minimum distance between detected peaks
        
    Returns:
        Dictionary with peak information:
        - peak_location: (x, y) coordinates of the peak
        - peak_value: Pixel value at the peak
        - distance_from_center: Distance in pixels from image center
        - direction_from_center: Direction vector from center to peak
        - relative_position: Description of where peak is relative to center
    """
    # Get non-zero pixels only
    mask = depth_image > 0
    if not np.any(mask):
        return {'peak_location': None, 'peak_value': None, 'distance_from_center': None}
    
    # Find the brightest pixel (closest depth)
    max_value = np.max(depth_image[mask])
    peak_locations = np.where(depth_image == max_value)
    
    if len(peak_locations[0]) == 0:
        return {'peak_location': None, 'peak_value': None, 'distance_from_center': None}
    
    # If multiple pixels have max value, take the centroid
    peak_y = int(np.mean(peak_locations[0]))
    peak_x = int(np.mean(peak_locations[1]))
    peak_location = (peak_x, peak_y)
    
    # Calculate distance from center
    height, width = depth_image.shape
    center_x, center_y = width // 2, height // 2
    
    distance_from_center = np.sqrt((peak_x - center_x)**2 + (peak_y - center_y)**2)
    
    # Calculate direction vector (from center to peak)
    direction_x = peak_x - center_x
    direction_y = peak_y - center_y
    direction_from_center = (direction_x, direction_y)
    
    # Create relative position description
    relative_position = "center"
    if distance_from_center > 10:  # Only describe if significantly off-center
        if abs(direction_x) > abs(direction_y):
            relative_position = "right" if direction_x > 0 else "left"
        else:
            relative_position = "bottom" if direction_y > 0 else "top"
        
        # Add secondary direction if significant
        if abs(direction_x) > 5 and abs(direction_y) > 5:
            secondary = "bottom" if direction_y > 0 else "top"
            if direction_x > 0:
                relative_position += "-right"
            else:
                relative_position += "-left"
    
    return {
        'peak_location': peak_location,
        'peak_value': max_value,
        'distance_from_center': distance_from_center,
        'direction_from_center': direction_from_center,
        'relative_position': relative_position,
        'center_location': (center_x, center_y)
    }


def log_peak_analysis(depth_image: np.ndarray, frame_number: Optional[int] = None) -> dict:
    """
    Analyze and log peak location information for debugging.
    
    Args:
        depth_image: Processed depth image
        frame_number: Optional frame number for logging
        
    Returns:
        Peak analysis results dictionary
    """
    peak_info = find_depth_peak(depth_image)
    
    frame_prefix = f"Frame {frame_number}: " if frame_number is not None else ""
    
    if peak_info['peak_location'] is not None:
        peak_x, peak_y = peak_info['peak_location']
        center_x, center_y = peak_info['center_location']
        distance = peak_info['distance_from_center']
        
        logger.info(f"{frame_prefix}Peak Analysis:")
        logger.info(f"  Peak location: ({peak_x}, {peak_y}), value: {peak_info['peak_value']}")
        logger.info(f"  Image center: ({center_x}, {center_y})")
        logger.info(f"  Distance from center: {distance:.1f} pixels")
        logger.info(f"  Direction: {peak_info['relative_position']}")
        
        if distance > 20:
            direction_x, direction_y = peak_info['direction_from_center']
            logger.warning(f"  ⚠ Peak is significantly off-center!")
            logger.warning(f"    Move sand peak {abs(direction_x):.0f}px left" if direction_x < 0 else f"    Move sand peak {direction_x:.0f}px right" if direction_x > 0 else "")
            logger.warning(f"    Move sand peak {abs(direction_y):.0f}px up" if direction_y < 0 else f"    Move sand peak {direction_y:.0f}px down" if direction_y > 0 else "")
        elif distance > 10:
            logger.info(f"  ℹ Peak is slightly off-center ({peak_info['relative_position']})")
        else:
            logger.info(f"  ✓ Peak is well-centered")
    else:
        logger.warning(f"{frame_prefix}No peak found in depth image")
    
    return peak_info


def debug_raw_depth_values(depth_image: np.ndarray, frame_number: Optional[int] = None) -> dict:
    """
    Debug function to check if we're getting raw depth values or colorized values.
    This helps identify if the issue is in depth sensing or colorization.
    
    Args:
        depth_image: The depth image from the camera
        frame_number: Optional frame number for logging
        
    Returns:
        Dictionary with raw depth analysis
    """
    frame_prefix = f"Frame {frame_number}: " if frame_number is not None else ""
    
    # Check image properties
    logger.info(f"{frame_prefix}Raw Depth Debug:")
    logger.info(f"  Image dtype: {depth_image.dtype}")
    logger.info(f"  Image shape: {depth_image.shape}")
    logger.info(f"  Value range: {depth_image.min()} - {depth_image.max()}")
    
    # Sample center region
    height, width = depth_image.shape
    center_y, center_x = height // 2, width // 2
    center_region_size = 10
    
    y1 = max(0, center_y - center_region_size // 2)
    y2 = min(height, center_y + center_region_size // 2)
    x1 = max(0, center_x - center_region_size // 2)
    x2 = min(width, center_x + center_region_size // 2)
    
    center_region = depth_image[y1:y2, x1:x2]
    center_non_zero = center_region[center_region > 0]
    
    if len(center_non_zero) > 0:
        center_mean = float(np.mean(center_non_zero))
        center_min = float(np.min(center_non_zero))
        center_max = float(np.max(center_non_zero))
        
        logger.info(f"  Center region ({center_region_size}x{center_region_size}) raw values:")
        logger.info(f"    Min: {center_min}")
        logger.info(f"    Max: {center_max}")
        logger.info(f"    Mean: {center_mean:.1f}")
        
        # Check if these look like raw depth values (typically 0-65535 for 16-bit)
        # or colorized values (0-255 for 8-bit)
        if depth_image.max() <= 255:
            logger.warning("  ⚠ This appears to be COLORIZED depth data (0-255 range)")
            logger.warning("    We may not be getting raw depth values!")
        else:
            logger.info("  ✓ This appears to be RAW depth data (>255 range)")
            
            # If we have raw depth values, convert to actual distances
            # RealSense typically uses millimeters in raw data
            if center_mean > 400:  # If values are in mm range
                actual_distance_mm = center_mean
                actual_distance_m = actual_distance_mm / 1000.0
                logger.info(f"    Estimated center distance: {actual_distance_m:.3f}m ({actual_distance_mm:.0f}mm)")
                
                # Compare with manual measurement
                manual_measurement = 0.455  # User's measurement
                difference_mm = abs(actual_distance_m - manual_measurement) * 1000
                logger.info(f"    Difference from manual measurement: {difference_mm:.0f}mm")
                
                if difference_mm > 10:  # More than 1cm difference
                    logger.warning(f"  ⚠ Significant difference from manual measurement!")
    
    # Check for invalid pixels
    zero_pixels = np.sum(depth_image == 0)
    total_pixels = depth_image.size
    invalid_ratio = zero_pixels / total_pixels
    
    logger.info(f"  Invalid pixels: {zero_pixels}/{total_pixels} ({invalid_ratio:.1%})")
    
    if invalid_ratio > 0.1:  # More than 10% invalid
        logger.warning(f"  ⚠ High number of invalid depth readings!")
        logger.warning(f"    This suggests depth sensing issues with the surface material")
    
    return {
        'dtype': str(depth_image.dtype),
        'range': (float(depth_image.min()), float(depth_image.max())),
        'center_values': center_non_zero.tolist() if len(center_non_zero) > 0 else [],
        'invalid_ratio': invalid_ratio,
        'appears_raw': depth_image.max() > 255
    }

