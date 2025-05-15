import os
import heapq
import time
import math

import numpy as np
import cv2
from ffmpegcv import VideoWriter
#from numba import njit, prange
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from skimage.morphology import flood_fill
from skimage import color
from sklearn.cluster import KMeans
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator

def create_gradient_map(image, reference_point):
    """Create a gradient map based on pixel similarity from a reference point."""
    h, w = image.shape[:2]
    ref_value = image[reference_point[1], reference_point[0]]
    
    diff_map = np.abs(image.astype(np.float32) - ref_value.astype(np.float32))
    gradient_map = np.sum(diff_map, axis=2)
    gradient_map = cv2.normalize(gradient_map, None, 0, 1, cv2.NORM_MINMAX)
    return gradient_map


def posterize_image(image, num_colors):
    """Posterize the image to reduce the number of colors."""
    if num_colors <= 0 or num_colors > 128: # if the steps is large then don't bother posterizing
        return image

    # Reshape the image to a 2D array of Lab values
    pixels = image.reshape(-1, 3)
    
    # Apply k-means clustering to quantize the colors
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), num_colors, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
                                    10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Replace each pixel value with its corresponding center value
    quantized_pixels = centers[labels.flatten()]
    posterized_image = quantized_pixels.reshape(image.shape).astype(np.uint8)
    
    return posterized_image


def posterize_image_fast(image, num_colors):
    """Posterize the image to reduce the number of colors."""
    if num_colors <= 0 or num_colors > 128:  # if the steps is large then don't bother posterizing
        return image
    
    # check total number of pixel color in image is enough
    if len(np.unique(image)) < num_colors:
        return image

    # Reshape the image to a 2D array of Lab values
    pixels = image.reshape(-1, 3)
    
    # Apply k-means clustering to quantize the colors
    kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # Replace each pixel value with its corresponding center value
    quantized_pixels = centers[labels].reshape(image.shape)
    
    return quantized_pixels.astype(np.uint8)


def posterize_gradient_map(gradient_map, num_steps):
    """Posterize the gradient map according to the number of steps."""
    if num_steps <= 0 or num_steps > 128: # if the steps is large then don't bother posterizing
        return gradient_map
    step_size = 1.0 / num_steps
    posterized_map = np.floor(gradient_map / step_size) * step_size
    posterized_map = np.clip(posterized_map, 0, 1)  # Ensure posterized_map is within [0, 1]
    return posterized_map

# NOTE: currently much slower than np.linalg.norm but getting error: 
#       NumbaWarning: The TBB threading layer requires TBB version 2021 update 6 or later i.e., TBB_INTERFACE_VERSION >= 12060. Found TBB_INTERFACE_VERSION = 12050. The TBB threading layer is disabled.
# @njit(parallel=True)
# def calculate_color_distance(image_lab, y1, x1, y2, x2):
#     return np.sqrt(np.sum((image_lab[y1, x1] - image_lab[y2, x2]) ** 2))

# @njit(parallel=True)
# def calculate_color_distance_fast(image_lab, y1, x1, y2, x2):
#     return np.sum((image_lab[y1, x1] - image_lab[y2, x2]) ** 2)

def flood_fill_color_image(image, start_point):
    """Create a gradient map using flood-fill starting from a given point."""
    h, w = image.shape[:2]
    gradient_map = np.full((h, w), np.inf)
    
    if isinstance(start_point, list):
        heap = [(0, p) for p in start_point]
    else:
        heap = [(0, start_point)]
    
    while heap:
        dist, (y, x) = heapq.heappop(heap)
        if gradient_map[y, x] <= dist:
            continue
        gradient_map[y, x] = dist
        
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                diff = image[ny, nx] - image[y, x]
                color_distance = np.dot(diff, diff)
                next_dist = np.clip(dist, 0, np.inf) + np.clip(color_distance, 0, np.inf)
                if next_dist < gradient_map[ny, nx]:
                    heapq.heappush(heap, (next_dist, (ny, nx)))
    
    max_dist = np.max(gradient_map[np.isfinite(gradient_map)])
    gradient_map[gradient_map == np.inf] = max_dist
    gradient_map = cv2.normalize(gradient_map, None, 0, 1, cv2.NORM_MINMAX)
    return gradient_map


def flood_fill_gradient_map(original_map, start_point):
    """Create a gradient map (from another gradient map) using flood-fill starting from a given point."""
    h, w = original_map.shape[:2]
    gradient_map = np.full((h, w), np.inf)
    if isinstance(start_point, list):
        heap = [(0, p) for p in start_point]
    else:
        heap = [(0, start_point)]
    
    while heap:
        dist, (y, x) = heapq.heappop(heap)
        if gradient_map[y, x] <= dist:
            continue
        gradient_map[y, x] = dist
        
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                color_distance = np.linalg.norm(original_map[ny, nx] - original_map[y, x])
                next_dist = dist + color_distance
                if next_dist < gradient_map[ny, nx]:
                    heapq.heappush(heap, (next_dist, (ny, nx)))
    
    max_dist = np.max(gradient_map[np.isfinite(gradient_map)])
    gradient_map[gradient_map == np.inf] = max_dist
    gradient_map = cv2.normalize(gradient_map, None, 0, 1, cv2.NORM_MINMAX)
    return gradient_map


def make_circle_gradient(width:int, height:int, origin_x:int, origin_y:int):
    # fill the entire image with a smooth circle gradient starting from the origin
    # black at center white at point furthest from center

    # Generate (x,y) coordinate arrays
    y,x = np.mgrid[0:height,0:width]
    # Calculate the weight for each pixel
    weights = np.sqrt((x - origin_x)**2 + (y - origin_y)**2)
    weights = weights / np.max(weights)
    return weights


def combine_gradient_maps(map1, map2, weight1=1.0, weight2=2.0, 
                          min_map_weight=0.0, distance_weight=0.0, location=None,
                          equalizer=None):
    """Combine two gradient maps using given weights."""
    # combine maps with weights
    maps = [map1, map2]
    weights = [weight1, weight2]
    if min_map_weight > 0.0:
        min_map = np.minimum(map1, map2)
        maps.append(min_map)
        weights.append(min_map_weight)
    
    # NOTE: must be after all other weights are added
    if distance_weight > 0.0 and location is not None:
        # create circular gradient starting from location
        if isinstance(location, list):
            # combine gradients from multiple locations
            # recompute all weights to maintain the same ration, despite having more than one distance_weight
            # update other weights to maintain original ratio
            weights = [w * len(location) for w in weights]
            for loc in location:
                circle_gradient = make_circle_gradient(map1.shape[1], map1.shape[0], loc[1], loc[0])
                maps.append(circle_gradient)
                weights.append(distance_weight)
        else:
            circle_gradient = make_circle_gradient(map1.shape[1], map1.shape[0], location[1], location[0])
            maps.append(circle_gradient)
            weights.append(distance_weight)
    
    
    combined_map = np.average(maps, axis=0, weights=weights)

    # adaptive histogram equalization
    if equalizer is not None:
        combined_map = equalizer.apply((combined_map * 255).astype(np.uint8)) / 255.0

    # normalize the combined map
    combined_map = cv2.normalize(combined_map, None, 0, 1, cv2.NORM_MINMAX)
    return combined_map


def smooth_gradient_map(gradient_map, ksize=15):
    """Apply Gaussian blur to the gradient map to smooth it."""
    if ksize <= 0:
        return gradient_map
    smoothed_map = cv2.GaussianBlur(gradient_map, (ksize, ksize), 0)
    smoothed_map = cv2.normalize(smoothed_map, None, 0, 1, cv2.NORM_MINMAX)
    return smoothed_map


def monotonic_spline(x, x1=0.3, x2=0.7, slope_start=2.0, slope_mid=0.5, slope_end=3.0):
    """
    Creates a strictly-increasing function f:[0,1]->[0,1] whose derivative
    is at least slope_mid > 0 in the center region and slope_high > 0 near 0 and 1.
    
    We specify the derivative at four 'knots':
        x=0    -> slope_start
        x=x1   -> slope_mid
        x=x2   -> slope_mid
        x=1    -> slope_end
    
    Then we use PchipInterpolator to get a shape-preserving piecewise cubic
    for f'(x). We integrate that derivative to get f(x), and finally we scale
    so that f(0)=0, f(1)=1.
    
    Parameters
    ----------
    x : float
        Input in [0,1].
    x1, x2 : float
        The region [x1, x2] will have lower slope (slope_mid),
        and outside that region the slope transitions to slope_high.
        Must have 0 < x1 < x2 < 1.
    slope_start : float
        Positive slope used near x=0 and x=1.
    slope_mid : float
        Positive slope used in the middle region.
    slope_end : float
        Positive slope used near x=1.
    
    Returns
    -------
    float
        f(x), a strictly increasing function from 0 at x=0 to 1 at x=1.
    """
    # 1) The x-values (knots) where we define the derivative:
    knots = np.array([0.0, x1, x2, 1.0])
    
    # 2) The derivative we want at those knots:
    #    slope_high at x=0, slope_mid at x1, slope_mid at x2, slope_high at x=1.
    d_knots = np.array([slope_start, slope_mid, slope_mid, slope_end])
    
    # Make sure they are all strictly > 0 so the derivative can't go zero/negative.
    if any(val <= 0 for val in d_knots):
        raise ValueError("All knot slopes must be strictly positive.")
    
    # 3) Use PchipInterpolator to create a shape-preserving spline for the derivative.
    #    PCHIP ensures no new minima/maxima beyond your supplied data points.
    d_interp = PchipInterpolator(knots, d_knots)
    
    # 4) Integrate that spline from 0 to x to get a raw function F(x).
    F = d_interp.antiderivative()
    
    # 5) Shift so that F(0) = 0:
    offset = F(0.0)
    
    # 6) Scale so that the final function f(1)=1 exactly.
    raw_end = F(1.0) - offset
    scale = 1.0 / raw_end
    
    # 7) The final function:
    return scale * (F(x) - offset)

def flood_fill_transition(image1, image2, combined_map, steps, blur_amount=15, blend_steps=0, fast=True):
    """Transition between two images using the combined gradient map and blur edges."""
    
    assert image1.shape == image2.shape, "Images must have the same shape"
    assert blur_amount == 0 or blur_amount % 2 == 1, "Blur amount must be an odd number"
    
    t = time.monotonic()

    h, w, _ = image1.shape
    min_val = combined_map.min()
    max_val = combined_map.max()
    #print("Image size:", h, w, "min:", min_val, "max:", max_val, "map size:", combined_map.shape)

    if not fast or blend_steps <= 0:
        # rescale the combined map to match the image sizes
        if combined_map.shape[0] != h or combined_map.shape[1] != w:
            combined_map = cv2.resize(combined_map, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create a list of masks for blending
    # With with empty masks
    masks = [ np.zeros(combined_map.shape, dtype=np.float32) for _ in range(blend_steps+1) ]
    for step in range(1, steps):
        if step >= steps - blend_steps:
            alpha = 1.0
            # no mask for the last blend_steps steps
            mask = np.ones(combined_map.shape, dtype=np.float32)
        else:
            #alpha = step / max(2, steps - blend_steps)
            # de-emphasis the start and end of the gradient map (strech the middle)
            # i.e. take larger alpha steps at start and end and larger steps in the middle

            # Normalize step to [0,1]
            t = step / (steps - blend_steps)
            alpha = monotonic_spline(t, 0.1, 0.65, 2.0, 0.5, 3.0)
            threshold = alpha * (max_val - min_val) + min_val

            mask = combined_map < threshold
            #cv2.imwrite(f'transition/mask_{step:03d}_a_{alpha:0.2f}_thres_{threshold:0.2f}.jpg', mask * 255)
            
        print(f"{step}/{steps} alpha={alpha:0.3f} (blend_steps={blend_steps})", end='\r')
        
        if blur_amount == 0 and blend_steps == 0:
            blended_image = np.where(mask[..., None], image2, image1)
        elif blend_steps == 0: # blur set
            mask = mask.astype(np.float32)
            mask = cv2.GaussianBlur(mask, (blur_amount, blur_amount), 0)
            blended_image = (image1 * (1.0 - mask[..., np.newaxis]) + image2 * mask[..., np.newaxis]).astype(np.uint8)
        else:
            mask = mask.astype(np.float32)

            # add new mask to end and remove the first mask
            masks.append(mask)
            oldest_mask = masks.pop(0)

            # use the inverted gradient map as an alpha mask,
            # find the part of the image between the oldest and newest mask
            # and normalize that section to [0, 1]
            
            # remove old mask from new
            new_mask_section = mask - oldest_mask
            # invert and convert to float32
            masked_map = (1.0 - combined_map) * new_mask_section
            # normalize
            norm_masked = cv2.normalize(masked_map, None, 0, 1, cv2.NORM_MINMAX, mask=new_mask_section.astype(np.uint8))
            #cv2.imwrite(f'transition/norm_masked_map_{step:03d}.jpg', norm_masked * 255)
            # remove everything outside the mask
            norm_masked = norm_masked * new_mask_section
            # combine with 100% of oldest mask
            masked_map = oldest_mask + norm_masked #np.maximum(norm_masked, oldest_mask)
            if blur_amount > 0:
                masked_map = cv2.GaussianBlur(masked_map, (blur_amount, blur_amount), 0)
            masked_map = np.clip(masked_map, 0, 1)
            #cv2.imwrite(f'transition/masked_map_final_{step:03d}.jpg', masked_map * 255)

            if fast and combined_map.shape[0] != h or combined_map.shape[1] != w:
                masked_map = cv2.resize(masked_map, (w, h), interpolation=cv2.INTER_LANCZOS4)
            blended_image = (image1 * (1.0 - masked_map[..., np.newaxis]) + image2 * masked_map[..., np.newaxis]).astype(np.uint8)

        yield step, steps, blended_image
    
    dur = time.monotonic() - t
    print("Transition time:", time.monotonic() - t, steps, "steps, avg:", dur / steps, "fps:", steps / dur)

class ProcessedImage:
    def __init__(self, image=None, poster_colors:int=16, convert_to_lab:bool=True, flow_map=None):
        # NOTE: conversion to lab is only done on resized image
        self.set_image(image) # sets instance variables to none
        self.poster_colors = poster_colors
        self.convert_to_lab = convert_to_lab

  
    def __repr__(self):
        return f"Processed image: {self.image.shape}"
    
    def __str__(self):
        return f"Processed image: {self.image.shape}"   
    
    def copy(self, pimg):
        self.image = pimg.image
        self.resized_image = pimg.resized_image
        self.posterized_image = pimg.posterized_image
        self.flow = pimg.flow
        self.location = pimg.location
        
    def set_image(self, image, flowmap=None):
        self.image = image
        self.resized_image = None
        self.posterized_image = None
        self._load_flowmap(flowmap)

    def _load_flowmap(self, flow_map):
        self.location = None
        self.flow = None

        if flow_map is not None:
            if isinstance(flow_map, str):
                print("Loading flow map from file:", flow_map)
                if os.path.exists(flow_map):
                    self.flow = cv2.imread(flow_map, cv2.IMREAD_GRAYSCALE) / 255.0
                else:
                    print("Flow map file does not exist:", flow_map)
                    self.flow = None
            else:
                self.flow = flow_map
            
            if self.flow is not None:
                # find starting location from flow map (start value = 0,0,0)
                self.location = np.unravel_index(np.argmin(self.flow), self.flow.shape)
                # convert to ints
                self.location = (int(self.location[0]), int(self.location[1]))
                print("Flow map autodetect starting location:", self.location)


    def process(self, image=None, size:tuple=None, poster_colors:int=0, location=None):
        if image is not None:
            self.set_image(image)
        if self.flow is None:
            self.resize(size)
            self.posterize(poster_colors)
            self.flow_map(location)

    def resize(self, size:tuple=None):
        if self.resized_image is not None:
            return
        
        if size is None or size[0] == 0 or size[1] == 0 or size == self.image.shape[:2]: # no resizing
            self.resized_image = self.image
        else:
            #print("Resizing image to:", size)
            # note, resize() takes (w,h) as input
            self.resized_image = cv2.resize(self.image, (size[1], size[0]), interpolation=cv2.INTER_LANCZOS4)
        if self.convert_to_lab:
            #print("Converting image to Lab")
            self.resized_image = cv2.cvtColor(self.resized_image, cv2.COLOR_BGR2Lab)

    def posterize(self, poster_colors:int=0):
        if self.resized_image is None:
            raise ValueError("Image must be resized before posterizing")
        if self.posterized_image is not None:
            return
        if poster_colors <= 0:
            poster_colors = self.poster_colors
        poster_colors = int(poster_colors)
        #print("Posterizing image to:", poster_colors)
        if self.poster_colors <= 0:
            self.posterized_image = self.resized_image
        else:
            self.posterized_image = posterize_image_fast(self.resized_image, poster_colors)

    def flow_map(self, location):
        if self.posterized_image is None:
            raise ValueError("Image must be posterized before creating gradient map")
        if self.flow is not None:
            return
        if location is None and self.location is not None:
            location = self.location
        #print("Creating flow map")
        t = time.monotonic()
        self.flow = flood_fill_color_image(self.posterized_image, location)
        print("Flow map time:", time.monotonic() - t)
    
    def size(self) -> tuple:
        if self.resized_image is not None:
            return self.resized_image.shape[:2]
        return self.image.shape[:2]

    def save_flowmap(self, filename:str):
        if self.flow is not None:
            print("Saving flow map to:", filename)
            cv2.imwrite(filename, self.flow * 255)

    def save_debug_images(self, prefix:str):
        cv2.imwrite(f'{prefix}_image.jpg', self.image)
        if self.resized_image is not None:
            if self.convert_to_lab:
                cv2.imwrite(f'{prefix}_resized_image.jpg', cv2.cvtColor(self.resized_image, cv2.COLOR_Lab2BGR))
            else:
                cv2.imwrite(f'{prefix}_resized_image.jpg', self.resized_image)
        if self.posterized_image is not None:
            if self.convert_to_lab:
                cv2.imwrite(f'{prefix}_posterized_image.jpg', cv2.cvtColor(self.posterized_image, cv2.COLOR_Lab2BGR))
            else:
                cv2.imwrite(f'{prefix}_posterized_image.jpg', self.posterized_image)
        if self.flow is not None:
            cv2.imwrite(f'{prefix}_flow_map.jpg', self.flow * 255)


class SimilarityTransition:
    def __init__(self, steps=30, blur_amount=9, blend_steps=3, 
                 posterization_range=(4, 20), size=0, 
                 weight1=1.0, weight2=1.0, # weights for mixing gradient maps
                 distance_weight=1.0, # weight for distance from start location
                 ksize:int=7, # kernel size for smoothing gradient maps
                 ):
        self.steps = steps
        self.blur_amount = blur_amount
        self.blend_steps = blend_steps
        self.weight1 = weight1
        self.weight2 = weight2
        self.distance_weight = distance_weight
        self.posterization_range = posterization_range
        self.size = size
        self.ksize = ksize
        assert(ksize <= 0 or ksize % 2 == 1), "ksize must be an odd number"

        # internal variables
        self.transition_count = 0
        self.scale = 1.0
        if posterization_range is None:
            self.poster_colors = 0
        elif isinstance(posterization_range, int):
            if posterization_range <= 0:
                self.poster_colors = 0
            else:
                self.poster_colors = posterization_range
        else:
            self.poster_colors = int(min(posterization_range[1], max(posterization_range[0], self.steps // max(self.blend_steps, 2))))
        self.pimg1 = ProcessedImage(poster_colors=self.poster_colors)
        self.pimg2 = ProcessedImage(poster_colors=self.poster_colors)
        
        # init adaptive histogram equalizer
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


    def __repr__(self):
        return f"Similarity transition: {self.steps} steps, {self.blur_amount} blur, {self.blend_steps} blend steps, weights: {self.weight1}, {self.weight2}, distance weight: {self.distance_weight}, posterization range: {self.posterization_range}, size: {self.size}, ksize: {self.ksize}"

    def __str__(self):
        return f"Similarity transition: {self.steps} steps, {self.blur_amount} blur, {self.blend_steps} blend steps, weights: {self.weight1}, {self.weight2}, distance weight: {self.distance_weight}, posterization range: {self.posterization_range}, size: {self.size}, ksize: {self.ksize}"

    def __len__(self):
        return self.transition_count

    def _get_scale_size(self, size:int, image):
        scale = 1.0
        if size <= 0:
            return scale, image.shape[:2]
        # finding a largest dimension and resizing it to size
        if image.shape[0] != size or image.shape[1] != size:
            if image.shape[0] > image.shape[1]:
                scale = size / image.shape[0]
            else:
                scale = size / image.shape[1]
            return scale, (int(image.shape[0] * scale), int(image.shape[1] * scale))
        else:
            return scale, image.shape[:2]


    def _similarity(self):
        if self.pimg1 is None or self.pimg2 is None:
            return None
        # find area of max similarity
        _, ssim_map = ssim(self.pimg1.resized_image, self.pimg2.resized_image, multichannel=True, full=True, channel_axis=-1)
        max_similarity_loc = np.unravel_index(np.argmax(ssim_map), ssim_map.shape)
        # remove third element if it exists
        if len(max_similarity_loc) == 3:
            max_similarity_loc = max_similarity_loc[:2]
        # convert to tuple of ints
        return (int(max_similarity_loc[0]), int(max_similarity_loc[1]))


    def _combine_maps(self, map1, map2, weight1:float=1.0, weight2:float=1.0, 
                      distance_weight:float=0.0, location:tuple=None):
        combined_map = combine_gradient_maps(map1, map2, weight1, weight2, 
                                             equalizer=self.clahe, 
                                             distance_weight=distance_weight, 
                                             location=location)
        #cv2.imwrite(f'transition/combined_map_{time.time()}.jpg', combined_map * 255)
        return smooth_gradient_map(combined_map, ksize=self.ksize)


    # size=(h, w), location=(h, w)|[(h,w),...]
    def _get_location(self, size:tuple, location=None): 
        h = 0
        w = 1
        if location is None:
            location = self._similarity()
            if location is None:
                return None, None
        else:
            if isinstance(location, list):
                for i, loc in enumerate(location):
                    if isinstance(location[h], float) or isinstance(location[w], float):
                        # assume it is a percentage
                        x = min(max(int(loc[w] * size[w]), 0), size[w] - 1)
                        y = min(max(int(loc[h] * size[h]), 0), size[h] - 1)
                        location[i] = (y, x)
                        print("Location:", loc, "in image:", size)
                        print(location)
            else: # single location (tuple)
                if isinstance(location[0], float) or isinstance(location[1], float):
                    # assume it is a percentage
                    x = min(max(int(location[w] * size[w]), 0), size[w] - 1)
                    y = min(max(int(location[h] * size[h]), 0), size[h] - 1)
                    location = (y, x)
                else: # assume it is a pixel location in the original image
                    location = (min(max(location[h], 0), size[h] - 1),
                                min(max(location[w], 0), size[w] - 1))

        print("Location:", location, "in (scaled) image:", size)
        return location


    # size is the maximum size for temporary image data used for calculations, 0 = no resizing
    def transition(self, imageA, imageB, steps:int=0, size:int=0, 
                   weight1:float=-1.0, weight2:float=-1.0, location:tuple=None,  # location is (h,w)
                   flowmapA=None, flowmapB=None):
        if weight1 < 0.0:
            weight1 = self.weight1
        if weight2 < 0.0:
            weight2 = self.weight2
        if steps <= 0:
            steps = self.steps
        
        # if only one image provided assume we are transitioning from the current image
        if imageA is None:
            return self.transition_to(imageB, weight1, weight2, location, flowmap=flowmapB)
        elif imageB is None:
            return self.transition_to(imageA, weight1, weight2, location, flowmap=flowmapA)

        assert imageA.shape == imageB.shape, "Images must have the same shape"
        #print("Transitioning from imageA to imageB")
        self.pimg1.set_image(imageA, flowmap=flowmapA)
        self.pimg2.set_image(imageB, flowmap=flowmapB)
        if size <= 0:
            size = self.size
        self.scale, size_tuple = self._get_scale_size(size, imageA)
        print("Size: ", size, "imageA.shape:", imageA.shape, "size tuple:", size_tuple, "scale:", self.scale)
        # images must be resized first
        self.pimg1.resize(size_tuple)
        self.pimg2.resize(size_tuple)
        # set the location for the flood fill
        scaled_location = self._get_location(size_tuple, location)
        # finish processing the images
        self.pimg1.process(location=scaled_location)
        self.pimg2.process(location=scaled_location)

        # save flowmaps if they are newly created
        if flowmapA is not None:
            if isinstance(flowmapA, str):
                if not os.path.exists(flowmapA):
                    self.pimg1.save_flowmap(flowmapA)
        if flowmapB is not None:
            if isinstance(flowmapB, str):
                if not os.path.exists(flowmapB):
                    self.pimg2.save_flowmap(flowmapB)

        #self.pimg1.save_debug_images('transition/image1')
        #self.pimg2.save_debug_images('transition/image2')
        transition_map = self._combine_maps(self.pimg1.flow, self.pimg2.flow, 
                                          weight1, weight2,
                                          distance_weight=self.distance_weight, 
                                          location=scaled_location)
        self.transition_count += 1
        #cv2.imwrite('transition/transition_map.jpg', transition_map * 255)
        return flood_fill_transition(self.pimg1.image, self.pimg2.image, transition_map, 
                steps=steps, 
                blur_amount=self.blur_amount, 
                blend_steps=self.blend_steps)
        
    # move image2 to image1 and transition to new image
    def transition_to(self, image, steps:int=0,
                      weight1:float=-1.0, weight2:float=-1.0, location:tuple=None,
                      flowmap=None):
        if weight1 < 0.0:
            weight1 = self.weight1
        if weight2 < 0.0:
            weight2 = self.weight2
        if steps <= 0:
            steps = self.steps

        # shift image2 to image1
        self.pimg1.copy(self.pimg2)
        assert self.pimg1.image.shape == image.shape, f"Images must have the same shape: {self.pimg1.image.shape} vs {image.shape}"
        self.pimg2.set_image(image, flowmap=flowmap)
        self.pimg2.resize(self.pimg1.size())
        # set the location for the flood fill
        scaled_location = self._get_location(self.pimg1.size(), location)
        self.pimg2.process(location=scaled_location)

        # save flowmaps if they are newly created
        if flowmap is not None:
            if isinstance(flowmap, str):
                if not os.path.exists(flowmap):
                    self.pimg1.save_flowmap(flowmap)

        #self.pimg1.save_debug_images('transition/image_to1')
        #self.pimg2.save_debug_images('transition/image_to2')
        transition_map = self._combine_maps(self.pimg1.flow, self.pimg2.flow, 
                                          weight1, weight2,
                                          distance_weight=self.distance_weight, 
                                          location=scaled_location)
        #cv2.imwrite('transition/transition_to_map.jpg', transition_map * 255)
        self.transition_count += 1
        return flood_fill_transition(self.pimg1.image, self.pimg2.image, transition_map, 
            steps=steps, 
            blur_amount=self.blur_amount, 
            blend_steps=self.blend_steps)


if __name__ == "__main__":
    import os
    dir_path = 'images/transition/exp_test6/'
    image_files = os.listdir(dir_path)
    image_files = sorted(image_files)
    image_files.reverse()
    #dir_path = ''
    # read image files from text list (remove end of line)
    #with open('processed_filelist.txt') as f:
    #   image_files = f.readlines()
    image_files = [x.strip() for x in image_files]
    #image_files = image_files[2:] # limit to 10 images
    print(image_files)
    total_images = len(image_files)
    loop = True
    if loop:
        total_images += 1
    image1 = cv2.imread(dir_path + image_files.pop(0))
    image2 = cv2.imread(dir_path + image_files.pop(0))
    fps = 24
    #total_time = 10.0 # seconds
    #total_steps = int(total_time * fps)
    #steps = total_steps // total_images
    steps = 3 * fps
    size = 1024 #int(image1.shape[0] / 2)
    stills = int(0.1 * fps)
    location = None # (1.0, 0.5) #None #(0, 0.5) # h, w
    #location = [(0.,0.),(0.,1.0),(1.0,0.0),(1.0,1.0)]#(0.5, 0.5) # None for similarity match
    distance_weight = 0.1
    
    st = SimilarityTransition(steps=steps, size=size, 
                              blur_amount=5, blend_steps=10, 
                              weight2=1.0, distance_weight=distance_weight,
                              ksize=5)
    i = 1

    def add_still_images(img, count, writer):
        for _ in range(count):
            writer.write(img)

    filename = f"images/transition/exp_test6/transition_{steps}_{size}_{time.time()}.mp4"

    with VideoWriter(filename, codec='libx264', fps=fps, preset="slow") as writer:

        add_still_images(image1, stills, writer)

        i += 1
        for step, _, frame in st.transition(image1, image2, location=location):
            writer.write(frame)
        i += 1
        add_still_images(image2, stills, writer)

        while len(image_files) > 0:
            i += 1
            image = cv2.imread(dir_path + image_files.pop(0))
            for step, _, frame in st.transition_to(image, location=location):
                writer.write(frame)
            i += 1
            add_still_images(image, stills, writer)
        
        if loop:
            # # loop back to first image
            i += 1
            for step, _, frame in st.transition_to(image1, location=location):
                writer.write(frame)
