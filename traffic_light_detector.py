#!/usr/bin/env python3
"""
Automatic Traffic Light Detection (Clean & Simple)

Color-based HSV detection for Red/Yellow/Green traffic lights with
lightweight preprocessing and straightforward scoring.
"""

import cv2
import numpy as np
import argparse
from datetime import datetime


class AutoTrafficLightDetector:

    def __init__(self):
        # HSV color ranges for traffic lights - more inclusive ranges
        self.color_ranges = {
            'Red': [
                {
                    'lower': np.array([0, 20, 100]),
                    'upper': np.array([10, 255, 255])
                },
                {
                    'lower': np.array([170, 20, 100]),
                    'upper': np.array([180, 255, 255])
                },
            ],
            'Yellow': [
                {
                    'lower': np.array([20, 20, 100]),
                    'upper': np.array([40, 255, 255])
                },
            ],
            'Green': [
                {
                    'lower': np.array([45, 20, 100]),
                    'upper': np.array([75, 255, 255])
                },
            ],
        }

        # Contour filtering - will be set dynamically based on image size
        self.min_contour_area = 80
        self.max_contour_area = 15000
        # Dynamic area bounds as a fraction of frame area (used in addition to absolute bounds)
        self.min_area_ratio = 0.0001  # 0.01% of frame (increased for better detection)
        self.max_area_ratio = 0.15  # 15% of frame (increased for better detection)
        self.min_circularity = 0.15
        self.min_color_purity = 0.05
        self.blur_kernel_size = (5, 5)  # Will be scaled dynamically
        self.confidence_threshold = 0.35
        # Illumination gates - relaxed for better detection
        self.min_mean_v = 20
        self.min_luminous_fraction = 0.02
        self.min_brightness_contrast = 1

        # Preprocessing normalization
        self.enable_gray_world = True
        self.enable_auto_gamma = True
        self.gamma_low = 0.7
        self.gamma_high = 1.6
        self.gamma_target_mean = 130.0  # desired mean brightness in [0,255]

        # Debug toggle (press 'd' in window)
        self.show_debug_masks = False

        # Tracking configuration
        self.tracks = [
        ]  # list of dicts: id, box, confidence, color_history, hits, misses, confirmed
        self.next_track_id = 1
        self.iou_match_threshold = 0.3
        self.max_missed_frames = 8
        self.required_hits_to_confirm = 3
        self.smooth_alpha = 0.6  # higher = more weight on previous
        self.conf_smooth_alpha = 0.5
        self.color_history_length = 8
        # Type classification thresholds
        self.max_track_speed_px = 15.0
        self.min_vertical_neighbor_overlap = 0.25
        self.max_aspect_ratio_vehicle_guess = 1.6
        self.min_aspect_ratio_traffic_guess = 0.7

    # -------------------- Tracking helpers --------------------
    def _compute_iou(self, box_a, box_b):
        x1, y1, w1, h1 = box_a
        x2, y2, w2, h2 = box_b
        xa0, ya0, xa1, ya1 = x1, y1, x1 + w1, y1 + h1
        xb0, yb0, xb1, yb1 = x2, y2, x2 + w2, y2 + h2
        inter_w = max(0, min(xa1, xb1) - max(xa0, xb0))
        inter_h = max(0, min(ya1, yb1) - max(ya0, yb0))
        inter = inter_w * inter_h
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0

    def _smooth_value(self, prev, new, alpha):
        return alpha * prev + (1.0 - alpha) * new

    def _majority_color(self, history):
        if not history:
            return None
        counts = {}
        for c in history:
            counts[c] = counts.get(c, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def _update_tracks(self, detections):
        # Mark all tracks as unmatched initially
        for t in self.tracks:
            t['matched'] = False

        # Greedy IoU matching
        unmatched_dets = set(range(len(detections)))
        while True:
            best_iou = 0.0
            best_pair = None
            for di in list(unmatched_dets):
                for ti, tr in enumerate(self.tracks):
                    if tr.get('matched'):
                        continue
                    iou = self._compute_iou(detections[di]['box'], tr['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_pair = (di, ti)
            if best_pair is None or best_iou < self.iou_match_threshold:
                break
            di, ti = best_pair
            det = detections[di]
            tr = self.tracks[ti]
            # Smooth box
            x, y, w, h = det['box']
            px, py, pw, ph = tr['box']
            sx = self._smooth_value(px, x, self.smooth_alpha)
            sy = self._smooth_value(py, y, self.smooth_alpha)
            sw = self._smooth_value(pw, w, self.smooth_alpha)
            sh = self._smooth_value(ph, h, self.smooth_alpha)
            tr['box'] = (int(sx), int(sy), int(sw), int(sh))
            # Smooth confidence
            tr['confidence'] = float(
                self._smooth_value(tr['confidence'], det['confidence'],
                                   self.conf_smooth_alpha))
            # Update color history
            tr['color_history'].append(det['color'])
            if len(tr['color_history']) > self.color_history_length:
                tr['color_history'].pop(0)
            tr['color'] = self._majority_color(
                tr['color_history']) or det['color']
            # Update counters
            tr['hits'] += 1
            tr['misses'] = 0
            if not tr['confirmed'] and tr[
                    'hits'] >= self.required_hits_to_confirm:
                tr['confirmed'] = True
            tr['matched'] = True
            unmatched_dets.discard(di)

        # Create new tracks for unmatched detections
        for di in list(unmatched_dets):
            det = detections[di]
            self.tracks.append({
                'id': self.next_track_id,
                'box': det['box'],
                'confidence': float(det['confidence']),
                'color_history': [det['color']],
                'color': det['color'],
                'hits': 1,
                'misses': 0,
                'confirmed': False,
                'matched': True,
            })
            self.next_track_id += 1

        # Age unmatched tracks and prune
        kept = []
        for tr in self.tracks:
            if not tr.get('matched'):
                tr['misses'] += 1
            # light decay of confidence when missed
            if tr['misses'] > 0:
                tr['confidence'] *= 0.95
            if tr['misses'] <= self.max_missed_frames:
                kept.append(tr)
        self.tracks = kept

    # -------------------- Simple TL vs Vehicle light heuristics --------------------
    def _estimate_motion(self, prev_box, new_box):
        px, py, pw, ph = prev_box
        nx, ny, nw, nh = new_box
        pcx, pcy = px + pw / 2.0, py + ph / 2.0
        ncx, ncy = nx + nw / 2.0, ny + nh / 2.0
        return ((ncx - pcx)**2 + (ncy - pcy)**2)**0.5

    def _classify_track_type(self, tr, all_tracks):
        # Default assumption
        guessed_type = 'TrafficLight'
        x, y, w, h = tr['box']
        aspect_ratio = h / w if w > 0 else 0
        # Heuristic 1: apparent motion speed (vehicle lights often move quickly across frame)
        speed = tr.get('last_speed', 0.0)
        if speed > self.max_track_speed_px:
            guessed_type = 'VehicleLight'
        # Heuristic 2: neighbor arrangement (vertical stack of similar widths suggests traffic signal)
        vertical_support = 0
        for other in all_tracks:
            if other is tr or not other.get('confirmed'):
                continue
            ox, oy, ow, oh = other['box']
            # similar x alignment and width
            width_similar = abs(ow - w) / max(w, 1) < 0.35
            x_overlap = max(0, min(x + w, ox + ow) - max(x, ox))
            x_overlap_ratio = x_overlap / max(min(w, ow), 1)
            vertical_gap = min(abs((oy + oh / 2) - (y + h / 2)), h)
            vertical_overlap_ok = x_overlap_ratio > 0.4 and vertical_gap > 0
            if width_similar and vertical_overlap_ok:
                vertical_support += 1
        if vertical_support >= 1 and aspect_ratio >= self.min_aspect_ratio_traffic_guess:
            guessed_type = 'TrafficLight'
        # Heuristic 3: aspect ratio and brightness texture
        if aspect_ratio < 0.6 or aspect_ratio > self.max_aspect_ratio_vehicle_guess:
            guessed_type = 'VehicleLight'
        return guessed_type

    # -------------------- Dynamic scaling helpers --------------------
    def _calculate_dynamic_params(self, frame):
        """Calculate dynamic parameters based on frame size for better scaling"""
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w

        # Calculate scale factor based on image size
        # Use a reference size of 640x480 (307,200 pixels) as baseline
        reference_area = 640 * 480
        scale_factor = max(1.0, (frame_area / reference_area)**0.5)

        # Scale contour areas
        min_area = max(50, int(self.min_contour_area * scale_factor))
        max_area = min(50000, int(self.max_contour_area * scale_factor))

        # Scale blur kernel size (must be odd numbers)
        blur_size = max(3, int(5 * scale_factor))
        if blur_size % 2 == 0:
            blur_size += 1
        blur_kernel_size = (blur_size, blur_size)

        # Scale morphology kernel sizes
        small_kernel_size = max(3, int(3 * scale_factor))
        if small_kernel_size % 2 == 0:
            small_kernel_size += 1
        large_kernel_size = max(5, int(5 * scale_factor))
        if large_kernel_size % 2 == 0:
            large_kernel_size += 1

        # Scale padding
        padding = max(2, int(3 * scale_factor))

        return {
            'min_area': min_area,
            'max_area': max_area,
            'blur_kernel_size': blur_kernel_size,
            'small_kernel_size': small_kernel_size,
            'large_kernel_size': large_kernel_size,
            'padding': padding,
            'scale_factor': scale_factor
        }

    # -------------------- Preprocessing helpers --------------------
    def _gray_world_white_balance(self, img_bgr):
        if not self.enable_gray_world:
            return img_bgr
        img = img_bgr.astype(np.float32)
        mean_b = np.mean(img[:, :, 0]) + 1e-6
        mean_g = np.mean(img[:, :, 1]) + 1e-6
        mean_r = np.mean(img[:, :, 2]) + 1e-6
        mean_gray = (mean_b + mean_g + mean_r) / 3.0
        scale_b = mean_gray / mean_b
        scale_g = mean_gray / mean_g
        scale_r = mean_gray / mean_r
        img[:, :, 0] *= scale_b
        img[:, :, 1] *= scale_g
        img[:, :, 2] *= scale_r
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _auto_gamma_correct(self, img_bgr):
        if not self.enable_auto_gamma:
            return img_bgr
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray)) + 1e-6
        # Estimate gamma that would map current mean to target mean under power-law
        gamma = np.log(self.gamma_target_mean / 255.0 +
                       1e-6) / np.log(mean_val / 255.0 + 1e-6)
        gamma = np.clip(gamma, self.gamma_low, self.gamma_high)
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0)**inv_gamma * 255
                          for i in range(256)]).astype("uint8")
        return cv2.LUT(img_bgr, table)

    def detect_traffic_lights(self, frame):
        """Return top detections as [{'color','box','confidence'}]."""
        detections = []
        if frame is None or frame.size == 0:
            return detections

        # Calculate dynamic parameters based on frame size
        dynamic_params = self._calculate_dynamic_params(frame)

        # Pre-processing: white-balance, gamma, blur
        pre = self._gray_world_white_balance(frame)
        pre = self._auto_gamma_correct(pre)
        blurred = cv2.GaussianBlur(pre, dynamic_params['blur_kernel_size'], 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        hsv = cv2.merge([h, s, v])

        # Brightness and saturation gates - improved for low-saturation images
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        # Use adaptive brightness threshold - much lower for better detection, especially for red lights
        bright_thresh = max(20, min(60, int(np.percentile(
            gray, 15))))  # Lower threshold for red lights
        _, bright_mask = cv2.threshold(gray, bright_thresh, 255,
                                       cv2.THRESH_BINARY)
        # Use very low saturation threshold or skip saturation filtering for low-saturation images
        if np.mean(
                s
        ) < 15:  # If image has very low saturation, skip saturation filtering
            combined_mask = bright_mask
        else:
            _, saturation_mask = cv2.threshold(
                s, 5, 255, cv2.THRESH_BINARY)  # Very low saturation threshold
            combined_mask = cv2.bitwise_and(bright_mask, saturation_mask)

        # Dynamic area bounds based on frame size
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w
        dyn_min_area = max(dynamic_params['min_area'],
                           int(frame_area * self.min_area_ratio))
        dyn_max_area = min(dynamic_params['max_area'],
                           int(frame_area * self.max_area_ratio))

        # Build per-color masks for later color voting
        color_masks_cache = {}

        # Iterate colors
        for color in ['Red', 'Yellow', 'Green']:
            ranges = self.color_ranges[color]
            color_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for r in ranges:
                mask = cv2.inRange(hsv, r['lower'], r['upper'])
                mask = cv2.bitwise_and(mask, combined_mask)
                color_mask = cv2.bitwise_or(color_mask, mask)
            color_masks_cache[color] = color_mask

            # Morphology: close -> open - improved for better connectivity
            kernel_small = np.ones((dynamic_params['small_kernel_size'],
                                    dynamic_params['small_kernel_size']),
                                   np.uint8)
            kernel_large = np.ones((dynamic_params['large_kernel_size'],
                                    dynamic_params['large_kernel_size']),
                                   np.uint8)
            # Use larger kernel for closing to better connect fragmented regions
            kernel_close = np.ones(
                (max(7, dynamic_params['large_kernel_size']),
                 max(7, dynamic_params['large_kernel_size'])), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE,
                                          kernel_close)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN,
                                          kernel_small)
            # Median blur to suppress salt-and-pepper noise
            color_mask = cv2.medianBlur(color_mask, 3)

            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                rect_area = w * h
                if rect_area < dyn_min_area or rect_area > dyn_max_area:
                    continue

                area = cv2.contourArea(contour)
                if area <= 0:
                    continue
                aspect_ratio = h / w if w > 0 else 0

                # Relax aspect ratio for red lights which might be more elongated
                if color == 'Red':
                    if aspect_ratio < 0.3 or aspect_ratio > 6.0:
                        continue
                else:
                    if aspect_ratio < 0.5 or aspect_ratio > 5.0:
                        continue

                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (
                    perimeter * perimeter) if perimeter > 0 else 0
                if circularity < self.min_circularity and aspect_ratio < 0.8:
                    continue

                roi_v = v[y:y + h, x:x + w]
                if roi_v.size == 0:
                    continue
                mean_v = float(np.mean(roi_v))
                local_thresh = max(110, np.percentile(roi_v, 70))
                lum_mask = roi_v >= local_thresh
                luminous_fraction = np.sum(lum_mask) / roi_v.size

                pad = max(dynamic_params['padding'], int(0.1 * max(w, h)))
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(frame.shape[1], x + w + pad)
                y1 = min(frame.shape[0], y + h + pad)
                if (x1 - x0) > (w + 5) and (y1 - y0) > (h + 5):
                    bg_region = v[y0:y1, x0:x1].copy()
                    roi_slice = bg_region[(y - y0):(y - y0 + h),
                                          (x - x0):(x - x0 + w)]
                    if roi_slice.size > 0:
                        roi_slice[:] = 0
                    background_mean = float(
                        np.mean(bg_region)) if bg_region.size else 0
                    brightness_contrast = mean_v - background_mean
                else:
                    brightness_contrast = mean_v

                if mean_v < self.min_mean_v:
                    continue
                if luminous_fraction < self.min_luminous_fraction:
                    continue
                if brightness_contrast < self.min_brightness_contrast:
                    continue

                color_purity = np.sum(
                    color_mask[y:y + h, x:x + w]) / (255 * w * h) if (w *
                                                                      h) else 0
                if color_purity < self.min_color_purity:
                    continue

                # Confidence: balance of geometry, purity, and brightness
                aspect_conf = 1.0 - min(abs(2.5 - aspect_ratio) / 2.5, 1.0)
                circ_conf = min(circularity / 0.9, 1.0)
                purity_conf = min(color_purity * 2.0, 1.0)
                bright_conf = min(mean_v / 160.0, 1.0)
                confidence = 0.25 * aspect_conf + 0.25 * circ_conf + 0.3 * purity_conf + 0.2 * bright_conf

                if confidence >= self.confidence_threshold:
                    detections.append({
                        'color': color,
                        'box': (x, y, w, h),
                        'confidence': float(confidence),
                    })

        # Re-vote color per detection using all color masks to avoid conflicting labels
        for d in detections:
            x, y, w, h = d['box']
            scores = {}
            for cname, cmask in color_masks_cache.items():
                region = cmask[y:y + h, x:x + w]
                scores[cname] = int(np.sum(region))
            if scores:
                # Get scores for each color
                green_score = scores.get('Green', 0)
                yellow_score = scores.get('Yellow', 0)
                red_score = scores.get('Red', 0)
                
                # Prioritize red detection (red is the most important traffic light color for safety)
                # If there's a significant red component, classify as red
                if red_score > 0 and red_score >= yellow_score * 0.6:
                    d['color'] = 'Red'
                    continue
                    
                # Bias rule: if Green is close to Yellow, prefer Yellow
                if yellow_score > 0 and green_score > 0:
                    if green_score <= int(yellow_score * 1.15):
                        d['color'] = 'Yellow'
                        continue
                
                # Hue median tie-break inside ROI
                hsv_roi = hsv[y:y + h, x:x + w]
                if hsv_roi.size > 0:
                    roi_h = hsv_roi[:, :, 0]
                    med_h = int(np.median(roi_h))
                    # More precise hue-based classification with expanded ranges
                    if 0 <= med_h <= 15 or 165 <= med_h <= 180:
                        d['color'] = 'Red'
                        continue
                    elif 22 <= med_h <= 38:
                        d['color'] = 'Yellow'
                        continue
                    elif 45 <= med_h <= 75:
                        d['color'] = 'Green'
                        continue
                
                # If we still haven't decided, use the color with highest score
                voted = max(scores.items(), key=lambda kv: kv[1])[0]
                d['color'] = voted

        # If no detections found with color-based method, try bright region detection
        if len(detections) == 0:
            detections = self._detect_bright_regions(frame, dynamic_params)

        # Non-maximum suppression (IoU) - more aggressive for better deduplication
        detections = self._remove_overlapping_boxes(detections,
                                                    overlap_threshold=0.15)
        # Remove close detections (position-based deduplication)
        detections = self._remove_close_detections(detections,
                                                   min_distance_threshold=25)
        # Additional false positive filtering
        detections = self._filter_false_positives(detections, frame)

        # Enhanced final pass: remove overlapping detections regardless of color
        final = []
        for det in detections:
            keep = True
            x, y, w, h = det['box']
            my_score = int(
                np.sum(color_masks_cache[det['color']][y:y + h, x:x + w]))

            for kept in list(final):
                kx, ky, kw, kh = kept['box']
                # IoU calculation
                x_overlap = max(0, min(x + w, kx + kw) - max(x, kx))
                y_overlap = max(0, min(y + h, ky + kh) - max(y, ky))
                inter = x_overlap * y_overlap
                if inter == 0:
                    continue
                union = w * h + kw * kh - inter
                iou = inter / union if union > 0 else 0

                # Check for overlapping detections (lower threshold for better deduplication)
                if iou > 0.25:  # Lower threshold to catch more duplicates
                    kept_score = int(
                        np.sum(color_masks_cache[kept['color']][ky:ky + kh,
                                                                kx:kx + kw]))

                    # If same color, keep the one with higher confidence
                    if det['color'] == kept['color']:
                        if det['confidence'] > kept['confidence']:
                            final.remove(kept)
                        else:
                            keep = False
                            break
                    # If different colors, keep the one with higher color score
                    else:
                        if my_score > kept_score or (my_score == kept_score
                                                     and det['confidence']
                                                     > kept['confidence']):
                            final.remove(kept)
                        else:
                            keep = False
                            break

            if keep:
                final.append(det)
        detections = final
        # Keep top detections (increased limit for better detection of multiple lights)
        detections.sort(key=lambda d: d['confidence'], reverse=True)
        return detections[:
                          10]  # Increased from 3 to 10 to allow more detections

    def _remove_overlapping_boxes(self, detections, overlap_threshold=0.3):
        """
        Remove overlapping bounding boxes, keeping the ones with higher confidence.
        
        Args:
            detections: List of detection dictionaries
            overlap_threshold: IoU threshold for considering boxes as overlapping
            
        Returns:
            filtered_detections: List of detections with overlaps removed
        """
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections,
                            key=lambda x: x['confidence'],
                            reverse=True)

        # List to hold filtered detections
        filtered_detections = []

        # Process all detections
        for detection in detections:
            should_keep = True

            # Get current box
            x1, y1, w1, h1 = detection['box']
            box1_area = w1 * h1

            # Compare with all kept detections
            for kept in filtered_detections:
                # Get kept box
                x2, y2, w2, h2 = kept['box']

                # Calculate intersection area
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                intersection_area = x_overlap * y_overlap

                # Calculate union area
                box2_area = w2 * h2
                union_area = box1_area + box2_area - intersection_area

                # Calculate IoU (Intersection over Union)
                iou = intersection_area / union_area if union_area > 0 else 0

                # If overlap exceeds threshold, don't keep this detection
                if iou > overlap_threshold:
                    should_keep = False
                    break

            # Add to filtered list if it doesn't overlap significantly with any kept detection
            if should_keep:
                filtered_detections.append(detection)

        return filtered_detections

    def _remove_close_detections(self, detections, min_distance_threshold=20):
        """
        Remove detections that are too close to each other, keeping the one with higher confidence.
        This catches cases where IoU might be low but detections are essentially the same.
        
        Args:
            detections: List of detection dictionaries
            min_distance_threshold: Minimum distance between detection centers (in pixels)
            
        Returns:
            filtered_detections: List of detections with close duplicates removed
        """
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections,
                            key=lambda x: x['confidence'],
                            reverse=True)

        filtered_detections = []

        for detection in detections:
            x, y, w, h = detection['box']
            center_x = x + w / 2
            center_y = y + h / 2

            should_keep = True

            for kept in filtered_detections:
                kx, ky, kw, kh = kept['box']
                kept_center_x = kx + kw / 2
                kept_center_y = ky + kh / 2

                # Calculate distance between centers
                distance = ((center_x - kept_center_x)**2 +
                            (center_y - kept_center_y)**2)**0.5

                # If too close and same color, remove the one with lower confidence
                if distance < min_distance_threshold and detection[
                        'color'] == kept['color']:
                    should_keep = False
                    break

            if should_keep:
                filtered_detections.append(detection)

        return filtered_detections

    def _filter_false_positives(self, detections, frame):
        """
        Filter out false positives using additional validation checks.
        
        Args:
            detections: List of detection dictionaries
            frame: Original frame for context
            
        Returns:
            filtered_detections: List of detections with false positives removed
        """
        if not detections:
            return []

        filtered_detections = []
        frame_height, frame_width = frame.shape[:2]

        for detection in detections:
            x, y, w, h = detection['box']

            # Skip detections too close to edges (likely partial detections)
            if x < 10 or y < 10 or x + w > frame_width - 10 or y + h > frame_height - 10:
                continue

            # Skip very small detections relative to frame size
            frame_area = frame_width * frame_height
            detection_area = w * h
            if detection_area < frame_area * 0.0005:  # Less than 0.05% of frame
                continue

            # Skip detections that are too large relative to frame size
            if detection_area > frame_area * 0.08:  # More than 8% of frame
                continue

            # Skip detections that are too small in absolute terms
            if detection_area < 100:  # Very small detections
                continue

            filtered_detections.append(detection)

        return filtered_detections

    def _detect_bright_regions(self, frame, dynamic_params):
        """
        Fallback method to detect bright circular regions when color detection fails.
        """
        detections = []

        # Convert to grayscale and find bright regions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, dynamic_params['blur_kernel_size'], 0)

        # Use adaptive threshold to find bright regions
        bright_thresh = max(200, int(np.percentile(blurred, 85)))
        _, bright_mask = cv2.threshold(blurred, bright_thresh, 255,
                                       cv2.THRESH_BINARY)

        # Morphology to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Dynamic area bounds
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w
        dyn_min_area = max(dynamic_params['min_area'],
                           int(frame_area * self.min_area_ratio))
        dyn_max_area = min(dynamic_params['max_area'],
                           int(frame_area * self.max_area_ratio))

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h

            if rect_area < dyn_min_area or rect_area > dyn_max_area:
                continue

            area = cv2.contourArea(contour)
            if area <= 0:
                continue

            aspect_ratio = h / w if w > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue

            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (
                perimeter * perimeter) if perimeter > 0 else 0

            if circularity < 0.3:
                continue

            # Check if region is bright enough
            roi = blurred[y:y + h, x:x + w]
            if roi.size == 0:
                continue

            mean_brightness = np.mean(roi)
            if mean_brightness < 200:
                continue

            # Try to determine color from the original image
            roi_bgr = frame[y:y + h, x:x + w]
            roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

            # Simple color classification based on mean hue
            mean_h = np.mean(roi_hsv[:, :, 0])
            mean_s = np.mean(roi_hsv[:, :, 1])
            mean_v = np.mean(roi_hsv[:, :, 2])

            # Classify based on hue and brightness
            if mean_s < 30:  # Low saturation - likely white/yellow
                color = 'Yellow'
            elif 0 <= mean_h <= 15 or 165 <= mean_h <= 180:
                color = 'Red'
            elif 40 <= mean_h <= 80:
                color = 'Green'
            elif 20 <= mean_h <= 40:
                color = 'Yellow'
            else:
                color = 'Yellow'  # Default to yellow for bright regions

            confidence = min(
                0.8, circularity * 0.5 + (mean_brightness - 200) / 55 * 0.3)

            if confidence >= 0.3:
                detections.append({
                    'color': color,
                    'box': (x, y, w, h),
                    'confidence': float(confidence),
                })

        return detections

    def process_frame(self, frame):
        """
        Process a single frame: detect traffic lights and draw results.
        
        Args:
            frame: Current video frame
            
        Returns:
            annotated_frame: Frame with detection results
        """
        if frame is None:
            return None

        # Start timing for performance measurement
        start_time = cv2.getTickCount()

        # Make a copy for visualization
        annotated_frame = frame.copy()

        # Detect traffic lights and update tracks
        detections = self.detect_traffic_lights(frame)
        # Before updating, estimate motion per existing track by comparing to best IoU det
        prev_tracks = [dict(t) for t in self.tracks]
        self._update_tracks(detections)
        # Attach motion estimate and classify type
        for tr in self.tracks:
            # find matching previous track by id
            prev = next((p for p in prev_tracks if p['id'] == tr['id']), None)
            if prev is not None:
                tr['last_speed'] = self._estimate_motion(
                    prev['box'], tr['box'])
            else:
                tr['last_speed'] = 0.0
            tr['type'] = self._classify_track_type(tr, self.tracks)

        # Draw confirmed (locked) tracks
        for tr in self.tracks:
            if not tr['confirmed']:
                continue
            x, y, w, h = tr['box']
            color_name = tr['color']
            confidence = tr['confidence']

            if color_name == 'Red':
                box_color = (0, 0, 255)
            elif color_name == 'Yellow':
                box_color = (0, 255, 255)
            else:
                box_color = (0, 255, 0)

            cv2.rectangle(annotated_frame, (x, y - 24), (x + w, y), box_color,
                          -1)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color,
                          3)
            ttype = tr.get('type', 'TrafficLight')
            label = f"LOCKED {color_name} {ttype} #{tr['id']} ({confidence:.2f})"
            cv2.putText(annotated_frame, label, (x + 5, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Calculate fps
        end_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (end_time - start_time)

        # Simple info bar
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, 32), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)
        locked_count = sum(1 for t in self.tracks if t['confirmed'])
        cv2.putText(annotated_frame, f"Locked: {locked_count}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add FPS counter
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (210, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, timestamp, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated_frame

    def process_image_file(self, image_path, save_output=False):
        """
        Process a single image file for traffic light detection.
        
        Args:
            image_path: Path to the image file
            save_output: Whether to save the processed image
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return None

        # For single image processing, use direct detection instead of tracking
        result_image = self.process_single_image(image)

        # Save output if requested
        if save_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"traffic_light_detection_{timestamp}.jpg"
            cv2.imwrite(output_filename, result_image)
            print(f"Processed image saved to {output_filename}")

        return result_image

    def process_single_image(self, image):
        """
        Process a single image for traffic light detection without tracking.
        This method is optimized for single image processing.
        
        Args:
            image: Input image
            
        Returns:
            annotated_image: Image with detection results drawn
        """
        if image is None:
            return None

        # Make a copy for visualization
        annotated_image = image.copy()

        # Detect traffic lights directly (no tracking)
        detections = self.detect_traffic_lights(image)

        # Draw detections
        for i, detection in enumerate(detections):
            x, y, w, h = detection['box']
            color_name = detection['color']
            confidence = detection['confidence']

            # Choose box color based on detected color
            if color_name == 'Red':
                box_color = (0, 0, 255)  # Red in BGR
            elif color_name == 'Yellow':
                box_color = (0, 255, 255)  # Yellow in BGR
            else:  # Green
                box_color = (0, 255, 0)  # Green in BGR

            # Draw bounding box
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), box_color,
                          3)

            # Draw label background
            cv2.rectangle(annotated_image, (x, y - 25), (x + w, y), box_color,
                          -1)

            # Draw label text
            label = f"{color_name} ({confidence:.2f})"
            cv2.putText(annotated_image, label, (x + 5, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add detection count info
        red_count = sum(1 for d in detections if d['color'] == 'Red')
        yellow_count = sum(1 for d in detections if d['color'] == 'Yellow')
        green_count = sum(1 for d in detections if d['color'] == 'Green')

        # Draw info bar
        info_height = 40
        overlay = annotated_image.copy()
        cv2.rectangle(overlay, (0, 0), (annotated_image.shape[1], info_height),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated_image, 0.3, 0, annotated_image)

        # Add detection counts
        info_text = f"Red: {red_count} | Yellow: {yellow_count} | Green: {green_count} | Total: {len(detections)}"
        cv2.putText(annotated_image, info_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated_image

    def run(self, source=0, save_output=False):
        """
        Run the automatic traffic light detection on a video source.
        
        Args:
            source: Camera index (0 for webcam) or video file path
            save_output: Whether to save the processed video
        """
        # Initialize video capture
        try:
            if isinstance(source, str) and (source.isdigit() or source == '0'):
                source = int(source)
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"Error: Could not open video source {source}")
                return
        except Exception as e:
            print(f"Error opening video source: {e}")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Initialize video writer if saving output
        out = None
        if save_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"auto_traffic_light_detection_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_filename, fourcc, fps,
                                  (width, height))
            print(f"Saving output to {output_filename}")

        # Process video frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            result_frame = self.process_frame(frame)

            # Display the result (press 'd' to toggle mask overlay)
            if self.show_debug_masks:
                debug_panel = np.zeros(
                    (result_frame.shape[0], result_frame.shape[1], 3),
                    dtype=np.uint8)
                # Calculate dynamic params for debug visualization
                debug_params = self._calculate_dynamic_params(frame)
                hsv_dbg = cv2.cvtColor(
                    cv2.GaussianBlur(frame, debug_params['blur_kernel_size'],
                                     0), cv2.COLOR_BGR2HSV)
                for color, ranges in self.color_ranges.items():
                    cmask = np.zeros((frame.shape[0], frame.shape[1]),
                                     dtype=np.uint8)
                    for r in ranges:
                        cmask = cv2.bitwise_or(
                            cmask, cv2.inRange(hsv_dbg, r['lower'],
                                               r['upper']))
                    color_bgr = (0, 0, 255) if color == 'Red' else (
                        0, 255, 255) if color == 'Yellow' else (0, 255, 0)
                    colored = cv2.merge([
                        cmask //
                        2 if color_bgr[0] > 0 else np.zeros_like(cmask),
                        cmask //
                        2 if color_bgr[1] > 0 else np.zeros_like(cmask),
                        cmask //
                        2 if color_bgr[2] > 0 else np.zeros_like(cmask),
                    ])
                    debug_panel = cv2.add(debug_panel, colored)
                combined = np.hstack([result_frame, debug_panel])
                cv2.imshow('Automatic Traffic Light Detector', combined)
            else:
                cv2.imshow('Automatic Traffic Light Detector', result_frame)

            # Save frame if requested
            if out is not None:
                out.write(result_frame)

            # Break on 'q' key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('d'):
                self.show_debug_masks = not self.show_debug_masks
                if self.show_debug_masks:
                    print('[DEBUG] Mask visualization ON (press d to toggle)')
                else:
                    print('[DEBUG] Mask visualization OFF')

        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()


def main():
    """Parse arguments and run the detector."""
    parser = argparse.ArgumentParser(
        description='Automatic Traffic Light Detection')
    parser.add_argument(
        '-s',
        '--source',
        default='0',
        help='Camera index (0), video file path, or image file path')
    parser.add_argument('-o',
                        '--output',
                        action='store_true',
                        help='Save processed video/image to file')
    parser.add_argument('-i',
                        '--image',
                        action='store_true',
                        help='Process as image file instead of video/camera')
    args = parser.parse_args()

    detector = AutoTrafficLightDetector()

    if args.image:
        # Process as image
        result = detector.process_image_file(args.source, args.output)
        if result is not None:
            # Display the result
            cv2.imshow('Traffic Light Detection Result', result)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        # Process as video/camera
        detector.run(args.source, args.output)


if __name__ == "__main__":
    main()
