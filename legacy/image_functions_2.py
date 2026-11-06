import os
import sys
import time
import io
import glob
import numpy as np
import imageio
import imageio.v3 as iio
import piexif
import cv2
import matplotlib.pyplot as plt
import cupy as cp
import exiftool

from PIL import Image
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from scipy import stats
from scipy.ndimage import gaussian_filter1d

class base_processor:
    """
    Base class for all processing strategies.
    Each subclass implements:
      - init(batch): initialise accumulator from first batch
      - update(batch): merge new batch into accumulator
      - finalize(): return final processed image
    """
    name = "base"

    def init(self, batch, **kwargs):
        raise NotImplementedError

    def update(self, batch, **kwargs):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

class average_processor:
    name = "average"

    def __init__(self, **kwargs):
        self.accum_array = None
        self.count = 0

    def init(self, batch, **kwargs):
        self.accum_array = batch.sum(axis=0, dtype=np.float64)
        self.count = batch.shape[0]

    def update(self, batch, **kwargs):
        self.accum_array += batch.sum(axis=0, dtype=np.float64)
        self.count += batch.shape[0]

    def finalize(self):
        return self.accum_array / self.count


class maximum_processor:
    name = "maximum"

    def __init__(self, **kwargs):
        self.max_image = None

    def init(self, batch, **kwargs):
        # batch arrives as clean 0.0 - 1.0 float32 data
        self.max_image = batch.max(axis=0)

    def update(self, batch, **kwargs):
        # Element-wise evaluation tracking the absolute ceiling across frames
        self.max_image = np.maximum(self.max_image, batch.max(axis=0))

    def finalize(self):
        # Returns raw unscaled float matrix directly to the orchestrator export gate
        return self.max_image


class minimum_processor:
    name = "minimum"

    def __init__(self, **kwargs):
        self.min_image = None

    def init(self, batch, **kwargs):
        # batch arrives as clean 0.0 - 1.0 float32 data
        self.min_image = batch.min(axis=0)

    def update(self, batch, **kwargs):
        # Element-wise evaluation tracking the absolute floor across frames
        self.min_image = np.minimum(self.min_image, batch.min(axis=0))

    def finalize(self):
        # Returns raw unscaled float matrix directly to the orchestrator export gate
        return self.min_image


class median_processor:
    name = "median"
    def __init__(self, **kwargs):
        self.images = []
        self.dtype_out = np.uint8

    def init(self, batch, **kwargs):
        self.dtype_out, _ = detect_depth_config(batch)
        self.images = [batch]

    def update(self, batch, **kwargs):
        self.images.append(batch)

    def finalize(self):
        all_imgs = np.concatenate(self.images, axis=0)
        return np.median(all_imgs, axis=0).astype(self.dtype_out)

class mode_processor:
    name = "mode"
    def init(self, batch, **kwargs):  # <--- Added **kwargs
        self.images = [batch]

    def update(self, batch, **kwargs):  # <--- Added **kwargs
        self.images.append(batch)

    def finalize(self):
        all_imgs = np.concatenate(self.images, axis=0)
        mode, _ = stats.mode(all_imgs, axis=0, keepdims=False)
        return mode.astype(np.uint8)

class percentile_processor:
    name = "percentile"
    def __init__(self, percentile=90, **kwargs):
        self.percentile = percentile
        self.images = []

    def init(self, batch):
        self.images.append(batch)

    def update(self, batch):
        self.images.append(batch)

    def finalize(self):
        all_imgs = np.concatenate(self.images, axis=0)
        return np.percentile(all_imgs, self.percentile, axis=0).astype(np.uint8)
    
class temporal_contrast_processor:
    name = "temporal_contrast"
    
    def __init__(self, **kwargs):
        self.max_img = None
        self.min_img = None

    def init(self, batch, **kwargs):
        # batch arrives as clean 0.0 - 1.0 float32 data
        self.max_img = batch.max(axis=0)
        self.min_img = batch.min(axis=0)

    def update(self, batch, **kwargs):
        self.max_img = np.maximum(self.max_img, batch.max(axis=0))
        self.min_img = np.minimum(self.min_img, batch.min(axis=0))

    def finalize(self):
        eps = 1e-6
        # Calculate localized temporal contrast ratio (Outputs absolute normalized range 0.0 to 1.0)
        tce = (self.max_img - self.min_img) / (self.max_img + self.min_img + eps)
        
        # Clamp strictly between 0.0 and 1.0 to protect against floating point rounding edge drift
        return np.clip(tce, 0.0, 1.0)
    
class temporal_gradient_processor:
    name = "temporal_gradient"
    
    def __init__(self, **kwargs):
        self.prev = None
        self.accum = None

    def init(self, batch, **kwargs):
        # batch arrives as clean 0.0 - 1.0 float32 data
        self.prev = batch[-1]
        self.accum = np.zeros_like(self.prev)

    def update(self, batch, **kwargs):
        for img in batch:
            # Accumulate absolute difference step maps natively in floating-point space
            self.accum += np.abs(img - self.prev)
            self.prev = img

    def finalize(self):
        # 1. Normalize accumulation canvas relative to its peak to yield a clean 0.0 - 1.0 map
        out = self.accum / (self.accum.max() + 1e-6)

        # 2. DYNAMIC AUTOMATIC BRIGHTNESS COMPENSATION (Uniform Float Domain)
        mean_val = out.mean()
        target_mean = 100.0 / 255.0  # Normalized target center point baseline (~0.392)
        
        if mean_val < target_mean:
            gain = target_mean / (mean_val + 1e-6)
            out = out * gain

        # Final protection clamp before sending it out to the export pipeline
        return np.clip(out, 0.0, 1.0)
    

class temporal_hue_motion_processor:
    name = "temporal_hue_motion"

    def init(self, batch, **kwargs):
        self.prev = batch[-1].astype(np.float32)
        h, w, _ = self.prev.shape

        self.hue_accum = np.zeros((h, w), np.float32)
        self.motion_accum = np.zeros((h, w), np.float32)
        self.frame_index = 0

    def update(self, batch):
        for img in batch:
            img_f = img.astype(np.float32)

            # Motion energy (per-pixel absolute difference)
            diff = np.abs(img_f - self.prev).mean(axis=2)
            self.prev = img_f

            # Map time → hue in OpenCV range [0,179]
            hue_val = (self.frame_index % 360) * (179.0 / 360.0)

            # Weighted accumulation
            self.hue_accum += diff * hue_val
            self.motion_accum += diff

            self.frame_index += 1

    def finalize(self):
        eps = 1e-6

        # Weighted average hue
        hue = self.hue_accum / (self.motion_accum + eps)
        hue = np.nan_to_num(hue).astype(np.uint8)

        # Brightness = normalised motion energy
        motion_norm = (self.motion_accum / (self.motion_accum.max() + eps)) * 255
        motion_norm = motion_norm.astype(np.uint8)

        sat = np.full_like(hue, 255, dtype=np.uint8)

        hsv = cv2.merge([hue, sat, motion_norm])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb
    
# class temporal_entropy_processor:
#     name = "temporal_entropy"

#     def init(self, batch, hist_bins=None, **kwargs):
#         h, w, c = batch[0].shape

#         # Allow user override
#         self.hist_bins = int(hist_bins) if hist_bins is not None else 32

#         self.hist = np.zeros((h, w, c, self.hist_bins), np.float32)

#     def update(self, batch):
#         for img in batch:
#             img_f = img.astype(np.float32)

#             bins = np.clip(
#                 (img_f / 256.0 * self.hist_bins).astype(int),
#                 0,
#                 self.hist_bins - 1
#             )

#             for ch in range(3):
#                 idx = bins[:, :, ch]
#                 self.hist[
#                     np.arange(idx.shape[0])[:, None],
#                     np.arange(idx.shape[1])[None, :],
#                     ch,
#                     idx
#                 ] += 1

#     def finalize(self):
#         eps = 1e-9
#         p = self.hist / (self.hist.sum(axis=3, keepdims=True) + eps)
#         entropy = -np.sum(p * np.log2(p + eps), axis=3)
#         entropy_norm = (entropy / entropy.max()) * 255
#         return entropy_norm.astype(np.uint8)

class temporal_entropy_processor:
    name = "temporal_entropy"

    def __init__(self, **kwargs):
        self.prev_img_gpu = None
        self.hist_bins = 32
        self.hist = None
        self.grid_y = None
        self.grid_x = None
        self.grid_c = None

    def init(self, batch, hist_bins=None, **kwargs):
        # batch arrives as clean, normalized 0.0 - 1.0 float32 data
        _, h, w, c = batch.shape
        self.hist_bins = int(hist_bins) if hist_bins is not None else 32

        # Initialize the 4D histogram tracker matrix on VRAM
        self.hist = cp.zeros((h, w, c, self.hist_bins), cp.int32)

        # Build indexing grids for the cupy atomic scatter addition
        self.grid_y, self.grid_x, self.grid_c = cp.meshgrid(
            cp.arange(h), cp.arange(w), cp.arange(c), indexing='ij'
        )

    def update(self, batch, **kwargs):
        # Push the normalized float32 slice to VRAM
        batch_gpu = cp.asarray(batch, dtype=cp.float32)

        # Fixed, highly optimized noise floor threshold scaled to the 0.0 - 1.0 float space
        noise_threshold = 3.0 / 255.0  

        for i in range(batch_gpu.shape[0]):
            current_frame = batch_gpu[i]

            if self.prev_img_gpu is None:
                self.prev_img_gpu = current_frame.copy()
                
            # Filter sensor noise variance across frames
            pixel_diff = cp.abs(current_frame - self.prev_img_gpu)
            current_frame = cp.where(pixel_diff < noise_threshold, self.prev_img_gpu, current_frame)
            self.prev_img_gpu = current_frame.copy()

            # Fast bin mapping: multiply by number of bins, floor it, and clamp to array limits
            bins = cp.clip(cp.floor(current_frame * self.hist_bins), 0, self.hist_bins - 1).astype(cp.int32)

            # Atomic histogram accumulation step on GPU
            cp.add.at(self.hist, (self.grid_y, self.grid_x, self.grid_c, bins), 1)

    def finalize(self):
        # Float16 safe epsilon boundary limits to prevent log(0)
        eps = cp.float16(6e-5) 
        h, w, c, bins = self.hist.shape
        
        entropy_final = cp.zeros((h, w, c), cp.float16)
        num_chunks = 32
        chunk_size = h // num_chunks
        
        # Stream processing via VRAM chunking to prevent out-of-memory errors
        for idx in range(num_chunks):
            start_y = idx * chunk_size
            end_y = h if idx == (num_chunks - 1) else (idx + 1) * chunk_size
            
            hist_chunk = self.hist[start_y:end_y].astype(cp.float16)
            hist_sum = hist_chunk.sum(axis=3, keepdims=True)
            hist_chunk /= cp.maximum(hist_sum, cp.float16(1.0))
            
            log_p = cp.log2(cp.maximum(hist_chunk, eps))
            hist_chunk *= log_p
            del log_p
            
            entropy_chunk = -cp.sum(hist_chunk, axis=3)
            del hist_chunk
            
            entropy_final[start_y:end_y] = entropy_chunk
            del entropy_chunk
            cp.get_default_memory_pool().free_all_blocks()

        # Clean VRAM footprints immediately
        self.hist = None
        self.prev_img_gpu = None  
        self.grid_y = self.grid_x = self.grid_c = None
        cp.get_default_memory_pool().free_all_blocks()

        # Divide the calculated Shannon map by its maximum theoretical ceiling limit
        max_theoretical = cp.log2(cp.float16(self.hist_bins))
        entropy_norm = cp.clip(entropy_final / max_theoretical, 0.0, 1.0)
        
        # Return a perfectly clean, normalized 0.0 - 1.0 numpy float array to the exporter
        result = cp.asnumpy(entropy_norm.astype(cp.float32))
        del entropy_final, entropy_norm
        return result

class optical_flow_accum_processor(base_processor):
    name = "optical_flow_accum"

    def init(self, batch, **kwargs):
        self.prev = cv2.cvtColor(batch[-1], cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = self.prev.shape

        self.flow_mag = np.zeros((h, w), np.float32)
        self.flow_ang = np.zeros((h, w), np.float32)
        self.count = 0

    def update(self, batch):
        for img in batch:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

            flow = cv2.calcOpticalFlowFarneback(
                self.prev, gray,
                None,
                pyr_scale=0.5,
                levels=8,
                winsize=4,
                iterations=5,
                poly_n=7,
                poly_sigma=1.2,
                flags=0
            )

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            self.flow_mag += mag
            self.flow_ang += ang
            self.count += 1

            self.prev = gray

    def finalize(self):
        eps = 1e-6

        # Normalised magnitude → brightness
        mag_norm = (self.flow_mag / (self.flow_mag.max() + eps)) * 255
        mag_norm = mag_norm.astype(np.uint8)

        # Average direction → hue
        ang_avg = self.flow_ang / max(self.count, 1)
        hue = (ang_avg * (180 / np.pi / 2)).astype(np.uint8)

        sat = np.full_like(hue, 255, dtype=np.uint8)

        hsv = cv2.merge([hue, sat, mag_norm])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb

class motion_overlay_processor:
    """
    Pass 1: Accumulates a running average to build a clean baseline background.
    Pass 2: Loops through the sequence files, calculates deviations in normalized space,
            and stitches the high-motion objects onto the background.
    """
    name = "motion_overlay"

    def __init__(self, motion_threshold=None, intensity_priority=True, **kwargs):
        self.sum_image = None
        self.count = 0
        self.file_names = []
        self.user_threshold = motion_threshold
        self.motion_threshold = None
        
        # Default fallback rule state tracking
        self.user_intensity_priority = intensity_priority
        self.intensity_priority = True

    def init(self, batch, file_names=None, **kwargs):
        # batch arrives as clean, normalized 0.0 - 1.0 float32 data
        
        # 1. Capture dynamic parameters directly from the configuration dictionary
        dynamic_threshold = kwargs.get("motion_threshold", self.user_threshold)
        
        # Safely extract intensity priority rule state forwarded by the loop parameters
        self.intensity_priority = kwargs.get("intensity_priority", self.user_intensity_priority)
        
        # 2. Determine motion sensitivity threshold natively in 0.0 - 1.0 float space
        if dynamic_threshold is not None:
            if dynamic_threshold > 1.0:
                self.motion_threshold = float(dynamic_threshold) / 255.0
            else:
                self.motion_threshold = float(dynamic_threshold)
        else:
            self.motion_threshold = 22 / 255.0
        
        # Initialize background accumulator
        self.sum_image = batch.sum(axis=0, dtype=np.float64)
        self.count = batch.shape[0]
        
        # Cache file paths to re-read them during Pass 2
        if file_names is not None:
            self.file_names = list(file_names)

    def update(self, batch, file_names=None, **kwargs):
        # Accumulate background baseline data
        self.sum_image += batch.sum(axis=0, dtype=np.float64)
        self.count += batch.shape[0]
        
        if file_names is not None:
            self.file_names.extend(file_names)

    def finalize(self, **kwargs):
        """
        Finalizes the composite image using properties saved in self during step 1.
        """
        if not self.file_names:
            print("[Warning] No file names provided to motion_overlay_processor. Returning baseline background.")
            return np.clip(self.sum_image / self.count, 0.0, 1.0).astype(np.float32)

        import scipy.ndimage as ndimage
        import sys

        print("\nPass 1 Complete. Calculating clean background baseline...")
        background = (self.sum_image / self.count).astype(np.float32)
        self.sum_image = None  # Free up memory footprint
        
        # Initialize final canvas with the static background average
        final_canvas = background.copy()

        # Initialize intensity tracking matrix if requested via self property state
        if self.intensity_priority:
            print(f"\nPass 2: Extracting moving objects (Intensity Priority Mode | Threshold: {self.motion_threshold:.4f})...")
            max_deviation = np.zeros(background.shape[:2], dtype=np.float32)
        else:
            print(f"\nPass 2: Extracting moving objects (Chronological Mode - Newest Wins | Threshold: {self.motion_threshold:.4f})...")

        total_files = len(self.file_names)

        for idx, fn in enumerate(self.file_names, start=1):
            percent = (idx / total_files) * 100
            sys.stdout.write(f"\r→ Overlaying frame {idx}/{total_files} ({percent:5.1f}%)")
            sys.stdout.flush()

            current_frame = load_image(fn)
            if current_frame is None:
                continue

            if current_frame.shape != background.shape:
                if current_frame.shape[0] == background.shape[1] and current_frame.shape[1] == background.shape[0]:
                    current_frame = np.rot90(current_frame, k=-1)
                else:
                    import cv2
                    current_frame = cv2.resize(current_frame, (background.shape[1], background.shape[0]))

            # -----------------------------------------------------------------
            # STATIC BACKGROUND DIFFERENCING VIA MAX COLOR CHANNEL
            # -----------------------------------------------------------------
            diff_background = np.abs(current_frame - background).max(axis=2)
            
            # Base motion detection mask
            motion_mask = diff_background > self.motion_threshold

            # Optional: Smooth out holes or jagged pixel fractures
            motion_mask = ndimage.binary_closing(motion_mask, iterations=2)

            # -----------------------------------------------------------------
            # INTENSITY CONDITIONAL MASKING SWITCH
            # -----------------------------------------------------------------
            if self.intensity_priority:
                # Refine mask: It must pass threshold AND be a greater deviation than seen before
                motion_mask = motion_mask & (diff_background > max_deviation)
                
                # Update our persistent intensity tracking canvas map
                max_deviation[motion_mask] = diff_background[motion_mask]

            # -----------------------------------------------------------------
            # 3-CHANNEL MASK STENCIL OVERRIDE
            # -----------------------------------------------------------------
            final_canvas[motion_mask] = current_frame[motion_mask]

        print(f"\nProcessing complete. Locking final composite layer matrix.")
        return np.clip(final_canvas, 0.0, 1.0)

PROCESSORS = {
    "average": average_processor,
    "maximum": maximum_processor,
    "minimum": minimum_processor,
    "median": median_processor,
    "mode": mode_processor,
    "percentile": percentile_processor,
    "temporal_contrast": temporal_contrast_processor,
    "temporal_gradient": temporal_gradient_processor,
    "temporal_hue_motion": temporal_hue_motion_processor,
    "temporal_entropy": temporal_entropy_processor,
    "optical_flow_accum": optical_flow_accum_processor,
    "motion_overlay": motion_overlay_processor,
}

def detect_depth_config(batch, ext="jpg"):
    """
    Returns the appropriate output dtype and scalar ceiling.
    Forces high-fidelity 16-bit mode if a TIFF export is requested, 
    otherwise falls back to standard 8-bit JPEG constraints.
    """
    # Force 16-bit structures if the user explicitly asked for a TIFF export
    if ext.lstrip(".").lower() in ["tiff", "tif"]:
        return np.uint16, 65535.0  # 16-bit TIFF Mode
    else:
        return np.uint8, 255.0     # 8-bit JPEG Mode

def get_processor(process_type, **kwargs):
    """
    Returns an instance of the correct processor class.
    Accepts arbitrary configuration parameters to support pipeline changes.
    """
    if process_type not in PROCESSORS:
        raise ValueError(f"Unknown process type: {process_type}")

    return PROCESSORS[process_type](**kwargs)

def load_image(fn):
    """
    Loads an image and immediately normalizes it to a 0.0 - 1.0 float32 space,
    safeguarding against channel stride wraps and bit-depth confusion.
    """
    import imageio.v3 as iio
    import numpy as np

    fn_str = str(fn).lower()
    
    if fn_str.endswith('.tiff') or fn_str.endswith('.tif'):
        # Read true 16-bit array numbers natively via tifffile
        img = iio.imread(fn, plugin="tifffile")
        if img.ndim == 3 and img.shape[0] == 3:  # Fix separate color planes
            img = np.moveaxis(img, 0, -1)
            
        # Natively map 16-bit unsigned maximum ceiling to 0.0 - 1.0 float space
        return img.astype(np.float32) / 65535.0
    else:
        # Natively map 8-bit unsigned maximum ceiling to 0.0 - 1.0 float space
        img = iio.imread(fn, plugin="pillow")
        return img.astype(np.float32) / 255.0

# def batch_process_images(file_names, processor, batch_size, **params):
#     first = True
#     count = 0
#     total = len(file_names)
#     start_time = time.time()

#     # Use all 8 cores
#     max_workers = 8

#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         for batch_index in range(0, total, batch_size):
#             batch_files = file_names[batch_index:batch_index + batch_size]

#             # Submit all image loads in parallel
#             futures = [executor.submit(load_image, fn) for fn in batch_files]

#             batch_images = []
#             for f in futures:
#                 img = f.result()
#                 batch_images.append(img)

#                 # Update counters
#                 count += 1
#                 percent = (count / total) * 100

#                 # Progress bar
#                 bar_length = 30
#                 filled = int(bar_length * percent / 100)
#                 bar = "█" * filled + "-" * (bar_length - filled)

#                 # ETA
#                 elapsed = time.time() - start_time
#                 avg_time = elapsed / count
#                 remaining = total - count
#                 eta_seconds = int(avg_time * remaining)
#                 eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

#                 # Single-line overwrite
#                 sys.stdout.write(
#                     f"\r[{bar}] {percent:5.1f}%  ETA: {eta_str}  ({count}/{total})"
#                 )
#                 sys.stdout.flush()

#             batch_images = np.array(batch_images)

#             # First batch initialises processor
#             if first:
#                 processor.init(batch_images, **params)
#                 processor.update(batch_images)
#                 first = False
#             else:
#                 processor.update(batch_images)

#     print("\nProcessing complete.\n")
#     return processor.finalize()

def batch_process_images(file_names, processor, batch_size, **params):
    count = 0
    total = len(file_names)
    start_time = time.time()
    max_workers = 8

    # Create a local copy of parameters to keep source clean
    run_params = params.copy()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_index in range(0, total, batch_size):
            batch_files = file_names[batch_index:batch_index + batch_size]
            futures = [executor.submit(load_image, fn) for fn in batch_files]

            batch_images = []
            for f in futures:
                img = f.result()
                batch_images.append(img)

                count += 1
                percent = (count / total) * 100
                bar_length = 30
                filled = int(bar_length * percent / 100)
                bar = "█" * filled + "-" * (bar_length - filled)

                elapsed = time.time() - start_time
                avg_time = elapsed / count
                remaining = total - count
                eta_seconds = int(avg_time * remaining)
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

                sys.stdout.write(
                    f"\r[{bar}] {percent:5.1f}%  ETA: {eta_str}  ({count}/{total})"
                )
                sys.stdout.flush()

            # -----------------------------------------------------------------
            # FLOATING POINT BATCH COMPILATION
            # -----------------------------------------------------------------
            # Amalgamates lists directly into a single unified float32 4D matrix.
            batch_images = np.array(batch_images, dtype=np.float32)

            # Safely route file paths without polluting global init state
            run_params["file_names"] = batch_files

            if batch_index == 0:
                # 1. Initialize processing frameworks
                processor.init(batch_images, **run_params)
                # 2. Process first batch images immediately
                processor.update(batch_images, **run_params)
            else:
                # Process subsequent batch loops standardly
                processor.update(batch_images, **run_params)

    print("\nProcessing complete.\n")
    return processor.finalize()

class image_stacker:
    """
    High-level controller for:
      - loading filenames
      - validating images
      - running batch or full-RAM processing
      - exporting the final image
    """

    def __init__(self, import_path, export_path):
        self.import_path = import_path
        self.export_path = export_path

    def check_paths(self):
        """Validate import and export directories."""
        if not os.path.isdir(self.import_path):
            raise ValueError(f"Invalid import path: {self.import_path}")
        if not os.path.isdir(self.export_path):
            raise ValueError(f"Invalid export path: {self.export_path}")

    def check_filename(self, export_name, ext="jpg"):
            """
            Check if output file exists with the chosen file type extension and optionally rename.
            Supports selecting formats cleanly via the 'ext' argument.
            """
            # Ensure extension string doesn't have an unintended leading dot
            clean_ext = ext.lstrip(".")
            output_file = os.path.join(self.export_path, f"{export_name}.{clean_ext}")

            if os.path.isfile(output_file):
                print(f"A file with the name {export_name}.{clean_ext} already exists.")
                response = input("Do you want to rename the file? (y/n/c): ")

                if response.lower() == "y":
                    new_name = input("Enter the new name for the file: ")
                    export_name = new_name
                elif response.lower() == "c":
                    print("Operation cancelled.")
                    raise SystemExit

            return export_name

    def load_filenames(self, exiftool_path=r"C:\Users\henry\OneDrive\Python\exiftool-13.59_32\exiftool.exe"):
        """
        Returns image files sorted strictly by internal EXIF capture timestamp.
        """
        files = []
        
        for ext in ["jpg", "tif"]:
            case_insensitive_pattern = "*." + "".join(f"[{char.lower()}{char.upper()}]" for char in ext)
            search_path = os.path.join(self.import_path, case_insensitive_pattern)
            files.extend(glob.glob(search_path))
            
        files = list(set(files))

        # Read EXIF DateTimeOriginal directly from file headers
        file_dates = {}
        try:
            with exiftool.ExifToolHelper(executable=exiftool_path) as et:
                metadata = et.get_tags(files, tags=["EXIF:DateTimeOriginal", "EXIF:CreateDate"])
                for meta in metadata:
                    filepath = meta.get("SourceFile")
                    # Fallback between DateTimeOriginal and CreateDate
                    capture_time = meta.get("EXIF:DateTimeOriginal") or meta.get("EXIF:CreateDate") or ""
                    file_dates[filepath] = capture_time
            
            # Sort files based on actual EXIF date string
            files.sort(key=lambda fn: file_dates.get(fn, fn))
            print(f"There are {len(files)} images in this folder (EXIF Capture Date Sorted)")
        except Exception as e:
            print(f"ExifTool sorting failed ({e}), falling back to natural filename sort.")
            files.sort(key=natural_sort_key)

        return files
    
    def check_images(self, file_names):
            """
            Validates that all images match the dimensions and bit-depth structures 
            of the reference file to ensure smooth, crash-free matrix processing.
            """
            def normalize_mode(mode_str):
                """Collapses internal Pillow 16-bit format strings into a uniform class."""
                if mode_str in ["I;16", "I;16R", "I;16B", "I;16S", "I;16L"]:
                    return "16BIT_RGB_OR_MONO"
                return mode_str

            with Image.open(file_names[0]) as ref_img:
                ref_size = ref_img.size
                ref_mode = normalize_mode(ref_img.mode)
                original_mode_display = ref_img.mode

            print(f"Reference image size: {ref_size}, detected layout mode: {original_mode_display}")

            bad_files = []

            for fn in file_names:
                try:
                    with Image.open(fn) as img:
                        current_mode = normalize_mode(img.mode)
                        
                        # Flag files that have structural layout or size mismatches
                        if img.size != ref_size or current_mode != ref_mode:
                            bad_files.append((fn, f"Size: {img.size}, Mode: {img.mode}"))
                except Exception as e:
                    bad_files.append((fn, f"Error reading file: {e}"))

            if bad_files:
                print("\nThe following files differ in dimensions or bit depth:")
                for fn, info in bad_files:
                    print(f" - {os.path.basename(fn)} ({info})")
                raise ValueError(f"\nTotal mismatched sequence files detected: {len(bad_files)}")
            else:
                print("All images are consistent in size and colour space layout.")

    def get_latest_timestamp(self, file_names):
            """Return the latest EXIF DateTimeOriginal from the sequence, or filesystem mtime."""
            latest_dt = None

            for fn in file_names:
                try:
                    # Use Pillow instead of piexif to look up EXIF safely across JPEGs and TIFFs
                    with Image.open(fn) as img:
                        exif_data = img.getexif()
                        # 36867 is the standard tag ID for DateTimeOriginal
                        dt_str = exif_data.get(36867) if exif_data else None

                    if dt_str:
                        # Handle bytes if returned, otherwise handle string representation
                        if isinstance(dt_str, bytes):
                            dt_str = dt_str.decode("utf-8")
                        dt = datetime.strptime(dt_str.strip("\x00"), "%Y:%m:%d %H:%M:%S")
                    else:
                        dt = datetime.fromtimestamp(os.path.getmtime(fn))

                except Exception:
                    dt = datetime.fromtimestamp(os.path.getmtime(fn))

                if latest_dt is None or dt > latest_dt:
                    latest_dt = dt

            return latest_dt

    def apply_timestamp_to_image(self, output_path, timestamp):
            """
            Safely applies timestamps to exported files. 
            Uses EXIF injecting for JPEGs, and clean OS file-system stamping 
            for 16-bit high-fidelity TIFF containers to prevent file corruption.
            """
            
            # Convert timestamp to standard string formats
            dt_str = timestamp.strftime("%Y:%m:%d %H:%M:%S")
            
            # Check file extension
            _, ext = os.path.splitext(output_path.lower())
            
            # Ensure the file exists on disk before doing anything
            if not os.path.exists(output_path):
                print(f"   [Warning] Target path {output_path} not found for timestamping.")
                return

            posix_timestamp = timestamp.timestamp()

            if ext in [".tiff", ".tif"]:
                # -----------------------------------------------------------------
                # HIGH-FIDELITY TIFF METADATA SAFETY PASS
                # -----------------------------------------------------------------
                # 1. Force the OS to commit any lingering write handles to disk
                try:
                    with open(output_path, 'a'):
                        os.fsync(0)  # Flushes global internal buffers
                except Exception:
                    pass
                
                # 2. Give the file system a split second (50ms) to release the writer lock
                time.sleep(0.05)
                
                # 3. Stamp the structural filesystem access and modification headers
                os.utime(output_path, (posix_timestamp, posix_timestamp))
                print(f"   -> System timestamps applied natively to master TIFF container.")
                
            else:
                # -----------------------------------------------------------------
                # STANDARD JPEG EXIF PIE PIPELINE
                # -----------------------------------------------------------------
                with Image.open(output_path) as img:
                    # Capture existing or build fresh EXIF payload dictionary
                    exif = img.getexif() if img.getexif() is not None else Image.Exif()
                    
                    # Standard Exif tags mapping
                    exif[271] = "Custom Timelapse Engine"  # Make
                    exif[306] = dt_str                     # DateTime
                    exif[36867] = dt_str                   # DateTimeOriginal
                    exif[36868] = dt_str                   # DateTimeDigitized
                    
                    # Resave the 8-bit image with embedded EXIF blocks safely
                    img.save(output_path, format="JPEG", quality=100, exif=exif)
                
                # Small buffer delay for JPEGs too since we just resaved it
                time.sleep(0.05)
                os.utime(output_path, (posix_timestamp, posix_timestamp))

    def export_image(self, export_name, processed_image, file_names, ext="jpg"):
            """
            Centralized export gate. Maps a normalized 0.0 - 1.0 float array
            to the explicit bit-depth required by the destination file type.
            Bakes native Lightroom XMP blocks directly into output TIFF headers via Tag 700.
            """
            import imageio.v3 as iio
            import numpy as np
            import os
            from PIL import Image
            from datetime import timedelta

            clean_ext = ext.lstrip(".")
            full_name = f"{export_name}_{self.current_process_type}"
            output_path = os.path.join(self.export_path, f"{full_name}.{clean_ext}")

            if clean_ext.lower() in ["tiff", "tif"]:
                # Scale cleanly to the absolute 16-bit ceiling and clamp to prevent rounding edge artifacts
                img_scaled = processed_image * 65535.0
                img_to_save = np.clip(img_scaled, 0.0, 65535.0).astype(np.uint16)
                
                print(f"→ Exporting Master 16-bit TIFF. Array bounds: {img_to_save.min()} to {img_to_save.max()}")
                
                # -----------------------------------------------------------------
                # LIGHTROOM XMP HARVESTING
                # -----------------------------------------------------------------
                xmp_payload = None
                if file_names and os.path.exists(file_names[-1]): # Focus on the last image
                    try:
                        with Image.open(file_names[-1]) as source_img:
                            if hasattr(source_img, "info") and "xmp" in source_img.info:
                                xmp_payload = source_img.info["xmp"]
                                # Ensure it's in raw bytes format for header injection
                                if isinstance(xmp_payload, str):
                                    xmp_payload = xmp_payload.encode('utf-8')
                                print("   -> Harvested embedded XMP structure from last frame.")
                    except Exception as e:
                        print(f"   [Notice] Could not parse internal XMP payload from sequence tail: {e}")

                # -----------------------------------------------------------------
                # EXTRA TAGS METADATA INJECTION
                # -----------------------------------------------------------------
                if xmp_payload is not None:
                    # TIFF Tag 700 is the official slot for XMP data.
                    # Format tuple: (code, datatype, count, value, writeonce)
                    # 700: XMP Tag, 1: Type BYTE, len(): count, xmp_payload: data, True: writeonce
                    xmp_tag = (700, 1, len(xmp_payload), xmp_payload, True)
                    
                    iio.imwrite(output_path, img_to_save, plugin="tifffile", extratags=[xmp_tag])
                    print("   -> Natively baked harvested XMP headers into TIFF Tag 700 slot.")
                else:
                    iio.imwrite(output_path, img_to_save, plugin="tifffile")
                    
            else:
                # Scale cleanly to standard 8-bit limits
                img_scaled = processed_image * 255.0
                img_to_save = np.clip(img_scaled, 0.0, 255.0).astype(np.uint8)
                
                print(f"→ Exporting 8-bit Preview JPEG. Array bounds: {img_to_save.min()} to {img_to_save.max()}")
                iio.imwrite(output_path, img_to_save, plugin="pillow", quality=100)

            # -----------------------------------------------------------------
            # TIMESTAMPER GATEWAY PASSTHROUGH (+1 MINUTE FROM LAST)
            # -----------------------------------------------------------------
            if file_names and hasattr(self, 'get_image_timestamp'):
                last_image_path = file_names[-1]
                last_timestamp = self.get_image_timestamp(last_image_path)
                
                target_timestamp = last_timestamp + timedelta(minutes=1)
                
                self.apply_timestamp_to_image(output_path, target_timestamp)
                print(f"   -> Chronological target set to 1 min post-sequence: {target_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    def process_range(self, process_type, start, end, export_name, ext="jpg"):
            """
            Loads a slice of images into RAM and processes them.
            Suitable for small sets. Allows explicit format selection via 'ext'.
            """
            processor = get_processor(process_type)
            self.current_process_type = process_type
            files = self.load_filenames()
            selected = files[start:end]
            print(f"{len(selected)} images selected")

            # FIXED: Use imageio v3 API (imageio.v3.imread) to safely preserve 16-bit structures
            import imageio.v3 as iio
            images = np.array([iio.imread(fn) for fn in selected])

            # Verify all images match in size/mode and check filenames using chosen format extension
            self.check_images(selected)
            self.check_paths()
            export_name = self.check_filename(export_name, ext=ext)

            # For full-RAM processing, skip batching loops
            processor.init(images)
            processor.update(images)
            result = processor.finalize()

            # Export using your selected format flag choice
            self.export_image(export_name, result, selected, ext=ext)

    def process_all(self, process_type, export_name, batch_size=100, ext="jpg", **params):
            """
            Processes all images using batching.
            Suitable for large timelapse sets. Allows explicit format selection via 'ext'.
            """
            # FIXED: Pass 'ext' explicitly into get_processor so the class constructor catches it!
            processor = get_processor(process_type, ext=ext, **params)
            self.current_process_type = process_type
            files = self.load_filenames()

            # Verify all images match in size/mode and check filenames using chosen format extension
            self.check_images(files)
            self.check_paths()
            export_name = self.check_filename(export_name, ext=ext)

            # FIXED: Explicitly pass 'ext=ext' into the batch engine so it doesn't get left behind!
            result = batch_process_images(files, processor, batch_size, ext=ext, **params)

            # Export using your selected format flag choice
            self.export_image(export_name, result, files, ext=ext)

    def process_frames(self, process_type, export_name, blend, step, ext="jpg", **params):
        """
        Processes sliding windows of frames for video blending sequences.
        Features one-time initialization validation, progress bar stdout tracking with ETA,
        and centralized export delegation.
        """
        # Resolve processor class
        processor_class = get_processor(process_type, ext=ext, **params)
        self.current_process_type = process_type

        # Validate paths and check filename format
        self.check_paths()
        export_name = self.check_filename(export_name, ext=ext)
        if export_name is None:
            return

        # Load filenames
        files = self.load_filenames()
        total_files = len(files)

        # -----------------------------------------------------------------
        # 1. EFFICIENT ONE-TIME VALIDATION
        # -----------------------------------------------------------------
        if total_files == 0:
            print("No files found to process.")
            return

        self.check_images(files)  # Validate sequence consistency ONCE at start

        # Calculate total sliding window frames to export
        total_frames = len(range(0, total_files, step))
        print(f"Generating {total_frames} windowed frames (Blend: {blend}, Step: {step})...")

        start_time = time.time()
        frame_index = 1

        # -----------------------------------------------------------------
        # 2. SLIDING WINDOW LOOP WITH PROGRESS BAR
        # -----------------------------------------------------------------
        for i in range(0, total_files, step):
            window_files = files[i:i+blend]
            if len(window_files) == 0:
                break

            # Load and normalize frames (standalone function load_image)
            images = np.array([load_image(fn) for fn in window_files])

            # Instantiate and run processor
            processor = processor_class
            processor.init(images)
            processor.update(images)
            result = processor.finalize()

            # Ensure 3D shape (Height, Width, Channels) if a 4D array was returned
            if result.ndim == 4 and result.shape[0] == 1:
                result = np.squeeze(result, axis=0)

            # Zero-padded frame filename (e.g. sequence_0001)
            frame_export_name = f"{export_name}_{frame_index:04d}"

            # Mute internal verbose export prints to keep terminal progress bar clean
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                self.export_image(
                    export_name=frame_export_name,
                    processed_image=result,
                    file_names=window_files,
                    ext=ext
                )
            finally:
                sys.stdout = old_stdout  # Restore stdout

            # -------------------------------------------------------------
            # 3. TERMINAL PROGRESS BAR UPDATER
            # -------------------------------------------------------------
            percent = (frame_index / total_frames) * 100
            bar_length = 30
            filled = int(bar_length * percent / 100)
            bar = "█" * filled + "-" * (bar_length - filled)

            elapsed = time.time() - start_time
            avg_time = elapsed / frame_index
            remaining = total_frames - frame_index
            eta_seconds = int(avg_time * remaining)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

            sys.stdout.write(
                f"\r[{bar}] {percent:5.1f}%  ETA: {eta_str}  ({frame_index}/{total_frames})"
            )
            sys.stdout.flush()

            frame_index += 1

        print("\nProcess frames complete.")
                
    def process_video_to_video(self, video_path, process_type, blend, step, fps=None):
        """
        Sliding-window processor for video files (MP4, MOV, etc.)
        Exports the processed output directly as a video.
        """

        # Resolve processor class
        processor_class = get_processor(process_type)
        self.current_process_type = process_type

        # Validate export path
        self.check_paths()

        # --- CHANGE 1: auto-generate export name ---
        base = os.path.splitext(os.path.basename(video_path))[0]
        export_name = f"{base}_{process_type}"

        # Open video reader
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()

        total_frames = reader.count_frames()
        print(f"{total_frames} frames found in video")

        # Determine FPS
        if fps is None:
            fps = meta.get("fps", 30)

        # Prepare output video writer
        outfile = os.path.join(self.export_path, f"{export_name}.mp4")
        writer = imageio.get_writer(outfile, fps=fps, quality=10)

        # --- Progress bar setup ---
        count = 0
        start_time = time.time()
        total_output_frames = total_frames // step

        frame_index = 1

        # Sliding window through video frames
        for i in range(0, total_frames, step):

            # Collect frames for this window
            window_frames = []
            for j in range(i, min(i + blend, total_frames)):
                try:
                    frame = reader.get_data(j)
                    window_frames.append(frame)
                except IndexError:
                    break

            if len(window_frames) == 0:
                break

            images = np.array(window_frames)

            # Instantiate and run processor
            processor = processor_class
            processor.init(images)
            processor.update(images)
            result = processor.finalize()

            # Ensure result is 3‑channel for video
            if result.ndim == 2:
                result = np.stack([result] * 3, axis=-1)

            writer.append_data(result)

            # --- CHANGE 2: Progress bar identical to batch_process_images ---
            count += 1
            percent = (count / total_output_frames) * 100

            bar_length = 30
            filled = int(bar_length * percent / 100)
            bar = "█" * filled + "-" * (bar_length - filled)

            elapsed = time.time() - start_time
            avg_time = elapsed / count
            remaining = total_output_frames - count
            eta_seconds = int(avg_time * remaining)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

            sys.stdout.write(
                f"\r[{bar}] {percent:5.1f}%  ETA: {eta_str}  ({count}/{total_output_frames})"
            )
            sys.stdout.flush()

            frame_index += 1

        reader.close()
        writer.close()

        print(f"\nVideo export complete: {outfile}\n")

    def process_video_to_image(self, video_path, process_type, batch_size=50, custom_name=None, **params):
            """
            Processes every frame of a video into a single image using batch_process_images()
            conforming perfectly to the 0.0 - 1.0 normalized pipeline.
            """
            import threading

            # FIX: Explicitly pass the keyword parameters into the processor instance
            # This forces motion_threshold to land in the processor's __init__ method!
            processor = get_processor(process_type, **params)
            self.current_process_type = process_type

            # Validate paths
            self.check_paths()

            if custom_name is not None:
                export_name = custom_name
            else:
                base = os.path.splitext(os.path.basename(video_path))[0]
                export_name = f"{base}_video"

            # 2. Open ONE single persistent video reader instance
            video_reader = imageio.v2.get_reader(video_path, format="ffmpeg")
            total_frames = video_reader.count_frames()
            print(f"\n→ {total_frames} frames identified in video container.")

            fake_file_list = [f"frame_{i:06d}" for i in range(total_frames)]
            reader_lock = threading.Lock()

            # 3. Define the Thread-Safe Normalized Override Loader
            def load_video_frame_normalized(fake_name):
                frame_idx = int(fake_name.split("_")[1])
                with reader_lock:
                    frame = video_reader.get_data(frame_idx)
                return frame.astype(np.float32) / 255.0

            global load_image
            original_loader = load_image
            load_image = load_video_frame_normalized

            try:
                # 5. Run batch processing (passing params down as well)
                processed_float_image = batch_process_images(
                    fake_file_list,
                    processor,
                    batch_size=batch_size,
                    **params
                )
            finally:
                load_image = original_loader
                video_reader.close()
                print("→ Thread-safe FFMPEG video reader closed cleanly.")

            # 6. Divert result directly to your centralized master export gate
            export_format = params.get("ext", "jpg")
            self.export_image(
                export_name=export_name,
                processed_image=processed_float_image,
                file_names=[video_path],
                ext=export_format
            )

            return export_name

    def sweep_motion_thresholds(self, video_path, threshold_list, batch_size=100, **params):
            """
            Runs a motion threshold sweep by executing the standard video processor 
            from scratch for each threshold value in the list.
            """
            import os

            base_name = os.path.splitext(os.path.basename(video_path))[0]
            export_format = params.get("ext", "jpg")

            print(f"🚀 Starting Straightforward Motion Threshold Sweep for: {base_name}")
            print(f"Sweeping values: {threshold_list}\n")

            for thresh in threshold_list:
                print("\n" + "="*80)
                print(f"▶️ RUNNING SWEEP STEP: Threshold = {thresh}")
                print("="*80)

                run_params = params.copy()
                
                # If the user passes an integer like 5 or 40, normalize it to float space on-the-fly
                # so the processor can compare it against normalized float images seamlessly.
                normalized_val = float(thresh) / 255.0 if float(thresh) > 1.0 else float(thresh)
                
                run_params["motion_threshold"] = normalized_val
                run_params["ext"] = export_format

                step_export_name = f"{base_name}_thresh_{thresh}"

                self.process_video_to_image(
                    video_path=video_path,
                    process_type="motion_overlay",
                    batch_size=batch_size,
                    custom_name=step_export_name,
                    **run_params
                )
                print(f"💾 Step complete! Export engine generated target: {step_export_name}")

            print("\n✨ All threshold sweep iterations completed successfully!")
# ---------------------------------------------------------
# STABALIZER 
# ---------------------------------------------------------


class point_selector:
    def __init__(self, folder_path, max_display_size=1200):
        """
        max_display_size: longest side of the displayed image (keeps GUI manageable)
        """
        self.folder_path = folder_path
        self.max_display_size = max_display_size

        # Load all JPEG variants
        exts = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
        self.files = []
        for e in exts:
            self.files.extend(glob.glob(os.path.join(folder_path, e)))

        self.files = sorted(self.files)

        if not self.files:
            raise ValueError(f"No JPEG images found in folder: {folder_path}")

        self.total_frames = len(self.files)
        self.frame_index = 0
        self.points = []

        self.window_name = "Point Selector"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # scale factor (computed per frame)
        self.scale = 1.0

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert displayed coords → original coords
            orig_x = int(x / self.scale)
            orig_y = int(y / self.scale)
            self.points.append((orig_x, orig_y))
            print(f"Added point (original coords): {(orig_x, orig_y)}")

    def draw_points(self, frame):
        for (x, y) in self.points:
            # draw scaled points
            sx = int(x * self.scale)
            sy = int(y * self.scale)
            cv2.circle(frame, (sx, sy), 5, (0, 255, 0), -1)
        return frame

    def get_frame(self, index):
        img = cv2.imread(self.files[index])
        if img is None:
            raise ValueError(f"Could not read image: {self.files[index]}")
        return img

    def resize_to_screen(self, img):
        h, w = img.shape[:2]
        longest = max(h, w)

        if longest > self.max_display_size:
            self.scale = self.max_display_size / longest
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        else:
            self.scale = 1.0

        return img

    def run(self):
        print("Controls:")
        print("  A = previous frame")
        print("  D = next frame")
        print("  Left click = add point")
        print("  S = save points")
        print("  ESC = quit")

        while True:
            frame = self.get_frame(self.frame_index)
            frame = self.resize_to_screen(frame)
            frame_disp = self.draw_points(frame.copy())

            cv2.imshow(self.window_name, frame_disp)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('a'):
                self.frame_index = max(0, self.frame_index - 1)

            elif key == ord('d'):
                self.frame_index = min(self.total_frames - 1, self.frame_index + 1)

            elif key == ord('s'):
                cv2.destroyAllWindows()
                return self.points, self.frame_index

            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return None, None


def track_points_in_sequence(folder_path, points, start_frame_index=0):
    """
    Tracks user-selected points across a sequence of JPEG images.
    Returns an array of shape (num_frames, num_points, 2)
    """

    # Load all frames
    files = sorted(
        glob.glob(os.path.join(folder_path, "*.jpg")))

    if not files:
        raise ValueError("No JPEG images found.")

    # Convert points to float32 for LK optical flow
    p0 = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

    num_frames = len(files)
    num_points = len(points)

    # Output trajectory array
    trajectory = np.zeros((num_frames, num_points, 2), dtype=np.float32)

    # Load the first frame
    prev_frame = cv2.imread(files[start_frame_index])
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    trajectory[start_frame_index] = p0[:, 0, :]

    # Progress bar setup
    count = 0
    total = num_frames - (start_frame_index + 1)
    start_time = time.time()

    # Track forward
    for i in range(start_frame_index + 1, num_frames):

        frame = cv2.imread(files[i])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p1, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, p0, None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # Store tracked points
        trajectory[i] = p1[:, 0, :]

        # Update for next iteration
        prev_gray = gray.copy()
        p0 = p1.reshape(-1, 1, 2)

        # -------------------------------
        # Progress bar (same as batch)
        # -------------------------------
        count += 1
        percent = (count / total) * 100

        bar_length = 30
        filled = int(bar_length * percent / 100)
        bar = "█" * filled + "-" * (bar_length - filled)

        elapsed = time.time() - start_time
        avg_time = elapsed / count
        remaining = total - count
        eta_seconds = int(avg_time * remaining)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

        sys.stdout.write(
            f"\r[{bar}] {percent:5.1f}%  ETA: {eta_str}  ({count}/{total})"
        )
        sys.stdout.flush()

    print("\nTracking complete.\n")
    return trajectory

def plot_point_drift(trajectory):
    """
    trajectory: array of shape (num_frames, num_points, 2)
    Plots ΔX vs ΔY for each point, coloured by frame index.
    """

    num_frames, num_points, _ = trajectory.shape
    origin = trajectory[0]  # reference positions

    # Prepare colour map for time
    t = np.arange(num_frames)

    plt.figure(figsize=(10, 10))
    plt.title("XY Drift of Tracking Points (Colour = Frame Number)")
    plt.xlabel("ΔX (pixels)")
    plt.ylabel("ΔY (pixels)")

    for p in range(num_points):
        dx = trajectory[:, p, 0] - origin[p, 0]
        dy = trajectory[:, p, 1] - origin[p, 1]

        # Scatter plot with colour mapped to time
        plt.scatter(
            dx, dy,
            c=t,
            cmap="viridis",
            s=20,
            alpha=0.8,
            label=f"Point {p}"
        )

        # Draw a line through the points for clarity
        plt.plot(dx, dy, alpha=0.3)

        # Mark the starting point
        plt.scatter(dx[0], dy[0], c="red", s=80, marker="x")

    plt.grid(True)
    plt.legend()
    plt.colorbar(label="Frame Number")
    plt.axis("equal")  # keep aspect ratio correct
    plt.show()


    # Original positions
    origin = trajectory[0]

    plt.figure(figsize=(12, 8))

    for p in range(num_points):
        dx = trajectory[:, p, 0] - origin[p, 0]
        dy = trajectory[:, p, 1] - origin[p, 1]
        dist = np.sqrt(dx**2 + dy**2)

        plt.plot(dist, label=f"Point {p} total drift")

    plt.title("Point Drift Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Pixel Drift")
    plt.legend()
    plt.grid(True)
    plt.show()


def compute_translation_stabilisation(trajectory, smooth_sigma=1.25):
    """
    trajectory: (num_frames, num_points, 2)
    Returns:
        dx_smooth, dy_smooth: stabilisation offsets per frame
    """

    num_frames, num_points, _ = trajectory.shape
    origin = trajectory[0]

    # Compute raw drift
    dx = trajectory[:, :, 0] - origin[:, 0]
    dy = trajectory[:, :, 1] - origin[:, 1]

    # Median drift per frame (robust to outliers)
    dx_med = np.median(dx, axis=1)
    dy_med = np.median(dy, axis=1)

    # Smooth the drift to avoid jitter
    dx_smooth = gaussian_filter1d(dx_med, sigma=smooth_sigma)
    dy_smooth = gaussian_filter1d(dy_med, sigma=smooth_sigma)

    return dx_smooth, dy_smooth

def apply_translation_stabilisation(folder_in, folder_out, dx, dy):
    """
    Applies translation stabilisation to a folder of JPEG images.
    Includes a batch-style progress bar.
    """

    files = sorted(
        glob.glob(os.path.join(folder_in, "*.jpg"))
    )

    os.makedirs(folder_out, exist_ok=True)

    # Ensure we never index beyond dx/dy length
    total = min(len(files), len(dx))
    count = 0
    start_time = time.time()

    for i in range(total):
        f = files[i]
        frame = cv2.imread(f)
        h, w = frame.shape[:2]

        # Build affine transform matrix
        M = np.float32([
            [1, 0, -dx[i]],
            [0, 1, -dy[i]]
        ])

        stabilised = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)

        out_path = os.path.join(folder_out, os.path.basename(f))
        cv2.imwrite(out_path, stabilised)

        # -----------------------------
        # Progress bar (same as batch)
        # -----------------------------
        count += 1
        percent = (count / total) * 100

        bar_length = 30
        filled = int(bar_length * percent / 100)
        bar = "█" * filled + "-" * (bar_length - filled)

        elapsed = time.time() - start_time
        avg_time = elapsed / count
        remaining = total - count
        eta_seconds = int(avg_time * remaining)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

        sys.stdout.write(
            f"\r[{bar}] {percent:5.1f}%  ETA: {eta_str}  ({count}/{total})"
        )
        sys.stdout.flush()

    print("\nStabilisation complete.\n")

import subprocess
import re

# =========================================================================
# HELPER: TEMPORAL WINDOW PROCESSOR DISPATCHER
# =========================================================================
def apply_window_processor(window_frames, ptype, kwargs, blend=1):
    """
    Applies chosen frame-stacking processor across a 3D/4D numpy array window (Time, Height, Width, Channels).
    Expects frames in normalized 0.0 - 1.0 float32 format.
    """
    if len(window_frames) == 1 or blend == 1:
        return window_frames[0]

    ptype = ptype.lower()

    if ptype in ("average", "mean"):
        return np.mean(window_frames, axis=0)
    
    elif ptype == "maximum":
        return np.max(window_frames, axis=0)
    
    elif ptype == "minimum":
        return np.min(window_frames, axis=0)
    
    elif ptype in ("temporal_contrast", "std"):
        std_img = np.std(window_frames, axis=0)
        max_val = np.max(std_img)
        return std_img / max_val if max_val > 0 else std_img

    elif ptype == "temporal_gradient":
        diff = np.abs(window_frames[-1] - window_frames[0])
        max_val = np.max(diff)
        return diff / max_val if max_val > 0 else diff

    elif ptype == "temporal_entropy":
        bins = kwargs.get("hist_bins", 32)
        norm_frames = np.clip(window_frames, 0.0, 1.0)
        quantized = (norm_frames * (bins - 1)).astype(np.int32)
        
        entropy_img = np.zeros_like(window_frames[0], dtype=np.float32)
        for c in range(quantized.shape[-1]):
            for b in range(bins):
                p = np.mean(quantized[..., c] == b, axis=0)
                mask = p > 0
                entropy_img[..., c] -= np.where(mask, p * np.log2(p + 1e-12), 0)
        
        max_e = np.max(entropy_img)
        return entropy_img / max_e if max_e > 0 else entropy_img

    elif ptype == "motion_overlay":
        threshold = kwargs.get("motion_threshold", 20) / 255.0
        priority = kwargs.get("intensity_priority", True)
        
        result = window_frames[0].copy()
        for frame in window_frames[1:]:
            diff = np.abs(frame - result)
            motion_mask = np.max(diff, axis=-1) > threshold
            if priority:
                brighter_mask = np.mean(frame, axis=-1) > np.mean(result, axis=-1)
                update_mask = motion_mask & brighter_mask
            else:
                update_mask = motion_mask
            result[update_mask] = frame[update_mask]
        return result

    else:
        # Default fallback
        return np.mean(window_frames, axis=0)


# =========================================================================
# MAIN PIPELINE FUNCTION
# =========================================================================
def render_timelapse_pipeline(
    import_path, 
    export_path, 
    export_name, 
    process_type="average",
    process_kwargs=None,
    blend=1, 
    step=1, 
    fps=24,
    export_master=True,          # Toggle Master Video Export ON/OFF
    master_codec="prores",       # Options: "prores" or "hevc_lossless"
    target_1080p=True,           # Toggle 1080p Preview Export ON/OFF
    target_4k=False,             # Toggle 4K Preview Export ON/OFF
    ffmpeg_path="ffmpeg",        
    max_workers=8                
):
    """
    Reads image sequences, applies selected temporal processors across rolling windows,
    and pipes frames directly into FFmpeg to generate Master and/or Preview videos.
    """
    if process_kwargs is None:
        process_kwargs = {}

    if not export_master and not target_1080p and not target_4k:
        print("⚠️ All export targets are set to False. Skipping render.")
        return

    # 1. Collect & validate file paths
    files = [
        os.path.join(import_path, f) for f in os.listdir(import_path) 
        if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg'))
    ]
    
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
    files.sort(key=natural_sort_key)
    
    total_files = len(files)
    if total_files == 0:
        print("No image files found in directory.")
        return

    # Probe dimensions from first image (assuming load_image is defined in image_functions_2)
    first_frame = load_image(files[0])
    height, width, channels = first_frame.shape

    total_output_frames = len(range(0, total_files, step))
    print(f"Found {total_files} source frames.")
    print(f"Processor: {process_type.upper()} | Output: {total_output_frames} frames @ {fps} fps (Blend: {blend}, Step: {step})")
    print(f"Native Resolution: {width}x{height}")

    os.makedirs(export_path, exist_ok=True)

    # -----------------------------------------------------------------
    # 2. CONFIGURE ACTIVE FFMPEG PIPE PROCESSES
    # -----------------------------------------------------------------
    master_proc = None
    if export_master:
        master_file = os.path.join(export_path, f"{export_name}_MASTER.mov")
        if master_codec == "prores":
            master_ffmpeg_cmd = [
                ffmpeg_path, '-y', '-threads', '0',
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}', '-pix_fmt', 'rgb48le',
                '-r', str(fps), '-i', '-',
                '-c:v', 'prores_ks', '-profile:v', '3',
                '-pix_fmt', 'yuv422p10le', master_file
            ]
        else:
            master_ffmpeg_cmd = [
                ffmpeg_path, '-y', '-threads', '0',
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}', '-pix_fmt', 'rgb48le',
                '-r', str(fps), '-i', '-',
                '-c:v', 'libx265', '-crf', '0', '-preset', 'fast',
                '-pix_fmt', 'yuv420p10le', master_file
            ]
        master_proc = subprocess.Popen(master_ffmpeg_cmd, stdin=subprocess.PIPE)

    preview_1080p_proc = None
    if target_1080p:
        preview_1080p_file = os.path.join(export_path, f"{export_name}_1080p.mp4")
        preview_1080p_cmd = [
            ffmpeg_path, '-y', '-threads', '0',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgb48le',
            '-r', str(fps), '-i', '-',
            '-vf', 'scale=-2:1080:flags=bicubic',
            '-c:v', 'libx264', '-crf', '18', '-preset', 'ultrafast',
            '-pix_fmt', 'yuv420p', preview_1080p_file
        ]
        preview_1080p_proc = subprocess.Popen(preview_1080p_cmd, stdin=subprocess.PIPE)

    preview_4k_proc = None
    if target_4k:
        preview_4k_file = os.path.join(export_path, f"{export_name}_4K.mp4")
        preview_4k_cmd = [
            ffmpeg_path, '-y', '-threads', '0',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgb48le',
            '-r', str(fps), '-i', '-',
            '-vf', 'scale=3840:-2:flags=bicubic',
            '-c:v', 'libx264', '-crf', '18', '-preset', 'ultrafast',
            '-pix_fmt', 'yuv420p', preview_4k_file
        ]
        preview_4k_proc = subprocess.Popen(preview_4k_cmd, stdin=subprocess.PIPE)

    # -----------------------------------------------------------------
    # 3. PROCESSING & PIPING LOOP
    # -----------------------------------------------------------------
    start_time = time.time()
    frame_index = 1

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in range(0, total_files, step):
                window_files = files[i:i+blend]
                if len(window_files) == 0:
                    break

                window_frames = np.array(list(executor.map(load_image, window_files)))
                
                # Call standalone helper
                processed_float = apply_window_processor(
                    window_frames, 
                    ptype=process_type, 
                    kwargs=process_kwargs, 
                    blend=blend
                )

                processed_16bit = np.clip(processed_float * 65535.0, 0, 65535).astype(np.uint16)
                raw_bytes = processed_16bit.tobytes()

                # Pipe bytes to active subprocesses
                if master_proc:
                    master_proc.stdin.write(raw_bytes)
                if preview_1080p_proc:
                    preview_1080p_proc.stdin.write(raw_bytes)
                if preview_4k_proc:
                    preview_4k_proc.stdin.write(raw_bytes)

                percent = (frame_index / total_output_frames) * 100
                bar = "█" * int(30 * percent / 100) + "-" * (30 - int(30 * percent / 100))
                elapsed = time.time() - start_time
                avg_time = elapsed / frame_index
                eta = int(avg_time * (total_output_frames - frame_index))
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))

                sys.stdout.write(f"\r[{bar}] {percent:5.1f}%  ETA: {eta_str}  ({frame_index}/{total_output_frames})")
                sys.stdout.flush()

                frame_index += 1

    finally:
        if master_proc:
            master_proc.stdin.close()
            master_proc.wait()
        if preview_1080p_proc:
            preview_1080p_proc.stdin.close()
            preview_1080p_proc.wait()
        if preview_4k_proc:
            preview_4k_proc.stdin.close()
            preview_4k_proc.wait()

    print("\n✅ Render task completed successfully:")
    if export_master:
        print(f"   • Master: {master_file}")
    if target_1080p:
        print(f"   • 1080p Preview: {preview_1080p_file}")
    if target_4k:
        print(f"   • 4K Preview: {preview_4k_file}")