import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from rembg import remove
from scipy import ndimage
from scipy.ndimage import gaussian_filter

# Fix basicsr compatibility with newer torchvision
import fix_basicsr

# Initialize Real-ESRGAN upscaler (lazy loading)
_upscaler = None

def get_upscaler():
    """
    Lazy load the Real-ESRGAN upscaler.
    Uses RealESRGAN_x2plus model for conservative, high-quality upscaling.
    """
    global _upscaler
    if _upscaler is None:
        try:
            print("  Loading Real-ESRGAN model (first time only)...")
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            _upscaler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=False  # Set to True if you have GPU for faster processing
            )
        except Exception as e:
            print(f"  Warning: Could not load Real-ESRGAN ({str(e)})")
            _upscaler = False  # Mark as failed
    return _upscaler if _upscaler is not False else None

def intelligent_upscale(image_array, max_dimension=1000):
    """
    Intelligently upscale image for human photos with natural-looking results.
    Very conservative - only upscales small images and stops at reasonable size.
    
    Args:
        image_array: Input image as numpy array (RGB or RGBA)
        max_dimension: Target maximum dimension (default 1000px)
    
    Returns:
        Upscaled image array
    """
    height, width = image_array.shape[:2]
    current_max = max(height, width)
    
    # Only upscale if image is significantly smaller than target
    # Leave a buffer zone to avoid unnecessary upscaling
    if current_max >= 800:
        print(f"  Image size good ({width}x{height}), skipping upscale for natural look")
        return image_array
    
    # Calculate how much we need to scale
    scale_needed = max_dimension / current_max
    
    # For human photos, be very conservative
    # Only use minimal upscaling to reach around 800-900px range
    if current_max < 400:
        # Very small image - upscale 2x
        scale_factor = 2
        print(f"  Small image detected. Upscaling {width}x{height} -> 2x with Real-ESRGAN...")
    elif current_max < 600:
        # Medium-small image - upscale 1.5x for natural look
        scale_factor = 1.5
        print(f"  Upscaling {width}x{height} -> 1.5x with Real-ESRGAN for natural look...")
    else:
        # Already decent size (600-800px) - minimal upscale
        # Calculate just enough to reach ~850px
        scale_factor = min(850 / current_max, 1.5)
        print(f"  Light upscale {width}x{height} -> {scale_factor:.1f}x to maintain natural photo quality...")
    
    # Separate alpha channel if present
    has_alpha = image_array.shape[2] == 4 if len(image_array.shape) == 3 else False
    
    if has_alpha:
        rgb = image_array[:, :, :3]
        alpha = image_array[:, :, 3]
    else:
        rgb = image_array
        alpha = None
    
    # Convert RGB to BGR for Real-ESRGAN
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # Upscale with Real-ESRGAN
    try:
        upscaler = get_upscaler()
        if upscaler is None:
            print(f"  Real-ESRGAN not available, skipping upscale")
            return image_array
            
        upscaled_bgr, _ = upscaler.enhance(bgr, outscale=scale_factor)
        upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
        
        # Upscale alpha channel if present (use high-quality interpolation)
        if has_alpha:
            new_height, new_width = upscaled_rgb.shape[:2]
            upscaled_alpha = cv2.resize(alpha, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            result = np.dstack([upscaled_rgb, upscaled_alpha])
        else:
            result = upscaled_rgb
        
        final_height, final_width = result.shape[:2]
        final_max = max(final_width, final_height)
        print(f"  ✓ Upscaled to {final_width}x{final_height} (max: {final_max}px) - natural photo quality preserved")
        
        return result
        
    except Exception as e:
        print(f"  Warning: Upscaling failed ({str(e)}), using original image")
        return image_array

def detect_background_color(image_array):
    """
    Detect if the image has a predominantly white or black background.
    Returns 'white', 'black', or 'other'
    """
    height, width = image_array.shape[:2]
    
    edge_pixels = []
    
    edge_thickness = min(20, width//10, height//10)
    
    edge_pixels.extend(image_array[:edge_thickness, :].reshape(-1, 3))
    edge_pixels.extend(image_array[-edge_thickness:, :].reshape(-1, 3))
    
    edge_pixels.extend(image_array[:, :edge_thickness].reshape(-1, 3))
    edge_pixels.extend(image_array[:, -edge_thickness:].reshape(-1, 3))

    edge_pixels = np.array(edge_pixels)
    
    avg_color = np.mean(edge_pixels, axis=0)
    
    # Check white bg
    if np.all(avg_color > 200):
        return 'white'
    # Check black bg
    elif np.all(avg_color < 55):
        return 'black'
    else:
        return 'other'

def smooth_alpha_channel(alpha, blur_radius=2):
    """
    Apply advanced smoothing to the alpha channel for better edges.
    """
    
    smoothed = gaussian_filter(alpha.astype(np.float32), sigma=blur_radius)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smoothed = cv2.morphologyEx(smoothed.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
    
    smoothed = gaussian_filter(smoothed.astype(np.float32), sigma=1)
    
    return smoothed.astype(np.uint8)

def remove_background_ai(image_array):
    """
    Remove background using AI-based approach with rembg.
    This provides the highest quality results.
    """
    try:
        pil_image = Image.fromarray(image_array)
        
        result = remove(pil_image)
        
        result_array = np.array(result)
        
        # if result_array.shape[2] == 4:
        #     alpha = result_array[:, :, 3]
        #     smoothed_alpha = smooth_alpha_channel(alpha, blur_radius=0)
        #     result_array[:, :, 3] = smoothed_alpha
        
        return result_array, True
    except Exception as e:
        print(f"  Warning: AI removal failed ({str(e)}), falling back to color-based method")
        return None, False

def remove_background_color_based_enhanced(image_array, bg_color):
    """
    Enhanced color-based background removal with superior edge smoothing.
    """
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    
    if bg_color == 'white': # white
        lower_white_hsv = np.array([0, 0, 180])
        upper_white_hsv = np.array([180, 40, 255])
        mask_hsv = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)
        
        lower_white_lab = np.array([180, 0, 0])
        upper_white_lab = np.array([255, 140, 140])
        mask_lab = cv2.inRange(lab, lower_white_lab, upper_white_lab)
        
        mask = cv2.bitwise_or(mask_hsv, mask_lab)
        
    else:  # black
        lower_black_hsv = np.array([0, 0, 0])
        upper_black_hsv = np.array([180, 255, 60])
        mask_hsv = cv2.inRange(hsv, lower_black_hsv, upper_black_hsv)
        
        lower_black_lab = np.array([0, 0, 0])
        upper_black_lab = np.array([80, 140, 140])
        mask_lab = cv2.inRange(lab, lower_black_lab, upper_black_lab)
        
        mask = cv2.bitwise_or(mask_hsv, mask_lab)
    
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)
    
    dist_transform = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
    dist_transform = np.clip(dist_transform * 50, 0, 255).astype(np.uint8)
    
    mask_smooth = cv2.max(255 - mask, dist_transform)

    mask_smooth = gaussian_filter(mask_smooth.astype(np.float32), sigma=2.0)
    mask_smooth = gaussian_filter(mask_smooth, sigma=0.8)
    
    alpha = mask_smooth.astype(np.uint8)
    
    rgba_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2RGBA)
    rgba_image[:, :, 3] = alpha
    
    return rgba_image

def process_image(input_path, output_path):
    """
    Process a single image to remove background and save as PNG.
    Returns True if successful, False otherwise.
    """
    try:
        image = cv2.imread(input_path)
        if image is None:
            return False, "Could not read image"
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect bg color
        bg_color = detect_background_color(image_rgb)
        
        if bg_color == 'other':
            # try to remove bg even if not white/black
            print(f"  Background color not white/black, trying AI removal...")
            ai_result, ai_success = remove_background_ai(image_rgb)
            if ai_success:
                result_image = ai_result
                method_used = "AI-based removal (background color: other)"
            else:
                return False, f"Background is not white or black and AI removal failed"
        else:
            print(f"  Detected {bg_color} background, trying AI removal...")
            ai_result, ai_success = remove_background_ai(image_rgb)
            
            if ai_success:
                result_image = ai_result
                method_used = f"AI-based removal (detected {bg_color} background)"
            else:
                print(f"  Using enhanced color-based removal...")
                result_image = remove_background_color_based_enhanced(image_rgb, bg_color)
                method_used = f"Enhanced color-based removal ({bg_color} background)"
        
        # Add 40px margin to the top first
        pil_image = Image.fromarray(result_image)
        
        # Get original dimensions
        original_width, original_height = pil_image.size
        
        # Create new image with 40px extra height at top
        top_margin = 10
        new_height = original_height + top_margin
        new_image = Image.new('RGBA', (original_width, new_height), (0, 0, 0, 0))
        
        # Paste the original image with 40px offset from top (using alpha channel as mask)
        new_image.paste(pil_image, (0, top_margin), pil_image)
        
        # Convert back to array for upscaling
        final_image_array = np.array(new_image)
        
        # Upscale image intelligently if needed (after adding margin)
        final_image_array = intelligent_upscale(final_image_array, max_dimension=1000)
        
        # Convert back to PIL and save
        final_pil_image = Image.fromarray(final_image_array)
        final_pil_image.save(output_path, 'PNG', optimize=False, compress_level=1)
        
        return True, method_used
        
    except Exception as e:
        return False, f"Error processing image: {str(e)}"

def main():
    """
    Main function to process all images in the input folder.
    """
    input_folder = 'input'
    output_folder = 'output'
    
    os.makedirs(output_folder, exist_ok=True)
    
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    results = []
    processed_count = 0
    skipped_count = 0
    
    print("Starting Enhanced Background Removal Process...")
    print("Using AI-based removal with fallback to enhanced color-based method")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print("-" * 60)
    
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist!")
        return
    
    files = os.listdir(input_folder)
    if not files:
        print(f"No files found in '{input_folder}' folder!")
        return
    
    for filename in files:
        if filename.startswith('.'):
            continue
            
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension not in supported_extensions:
            continue
        
        input_path = os.path.join(input_folder, filename)
        
        base_name = os.path.splitext(filename)[0]
        output_filename = base_name + '.png'
        output_path = os.path.join(output_folder, output_filename)
        
        print(f"Processing: {filename}")
        
        success, message = process_image(input_path, output_path)
        
        if success:
            processed_count += 1
            print(f"  ✓ {message}")
            results.append(f"✓ {filename} -> {output_filename}: {message}")
        else:
            skipped_count += 1
            print(f"  ✗ Skipped: {message}")
            results.append(f"✗ {filename}: {message}")
        
        print()
    
    with open('output.txt', 'w') as f:
        f.write("Enhanced Background Removal Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total files processed: {processed_count}\n")
        f.write(f"Total files skipped: {skipped_count}\n\n")
        f.write("Detailed Results:\n")
        f.write("-" * 30 + "\n")
        
        for result in results:
            f.write(result + "\n")
    
    print("-" * 60)
    print(f"Enhanced background removal completed!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Skipped: {skipped_count} images")
    print(f"Results saved to: output.txt")

if __name__ == "__main__":
    main()
