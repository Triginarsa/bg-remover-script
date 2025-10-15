import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from rembg import remove
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import time
import sys
from io import StringIO

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
    if current_max >= 800:
        return image_array
    
    # Calculate scale factor
    if current_max < 400:
        scale_factor = 2
    elif current_max < 600:
        scale_factor = 1.5
    else:
        scale_factor = min(850 / current_max, 1.5)
    
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
    
    # Upscale with Real-ESRGAN (suppress tile output)
    try:
        upscaler = get_upscaler()
        if upscaler is None:
            return image_array
        
        # Redirect stdout to suppress tile messages
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            upscaled_bgr, _ = upscaler.enhance(bgr, outscale=scale_factor)
        finally:
            sys.stdout = old_stdout
        
        upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
        
        # Upscale alpha channel if present
        if has_alpha:
            new_height, new_width = upscaled_rgb.shape[:2]
            upscaled_alpha = cv2.resize(alpha, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            result = np.dstack([upscaled_rgb, upscaled_alpha])
        else:
            result = upscaled_rgb
        
        return result
        
    except Exception as e:
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

def process_image(input_path, output_path, pbar=None):
    """
    Process a single image to remove background and save as PNG.
    Returns True if successful, False otherwise.
    """
    steps = []
    try:
        # Step 1: Read image
        if pbar:
            pbar.set_description("📖 Reading image")
        image = cv2.imread(input_path)
        if image is None:
            return False, "Could not read image", []
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 2: Detect background
        if pbar:
            pbar.set_description("🔍 Detecting background")
        bg_color = detect_background_color(image_rgb)
        steps.append(f"✓ Background: {bg_color}")
        
        # Step 3: Remove background
        if pbar:
            pbar.set_description("🎨 Removing background")
        
        if bg_color == 'other':
            ai_result, ai_success = remove_background_ai(image_rgb)
            if ai_success:
                result_image = ai_result
                method_used = "AI removal"
                steps.append("✓ AI removal successful")
            else:
                return False, f"Background removal failed", steps
        else:
            ai_result, ai_success = remove_background_ai(image_rgb)
            
            if ai_success:
                result_image = ai_result
                method_used = "AI removal"
                steps.append("✓ AI removal successful")
            else:
                result_image = remove_background_color_based_enhanced(image_rgb, bg_color)
                method_used = "Color-based removal"
                steps.append("✓ Color-based removal")
        
        # Step 4: Add margin
        if pbar:
            pbar.set_description("📏 Adding margin")
        pil_image = Image.fromarray(result_image)
        original_width, original_height = pil_image.size
        
        top_margin = 10
        new_height = original_height + top_margin
        new_image = Image.new('RGBA', (original_width, new_height), (0, 0, 0, 0))
        new_image.paste(pil_image, (0, top_margin), pil_image)
        steps.append(f"✓ Margin added ({top_margin}px)")
        
        # Step 5: Upscale if needed
        if pbar:
            pbar.set_description("⬆️  Checking upscale")
        final_image_array = np.array(new_image)
        height, width = final_image_array.shape[:2]
        current_max = max(height, width)
        
        if current_max < 800:
            if pbar:
                pbar.set_description("⬆️  Upscaling image")
            final_image_array = intelligent_upscale(final_image_array, max_dimension=1000)
            final_height, final_width = final_image_array.shape[:2]
            steps.append(f"✓ Upscaled to {final_width}x{final_height}px")
        else:
            steps.append(f"✓ Size good ({width}x{height}px)")
        
        # Step 6: Save
        if pbar:
            pbar.set_description("💾 Saving image")
        final_pil_image = Image.fromarray(final_image_array)
        final_pil_image.save(output_path, 'PNG', optimize=False, compress_level=1)
        
        if pbar:
            pbar.set_description("✅ Complete")
        
        return True, method_used, steps
        
    except Exception as e:
        steps.append(f"✗ Error: {str(e)}")
        return False, f"Error: {str(e)}", steps

def main():
    """
    Main function to process all images in the input folder.
    """
    input_folder = 'input'
    output_folder = 'output'
    
    os.makedirs(output_folder, exist_ok=True)
    
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Print header
    print("\n" + "="*70)
    print("🎨  BACKGROUND REMOVER & IMAGE UPSCALER".center(70))
    print("="*70)
    
    if not os.path.exists(input_folder):
        print(f"\n❌ Error: Input folder '{input_folder}' does not exist!")
        return
    
    # Get list of files to process
    all_files = os.listdir(input_folder)
    files_to_process = []
    
    for filename in all_files:
        if filename.startswith('.'):
            continue
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension in supported_extensions:
            files_to_process.append(filename)
    
    if not files_to_process:
        print(f"\n❌ No supported images found in '{input_folder}' folder!")
        return
    
    total_files = len(files_to_process)
    print(f"\n🔢 Found {total_files} image(s) to process")
    print("="*70 + "\n")
    
    results = []
    processed_count = 0
    skipped_count = 0
    start_time = time.time()
    processing_times = []
    
    # Process each file with progress bar
    for idx, filename in enumerate(files_to_process, 1):
        input_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        output_filename = base_name + '.png'
        output_path = os.path.join(output_folder, output_filename)
        
        # Calculate overall ETA
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            remaining_files = total_files - idx + 1
            eta_seconds = avg_time * remaining_files
            eta_min = int(eta_seconds // 60)
            eta_sec = int(eta_seconds % 60)
            eta_str = f"{eta_min}m {eta_sec}s" if eta_min > 0 else f"{eta_sec}s"
            print(f"\n[{idx}/{total_files}] 📸 {filename} | ⏱️  Overall ETA: {eta_str}")
        else:
            print(f"\n[{idx}/{total_files}] 📸 {filename}")
        
        print("-" * 70)
        
        with tqdm(total=100, bar_format='{l_bar}{bar}| {elapsed} < {remaining}', 
                  ncols=70, colour='green') as pbar:
            
            file_start_time = time.time()
            success, method, steps = process_image(input_path, output_path, pbar)
            file_end_time = time.time()
            processing_time = file_end_time - file_start_time
            processing_times.append(processing_time)
            
            pbar.update(100)
        
        # Print processing steps
        for step in steps:
            print(f"  {step}")
        
        if success:
            processed_count += 1
            print(f"  ⏱️  Time: {processing_time:.2f}s | Method: {method}")
            print(f"  ✅ SUCCESS")
            results.append(f"✓ {filename} -> {output_filename}: {method}")
        else:
            skipped_count += 1
            print(f"  ❌ FAILED: {method}")
            results.append(f"✗ {filename}: {method}")
    
    # Summary
    total_time = time.time() - start_time
    avg_time = total_time / total_files if total_files > 0 else 0
    
    print("\n" + "="*70)
    print("📊 SUMMARY".center(70))
    print("="*70)
    print(f"✅ Processed:  {processed_count}/{total_files}")
    print(f"❌ Failed:     {skipped_count}/{total_files}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"⚡ Avg/image:  {avg_time:.2f}s")
    print("="*70)
    
    # Save results to file
    with open('output.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("BACKGROUND REMOVAL & UPSCALE RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total processed: {processed_count}\n")
        f.write(f"Total failed:    {skipped_count}\n")
        f.write(f"Total time:      {total_time:.2f}s\n")
        f.write(f"Average time:    {avg_time:.2f}s per image\n\n")
        f.write("Detailed Results:\n")
        f.write("-"*70 + "\n")
        for result in results:
            f.write(result + "\n")
    
    print(f"\n💾 Results saved to: output.txt\n")

if __name__ == "__main__":
    main()
