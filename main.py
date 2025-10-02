import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from rembg import remove
from scipy import ndimage
from scipy.ndimage import gaussian_filter

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
        
        pil_image = Image.fromarray(result_image, 'RGBA')
        
        pil_image.save(output_path, 'PNG', optimize=False, compress_level=1)
        
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
        f.write("Features used:\n")
        f.write("- AI-based background removal (rembg)\n")
        f.write("- Enhanced color-based fallback method\n")
        f.write("- Advanced edge smoothing and anti-aliasing\n")
        f.write("- Multi-color space analysis (HSV + LAB)\n")
        f.write("- Morphological operations for noise reduction\n\n")
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
    print("\nThe new version provides:")
    print("• AI-powered background removal for superior quality")
    print("• Enhanced edge smoothing and anti-aliasing")
    print("• Better color detection using multiple color spaces")
    print("• Advanced morphological operations")
    print("• Smooth gradients at object edges")

if __name__ == "__main__":
    main()
