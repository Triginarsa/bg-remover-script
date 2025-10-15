# Background Remover & Image Upscaler

Automatically remove backgrounds from images and intelligently upscale them to high quality.

## Features

- 🎨 **Background Removal** - Uses rembg for clean, natural edges
- ⬆️ **Smart Image Upscaling** - Real-ESRGAN upscaling for non-pixelated results
- 📏 **Auto Margin** - Adds 10px top margin to all processed images
- 🚀 **Batch Processing** - Process multiple images at once

## Requirements

- Python 3.12+
- macOS, Linux, or Windows

## Installation

1. **Clone or download this repository**

2. **Create and activate virtual environment**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

1. **Place your images** in the `input/` folder

   - Supported formats: JPG, JPEG, PNG, BMP, TIFF

2. **Run the script**:

   ```bash
   python main.py
   ```

3. **Find your processed images** in the `output/` folder
   - All images saved as PNG with transparent backgrounds
   - Original filenames preserved
   - Processing report saved to `output.txt`

## How It Works

1. **Background Detection** - Automatically detects white, black, or other backgrounds
2. **AI Removal** - Uses rembg model for clean background removal
3. **Add Margin** - Adds 10px margin at the top
4. **Smart Upscale** - Upscales images smaller than 800px (max 1000px) using Real-ESRGAN
5. **Save PNG** - Outputs transparent PNG files

## Upscaling Behavior

Images are upscaled conservatively to maintain natural photo quality:

- **< 400px**: Upscaled 2x
- **400-600px**: Upscaled 1.5x
- **600-800px**: Minimal upscale to ~850px
- **≥ 800px**: No upscaling (already good size)

## Notes

- The first run will download AI models (may take a moment)
- Larger images take longer to process
- All output files are saved as PNG for transparency support
- Original images in `input/` are never modified
