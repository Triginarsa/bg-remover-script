# Background Remover Script

This Python script automatically removes white or black backgrounds from images and converts them to transparent PNG files.

## Requirements

- Python 3.12+

## Installation

Install dependencies using a virtual environment and the provided `requirements.txt`:

```bash
# from the project root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

1. **Place your images** in the `input/` folder

   - Supported formats: JPG, JPEG, PNG, BMP, TIFF, WEBP

2. **Run the script**:

   ```bash
   python main.py
   ```

3. **Check results**:
   - Processed images will be saved in the `output/` folder as PNG files
   - A detailed report will be generated in `output.txt`

## How it Works

1. **Background Detection**:

   - Samples pixels from image edges (corners and midpoints)
   - Determines if background is predominantly white, black, or other
   - Uses a tolerance of 30 RGB values for color detection

2. **Background Removal**:

   - For white backgrounds: Makes white pixels transparent
   - For black backgrounds: Makes black pixels transparent
   - Uses tolerance to handle slight color variations

3. **File Processing**:
   - Keeps original filename but changes extension to `.png`
   - Skips files with backgrounds that aren't white or black
   - Handles errors gracefully and reports them

## Output

- **Processed Images**: Saved in `output/` folder with transparent backgrounds
- **Results Report**: `output.txt` contains:
  - Processing summary (total processed, skipped, errors)
  - Detailed file-by-file results
  - Background color detection results
  - Processing timestamp
