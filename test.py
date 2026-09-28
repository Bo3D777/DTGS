
from PIL import Image
import os


def get_image_size(image_path):
    """Read an image and return its width and height."""
    
    if not os.path.exists(image_path):
        print(f"Error: file {image_path} does not exist.")
        return None, None

    try:
        
        with Image.open(image_path) as img:
            
            width, height = img.size
            return width, height
    except Exception as e:
        print(f"Failed to read image: {e}")
        return None, None



image_path = "data/D_thermal/scene1/images/000.png"


width, height = get_image_size(image_path)


if width and height:
    print(f"Image width: {width} pixels")
    print(f"Image height: {height} pixels")
    print(f"Image size: {width} × {height}")

