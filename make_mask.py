import os
from PIL import Image


def generate_black_images(input_dir, output_dir):
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    
    os.makedirs(output_dir, exist_ok=True)

    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_extensions):
            input_path = os.path.join(input_dir, filename)

            
            with Image.open(input_path) as img:
                
                black_img = Image.new('L', img.size, 0)  

                
                output_path = os.path.join(output_dir, filename)
                black_img.save(output_path)

    print("Black single-channel images generated.")


if __name__ == '__main__':
    input_folder = 'data/D_thermal/scene31/images'  
    output_folder = 'data/D_thermal/scene31/masks'  
    generate_black_images(input_folder, output_folder)

