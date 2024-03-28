import numpy as np
from skimage.transform import resize
from skimage.util import random_noise
from PIL import Image, ImageFont, ImageDraw
import string

def generate_image_data(num_images, image_size=(64, 64), font_path="arial.ttf", font_size=40):
    
    # Initialize an array to hold the generated images
    images = np.zeros((num_images, image_size[1], image_size[0]), dtype=np.float32)
    
    # Load or define a font
    font = ImageFont.truetype(font_path, font_size)
    
    for i in range(num_images):
        # Create a blank image with 'L' mode for grayscale
        img = Image.new('L', (image_size[0], image_size[1]), "white")
        draw = ImageDraw.Draw(img)
        
        # Select a random letter
        letter = np.random.choice(list(string.ascii_uppercase))
        
        # Calculate text position to be centered
        text_width, text_height = draw.textsize(letter, font=font)
        text_x = (image_size[0] - text_width) / 2
        text_y = (image_size[1] - text_height) / 2
        
        # Draw the letter on the image
        draw.text((text_x, text_y), letter, fill="black", font=font)
        
        # Convert PIL image to a numpy array and normalize it
        img_array = np.array(img) / 255.0
        
        img_array = resize(img_array, image_size, anti_aliasing=True)
        
        # Add random noise to the image
        img_array_noisy = random_noise(img_array, mode='gaussian', var=0.01)
        
        images[i] = img_array_noisy
        
    return images
