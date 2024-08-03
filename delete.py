import cv2
import os

def resize_images(input_folder, output_folder, size=(64, 64)):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    for filename in os.listdir(input_folder):
        # Construct full file path
        file_path = os.path.join(input_folder, filename)
        
        # Check if it's a file (and not a directory)
        if os.path.isfile(file_path):
            # Read the image
            image = cv2.imread(file_path)
            
            if image is None:
                print(f"Failed to load image {file_path}")
                continue
            
            # Resize the image
            resized_image = cv2.resize(image, size)
            
            # Save the resized image to the output folder
            output_file_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_file_path, resized_image)

# Example usage
input_folder = './data/kaggle/kaggle/train'
output_folder = './data/kaggle/kaggle/train4'
resize_images(input_folder, output_folder)
