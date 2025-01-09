from PIL import Image
import os

def reduce_resolution(input_dir, output_dir, target_width, target_height):
    """
    Reduces the resolution of all images in a directory and saves them to an output directory.

    Parameters:
        input_dir (str): Path to the directory containing the high-resolution images.
        output_dir (str): Path to the directory where resized images will be saved.
        target_width (int): Desired width of the resized images.
        target_height (int): Desired height of the resized images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        try:
            with Image.open(input_path) as img:
                img_resized = img.resize((target_width, target_height))
                img_resized.save(output_path)
                print(f"Processed {file_name} -> {output_path}")
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reduce the resolution of images in a directory.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing images.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory for resized images.")
    parser.add_argument("target_width", type=int, help="Target width of the resized images.")
    parser.add_argument("target_height", type=int, help="Target height of the resized images.")

    args = parser.parse_args()

    reduce_resolution(args.input_dir, args.output_dir, args.target_width, args.target_height) # 1000, 600
