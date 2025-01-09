import os
import numpy as np
from PIL import Image

def read_flo_file(filename):
    """
    Reads a .flo optical flow file and returns it as a numpy array.

    Parameters:
        filename (str): Path to the .flo file.

    Returns:
        np.ndarray: Optical flow data as a (H, W, 2) array.
    """
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError(f"Magic number incorrect in {filename}. Expected 202021.25, got {magic}")
        
        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]
        flow = np.fromfile(f, np.float32, count=2 * width * height)
        flow = flow.reshape((height, width, 2))
    return flow

def flow2img(flow):
    import cv2
    x, y = flow[..., 0], flow[..., 1]
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    ma, an = cv2.cartToPolar(x, y, angleInDegrees=True)
    hsv[..., 0] = (an / 2).astype(np.uint8)
    hsv[..., 1] = (cv2.normalize(ma, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)).astype(np.uint8)
    hsv[..., 2] = 255
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def convert_flo_to_png(input_dir, output_dir):
    import matplotlib.pyplot as plt 
    """
    Converts all .flo files in a directory to .png format.

    Parameters:
        input_dir (str): Path to the directory containing .flo files.
        output_dir (str): Path to the directory where .png files will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.flo'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name.replace('.flo', '.png'))

            try:
                flow = read_flo_file(input_path)
                image = flow2img(flow)
                print(f"shape of flow: {flow.shape}")
                Image.fromarray(image).save(output_path)
                # flow2img(flow)
                plt.imshow(flow2img(flow))
                plt.show()
                # print(f"Converted {file_name} to {output_path}")
            except Exception as e:
                print(f"Failed to convert {file_name}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert .flo optical flow files to .png images.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing .flo files.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory for .png files.")

    args = parser.parse_args()

    convert_flo_to_png(args.input_dir, args.output_dir)
