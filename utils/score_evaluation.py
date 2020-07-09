import os
from PIL import Image
import numpy as np


def has_file_allowed_extension(filename):
    extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def get_coarse_score(input_dir,output_dir):
    for root, par, fnames in sorted(os.walk(input_dir)):
        scores=[]
        for fname in sorted(fnames):

            if not has_file_allowed_extension(fname):
                continue

            path_input = os.path.join(input_dir, fname)
            img_input = Image.open(path_input)
            img_input = np.array(img_input)

            path_output = os.path.join(output_dir, fname)
            img_output = Image.open(path_output)
            img_output = np.array(img_output)

            delta_img= img_output.reshape(-1, 3) - img_input.reshape(-1,3)

            score=np.mean(np.sqrt(np.sum((delta_img ** 2), axis=1)))

            scores.append(score)
        print("Average_score: {:.4f}".format(np.mean(np.array(scores))))
    return np.mean(np.array(scores))