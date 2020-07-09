import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from scipy.misc import imread
from scipy.misc import imresize
slim = tf.contrib.slim


def dimension_224(input_tensor):
    input_tensor = tf.image.resize_bilinear(input_tensor, [224, 224], align_corners=False)

    return input_tensor


def preprocess_for_model(images, model_type):
    if 'inception' in model_type.lower():
        images = tf.image.resize_bilinear(images, [299, 299], align_corners=False)
        # tensor-scalar operation
        images = (images / 255.0) * 2.0 - 1.0

        return images

    if 'resnet' in model_type.lower() or 'vgg' in model_type.lower():
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        images = tf.image.resize_bilinear(images, [299, 299], align_corners=False)
        tmp_0 = images[:, :, :, 0] - _R_MEAN
        tmp_1 = images[:, :, :, 1] - _G_MEAN
        tmp_2 = images[:, :, :, 2] - _B_MEAN
        images = tf.stack([tmp_0, tmp_1, tmp_2], 3)

        return images


def load_images_with_target_label(input_dir, args):
    images = []
    filenames = []
    target_labels = []
    idx = 0

    dev = pd.read_csv(os.path.join(input_dir, 'dev.csv'))
    filename2label = {dev.iloc[i]['filename']: dev.iloc[i]['targetedLabel'] for i in range(len(dev))}
    for filename in filename2label.keys():
        image = imread(os.path.join(input_dir, filename), mode='RGB')
        images.append(image)
        filenames.append(filename)
        target_labels.append(filename2label[filename])
        idx += 1
        if idx == args.batch_size:
            images = np.array(images)
            yield filenames, images, target_labels
            filenames = []
            images = []
            target_labels = []
            idx = 0
    if idx > 0:
        images = np.array(images)
        yield filenames, images, target_labels


def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        image = (((images[i] + 1.0) * 0.5) * 255.0).astype(np.uint8)
        image = imresize(image, [299, 299])
        Image.fromarray(image).save(os.path.join(output_dir, filename), format='PNG')


def check_or_create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)