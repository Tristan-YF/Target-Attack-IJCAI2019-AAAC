from utils.data_process import *
import scipy.stats as st
from tensorflow.contrib.slim.nets import resnet_v1, inception, vgg


def get_adaptive_weight(epoch, success_num):
    """adjust the ensemble weight according to current success attack number"""

    sum_w = tf.exp(-success_num[0]) + tf.exp(-success_num[1]) + tf.exp(-success_num[2])
    adapt_w1 = tf.exp(-success_num[0]) / sum_w
    adapt_w2 = tf.exp(-success_num[1]) / sum_w
    adapt_w3 = tf.exp(-success_num[2]) / sum_w

    w1, w2, w3 = tf.cond(tf.less(epoch, 10),
                         lambda: (1 / 3.0, 1 / 3.0, 1 / 3.0), lambda: (adapt_w1, adapt_w2, adapt_w3))

    return w1, w2, w3


def get_adaptive_alpha(success_num_list, alpha, args):
    """adjust the adaptive alpha for each image"""

    alpha_list=[]
    success_num1, success_num2, success_num3 = success_num_list
    for i in range(args.batch_size):
        alpha_i = (tf.constant(4.0) - success_num1[i] -success_num2[i] - success_num3[i]) * alpha / tf.constant(3.0)
        alpha_i = tf.fill([299, 299, 3], alpha_i)
        alpha_list.append(alpha_i)

    return tf.convert_to_tensor(alpha_list)


def generate_perturbation(x, grad, criterion, args):
    """Generate the perturbation based on gradient information"""

    noise = tf.gradients(criterion, x)[0]

    if args.kernel:
        kernel = gauss_kernel(9, 3).astype(np.float32)
        stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
        stack_kernel = np.expand_dims(stack_kernel, 3)
        noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')

    noise = noise / tf.reshape(
        tf.contrib.keras.backend.std(tf.reshape(noise, [args.batch_size, -1]), axis=1),
        [args.batch_size, 1, 1, 1])

    noise = args.momentum * grad + noise
    perturbation = noise / tf.reshape(
        tf.contrib.keras.backend.std(tf.reshape(noise, [args.batch_size, -1]), axis=1),
        [args.batch_size, 1, 1, 1])

    return perturbation


def get_success_num(y, num_classes, pred_y, logits, args):
    """Calculate the success number of attack"""

    y_one_hot = tf.one_hot(y, num_classes)

    pred1, pred2, pred3 = tf.nn.softmax(logits[0]), tf.nn.softmax(logits[1]), tf.nn.softmax(logits[2])
    pred_y[0], pred_y[1], pred_y[2] = tf.argmax(pred1, 1), tf.argmax(pred2, 1), tf.argmax(pred3, 1)

    temp_success_num = [0] * args.num_model
    success_num = [0] * args.num_model
    total_success_num = tf.constant(0.0)

    for i in range(args.num_model):
        pred_y[i] = tf.cast(pred_y[i], tf.int32)
        pred_y[i] = tf.one_hot(pred_y[i], num_classes)
        temp_success_num[i] = tf.reduce_sum(tf.multiply(pred_y[i], y_one_hot), 1)
        success_num[i] = tf.reduce_sum(temp_success_num[i])
        total_success_num += success_num[i]

    total_success_num = tf.cast(total_success_num, tf.int32)

    return total_success_num, temp_success_num, success_num, y_one_hot


def get_predicted_y(x, num_classes, args):
    """Calculate predicted label"""

    slim = tf.contrib.slim
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits_inc_v1, end_points_inc_v1 = inception.inception_v1(
            dimension_224(x), num_classes=num_classes, is_training=False, scope='InceptionV1')

    image = (((x + 1.0) * 0.5) * 255.0)
    processed_imgs_res_v1_50 = preprocess_for_model(image, 'resnet_v1_50')
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits_res_v1_50, end_points_res_v1_50 = resnet_v1.resnet_v1_50(
            dimension_224(processed_imgs_res_v1_50),
            num_classes=num_classes, is_training=False, scope='resnet_v1_50')

    end_points_res_v1_50['logits'] = tf.squeeze(end_points_res_v1_50['resnet_v1_50/logits'], [1, 2])
    end_points_res_v1_50['probs'] = tf.nn.softmax(end_points_res_v1_50['logits'])

    processed_imgs_vgg_16 = preprocess_for_model(image, 'vgg_16')
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits_vgg_16, end_points_vgg_16 = vgg.vgg_16(
            dimension_224(processed_imgs_vgg_16), num_classes=num_classes, is_training=False, scope='vgg_16')

    end_points_vgg_16['logits'] = end_points_vgg_16['vgg_16/fc8']
    end_points_vgg_16['probs'] = tf.nn.softmax(end_points_vgg_16['logits'])

    logits = [None] * args.num_model
    pred_y = [None] *  args.num_model

    logits[0] = end_points_inc_v1['Logits']
    logits[1] = end_points_res_v1_50['logits']
    logits[2] = end_points_vgg_16['logits']

    for i in range(args.num_model):
        pred_y[i] = tf.argmax(tf.nn.softmax(logits[i]), 1)

    return pred_y, logits


def gauss_kernel(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array"""
    
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()

    return kernel