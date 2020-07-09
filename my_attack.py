import argparse
from utils.score_evaluation import get_coarse_score
from utils.data_process import *
from utils.untils_fun import get_adaptive_alpha, get_adaptive_weight, get_predicted_y, get_success_num, generate_perturbation
import time


def target_graph(x, y, epoch, x_max, x_min, grad, success_num, losses):

    eps = 2.0 * args.max_epsilon / 255.0
    alpha = eps / args.num_iter
    num_classes = args.num_classes

    pred_y, logits = get_predicted_y(x, num_classes, args)
    
    total_success_num, success_num_list, success_num, y_one_hot = get_success_num(y, num_classes, pred_y, logits, args)

    # adjust the alpha according to current success attack number
    alpha = tf.cond(tf.greater(total_success_num, 10) & tf.less(total_success_num, 20),
                    lambda: alpha / 2.0, lambda: alpha)
    alpha = tf.cond(tf.greater(total_success_num, 20) & tf.less(total_success_num, 25),
                    lambda: alpha / 3.0, lambda: alpha)
    alpha = tf.cond(tf.greater(total_success_num, 25) & tf.less(total_success_num, 30),
                    lambda: alpha / 4.0 , lambda: alpha)

    # calculate different alpha for each image
    adaptive_alpha = get_adaptive_alpha(success_num_list, alpha, args)

    # adjust the ensemble weight according to current success attack number
    w1, w2, w3 = get_adaptive_weight(epoch, success_num)

    final_logits = w1 * logits[0] + w2 * logits[1] + w3 * logits[2]

    losses = tf.losses.softmax_cross_entropy(y_one_hot, final_logits, label_smoothing=0.0, weights=1.0)

    perturbation = generate_perturbation(x, grad, losses, args)

    # Generate adversarial sample
    x = x - adaptive_alpha * tf.clip_by_value(tf.round(perturbation), -2, 2)
    x = tf.clip_by_value(x, x_min, x_max)

    epoch = tf.add(epoch, 1)

    return x, y, epoch, x_max, x_min, perturbation, total_success_num, losses


def stop(x, y, epoch, x_max, x_min, grad, success_num, losses):
    return tf.less(epoch, args.num_iter)


# Momentum Iterative FGSM
def my_target_attack(input_dir, output_dir, args):
    # some parameter
    eps = 2.0 * args.max_epsilon / 255.0
    batch_shape = [args.batch_size, 299, 299, 3]
    check_or_create_dir(output_dir)

    #main programe
    with tf.Graph().as_default():

        # Prepare graph
        raw_inputs = tf.placeholder(tf.uint8, shape=[None, 299, 299, 3])

        # preprocessing for model input,
        # note that images for all classifier will be normalized to be in [-1, 1]
        processed_imgs = preprocess_for_model(raw_inputs, 'inception_v1')

        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        y = tf.placeholder(tf.int32, shape=[args.batch_size])
        epoch = tf.constant(0)
        losses=tf.constant(0.0)
        success_num=tf.constant(0)
        grad = tf.zeros(shape=batch_shape)

        x_adv, _, _, _, _, _, success_num, losses = tf.while_loop(stop, target_graph,
                                                              [x_input, y, epoch, x_max, x_min, grad,success_num,losses])
        slim = tf.contrib.slim
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))

        with tf.Session() as sess:
            s1.restore(sess, model_checkpoint_map['inception_v1'])
            s2.restore(sess, model_checkpoint_map['resnet_v1_50'])
            s3.restore(sess, model_checkpoint_map['vgg_16'])

            success_nums_list=[]

            for filenames, raw_images, target_labels in load_images_with_target_label(input_dir, args):

                processed_imgs_ = sess.run(processed_imgs, feed_dict={raw_inputs: raw_images})
                adv_images, total_success_num, losses_= sess.run([x_adv,success_num,losses],
                                                                feed_dict={x_input: processed_imgs_, y: target_labels})
                success_nums_list.append(total_success_num)
                save_images(adv_images, filenames, output_dir)
                print("Loss: {:.10f} success num: {}/{}".format(losses_, total_success_num, 3 * args.batch_size))

            acc = np.mean(success_nums_list) / (3.0 * args.batch_size)

            print("total_avarge_accuracy: ", acc)
            print("total_successect:", np.sum(success_nums_list))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default = "./input/", type=str, help='input dir')
    parser.add_argument('--output_dir', default="./output", type=str, help='output dir')
    parser.add_argument('--gpu', default= 0, type=int, help='GPU')
    parser.add_argument('--momentum', default=1.0, type=float, help='Momentum')
    parser.add_argument('--batch_size', default=11, type=int, help='BatchSize')
    parser.add_argument('--num_iter', default=120, type=int, help='the number of iterations')
    parser.add_argument('--max_epsilon', default=16.0, type=float, help='epsilon')
    parser.add_argument('--kernel', default=1, type=int, help='whether to use Gaussian kernel  0/1')
    parser.add_argument('--num_model', default=3, type=int, help='the number of ensemble models')
    parser.add_argument('--num_classes', default=110, type=int, help='the number of classes')

    CHECKPOINTS_DIR = './checkpoints/'
    model_checkpoint_map = {
        'inception_v1': os.path.join(CHECKPOINTS_DIR, 'inception_v1', 'inception_v1.ckpt'),
        'resnet_v1_50': os.path.join(CHECKPOINTS_DIR, 'resnet_v1_50', 'model.ckpt-49800'),
        'vgg_16': os.path.join(CHECKPOINTS_DIR, 'vgg_16', 'vgg_16.ckpt')}

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    time_start = time.time()

    # generate adversarial sample
    my_target_attack(input_dir, output_dir, args)

    # evaluation
    get_coarse_score(input_dir,output_dir)

    print("Total_time: {:.02f} s".format(time.time()-time_start))