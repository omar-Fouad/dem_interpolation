import time
import os
import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument(
    '--flist', default='', type=str,
    help='The filenames of image to be processed: input, mask, output.')
parser.add_argument(
    '--image_height', default=256, type=int,
    help='The height of images should be defined, otherwise batch mode is not'
    ' supported.')
parser.add_argument(
    '--image_width', default=256, type=int,
    help='The width of images should be defined, otherwise batch mode is not'
    ' supported.')
parser.add_argument(
    '--checkpoint_dir', default='', type=str,
    help='The directory of tensorflow checkpoint.')
parser.add_argument(
    '--outlist', default='', type=str,
    help='The directory of putting out image.')

if __name__ == "__main__":
    FLAGS = ng.Config('./inpaint_dem.yml')
    ng.get_gpus(1)
    # os.environ['CUDA_VISIBLE_DEVICES'] =''
    args = parser.parse_args()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = InpaintCAModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, args.image_height, args.image_width*2, 1))
    output = model.build_server_graph(FLAGS, input_image_ph)
    output = tf.reverse(output, [-1])
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    meanMSE = 0
    trueMSE = 0
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            args.checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')

    downsample_rate = 3
    mask = np.zeros([256, 256])
    i = 0
    j = 0
    for i in range(0,255,downsample_rate):
        for j in range(0,255,downsample_rate):
            mask[i,j] = 1
    mask = 1-mask
    mask = mask[np.newaxis, ..., np.newaxis]
    with open(args.flist, 'r') as f:
        lines = f.read().splitlines()
    t = time.time()
    for line in lines:
        image = line.split()
        image = cv2.imread(line, -1)
        filename = line.split('\\')[-1].strip("\n")
        image = cv2.resize(image, (args.image_width, args.image_height))

        h, w = image.shape
        grid = 4
        image = image[:h//grid*grid, :w//grid*grid]
        mask = mask[:h//grid*grid, :w//grid*grid]
        # print('Shape of image: {}'.format(image.shape))

        image = image[np.newaxis,...,np.newaxis]

        assert image.shape == mask.shape

        max = np.max(image, keepdims=True)
        min = np.min(image, keepdims=True)
        image = (image-min) * 2 / (max-min) - 1.
        input_image = np.concatenate([image, mask], axis=2)

        # load pretrained model
        result = sess.run(output, feed_dict={input_image_ph: input_image})
        o1 = Image.fromarray(np.array(result[0, :, :, 0]))
        if args.outlist !='':
            o1.save(os.path.join(args.outlist,filename))

    print('Time total: {}'.format(time.time() - t))
