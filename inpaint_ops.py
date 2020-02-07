import logging
import math

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from PIL import Image, ImageDraw

from neuralgym.ops.layers import resize
from neuralgym.ops.layers import *
from neuralgym.ops.loss_ops import *
from neuralgym.ops.gan_ops import *
from neuralgym.ops.summary_ops import *
import matplotlib.pyplot as plt


logger = logging.getLogger()
np.random.seed(2018)


@add_arg_scope
def gen_conv(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation=tf.nn.elu, training=True):
    """Define conv for generator.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name)
    if cnum == 3 or activation is None:
        # conv for output
        return x
    x, y = tf.split(x, 2, 3)
    x = activation(x)
    y = tf.nn.sigmoid(y)
    x = x * y
    return x


@add_arg_scope
def gen_deconv(x, cnum, name='upsample', padding='SAME', training=True):
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.

    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_nearest_neighbor)
        x = gen_conv(
            x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training)
    return x


@add_arg_scope
def dis_conv(x, cnum, ksize=5, stride=2, name='conv', training=True):
    """Define conv for discriminator.
    Activation is set to leaky_relu.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    x = conv2d_spectral_norm(x, cnum, ksize, stride, 'SAME', name=name)
    x = tf.nn.leaky_relu(x)
    return x


def random_bbox(FLAGS):
    """Generate a random tlhw.

    Returns:
        tuple: (top, left, height, width)

    """
    img_shape = FLAGS.img_shapes
    img_height = img_shape[0]
    img_width = img_shape[1]
    maxt = img_height - FLAGS.vertical_margin - FLAGS.height
    maxl = img_width - FLAGS.horizontal_margin - FLAGS.width
    t = tf.random_uniform(
        [], minval=FLAGS.vertical_margin, maxval=maxt, dtype=tf.int32)
    l = tf.random_uniform(
        [], minval=FLAGS.horizontal_margin, maxval=maxl, dtype=tf.int32)
    h = tf.constant(FLAGS.height)
    w = tf.constant(FLAGS.width)
    return (t, l, h, w)


def bbox2mask(FLAGS, bbox, name='mask'):
    """Generate mask tensor from bbox.

    Args:
        bbox: tuple, (top, left, height, width)

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """
    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((1, height, width, 1), np.float32)
        h = np.random.randint(delta_h//2+1)
        w = np.random.randint(delta_w//2+1)
        mask[:, bbox[0]+h:bbox[0]+bbox[2]-h,
             bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
        return mask
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img_shape = FLAGS.img_shapes
        height = img_shape[0]
        width = img_shape[1]
        mask = tf.py_func(
            npmask,
            [bbox, height, width,
             FLAGS.max_delta_height, FLAGS.max_delta_width],
            tf.float32, stateful=False)
        mask.set_shape([1] + [height, width] + [1])
    return mask


def brush_stroke_mask(FLAGS, name='mask'):
    """Generate mask tensor from bbox.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """
    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 40
    def generate_mask(H, W):
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('L', (W, H), 0)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.float32)
        mask = np.reshape(mask, (1, H, W, 1))
        return mask
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img_shape = FLAGS.img_shapes
        height = img_shape[0]
        width = img_shape[1]
        mask = tf.py_func(
            generate_mask,
            [height, width],
            tf.float32, stateful=True)
        mask.set_shape([1] + [height, width] + [1])
    return mask


def local_patch(x, bbox):
    """Crop local patch according to bbox.

    Args:
        x: input
        bbox: (top, left, height, width)

    Returns:
        tf.Tensor: local patch

    """
    x = tf.image.crop_to_bounding_box(x, bbox[0], bbox[1], bbox[2], bbox[3])
    return x


def resize_mask_like(mask, x):
    """Resize mask like shape of x.

    Args:
        mask: Original mask.
        x: To shape of x.

    Returns:
        tf.Tensor: resized mask

    """
    mask_resize = resize(
        mask, to_shape=x.get_shape().as_list()[1:3],
        func=tf.image.resize_nearest_neighbor)
    return mask_resize


def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True):
    """ Contextual attention layer implementation.

    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        x: Input feature to match (foreground).
        t: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.

    Returns:
        tf.Tensor: output

    """
    # get shapes
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    # extract patches from background with stride and rate
    kernel = 2*rate
    raw_w = tf.extract_image_patches(
        b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    f = resize(f, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    b = resize(b, to_shape=[int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)], func=tf.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651
    if mask is not None:
        mask = resize(mask, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0], axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.extract_image_patches(
        b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    m = tf.extract_image_patches(
        mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0,1,2], keep_dims=True), 0.), tf.float32)
    w_groups = tf.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
    y = []
    offsets = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

        # softmax to match
        yi *=  mm  # mask
        yi = tf.nn.softmax(yi*scale, 3)
        yi *=  mm  # mask

        offset = tf.argmax(yi, axis=3, output_type=tf.int32)
        offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
        y.append(yi)
        offsets.append(offset)
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    offsets = tf.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])
    # case1: visualize optical flow: minus current position
    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.concat([h_add, w_add], axis=3)
    # to flow image
    flow = flow_to_image_tf(offsets)
    # # case2: visualize which pixels are attended
    # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
    if rate != 1:
        flow = resize(flow, scale=rate, func=tf.image.resize_bilinear)
    flow = flow[:, :, :, 0:1]
    return y, flow
# def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
#                          fuse_k=3, softmax_scale=10., training=True, fuse=True):
#     """ Contextual attention layer implementation.
#
#     Contextual attention is first introduced in publication:
#         Generative Image Inpainting with Contextual Attention, Yu et al.
#
#     Args:
#         x: Input feature to match (foreground).
#         t: Input feature for match (background).
#         mask: Input mask for t, indicating patches not available.
#         ksize: Kernel size for contextual attention.
#         stride: Stride for extracting patches from t.
#         rate: Dilation for matching.
#         softmax_scale: Scaled softmax for attention.
#         training: Indicating if current graph is training or inference.
#
#     Returns:
#         tf.Tensor: output
#
#     """
#
#     fig = plt.figure()
#     fig1 = plt.figure()
#     # get shapes
#     raw_fs = tf.shape(f)
#     raw_int_fs = f.get_shape().as_list()
#     raw_int_bs = b.get_shape().as_list()
#     # extract patches from background with stride and rate
#     kernel = 2*rate
#
#     # images: 需要分成patch的原始图像。可以是4 - D张量（batch_size, weight, high, channels）
#     # ksize: 一般是1 - D张量，length大于等于4，表示patch的大小，比如5 * 5, 则为（1，5，5，1）
#     # strides表示一个补丁的开始与原始图像中下一个连续补丁的开始之间的间隙的长度
#     # strides：类似于tf.conv2d里面的strides，表示步长，也就是一次滑动几个像素，如步长为2，则为（1，2，2，1）
#     # padding： 填充方式，类似于tf.conv2d
#     # rates: 是一个不容易理解的量，看了很久才搞明白。类似于空洞卷积，同一个patch里面隔几个点取为有效点
#
#     # 因为b输入是192x64x64,这里的输出应该是192x(64/2)x(64/2x4x4)为192x32x(32x16)
#     raw_w = tf.extract_image_patches(
#         b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
#     # 16,-1,4,4,1,这里的理解就是得到了192张图片的n个4x4的小patch
#     raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
#     # 16,4,4,1,-1,
#     raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
#     # downscaling foreground option: downscaling both foreground and
#     # background for matching and use original background for reconstruction.
#     #2倍降采样
#     f = resize(f, scale=1./rate, func=tf.image.resize_nearest_neighbor)
#     b = resize(b, to_shape=[int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)], func=tf.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651
#
#     if mask is not None:
#         mask = resize(mask, scale=1./rate, func=tf.image.resize_nearest_neighbor)
#     fs = tf.shape(f)
#     int_fs = f.get_shape().as_list()
#     f_groups = tf.split(f, int_fs[0], axis=0)
#     # from t(H*W*C) to w(b*k*k*c*h*w)
#     bs = tf.shape(b)
#     int_bs = b.get_shape().as_list()
#     # ksize=3,这里的b是用rate降采样的
#     w = tf.extract_image_patches(
#         b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
#     w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
#     w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
#
#     ax = fig1.add_subplot(224)
#     tmp = mask[0, :, :, 0:1]
#     max = tf.reduce_max(tmp)
#     min = tf.reduce_min(tmp)
#     # 内存溢出跑不出来
#     tmp = (tf.Session().run((tmp - min) / (max - min)) * 255)
#     tmp = np.array((tmp)).astype(np.uint8)
#     tmp = np.repeat(tmp, 3, axis=2)
#     ax.imshow(tmp)
#     ax.set_title('original_mask')
#     # process mask
#     if mask is None:
#         mask = tf.zeros([1, bs[1], bs[2], 1])
#     m = tf.extract_image_patches(
#         mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
#     m = tf.reshape(m, [1, -1, ksize, ksize, 1])
#     m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
#     m = m[0]
#     # mm = tf.reduce_mean(m, axis=[0,1,2], keep_dims=True)
#     mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0,1,2], keep_dims=True), 0.), tf.float32)
#     w_groups = tf.split(w, int_bs[0], axis=0)
#     raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
#     y = []
#     offsets = []
#     # fuse_k=3
#     k = fuse_k
#     scale = softmax_scale
#     fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
#     # w = tf.extract_image_patches(
#     #         b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
#     for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):# 192
#         # conv for compare
#         wi = wi[0]
#         # 相当于是对patch平方再求和再开方，再与1e-4相比较，取最大，for循环就是对每个batch的意思
#         # l2
#         wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
#         # wi_normed是卷积核，shape为 [ filter_height, filter_weight, in_channel, out_channels ]
#         # k*k*c*hw 所以yi是被卷积成前面的patch个数的通道，相当于是在192的1的基础上又做了一次通道切分一样
#         # hw=128*128
#
#         ax = fig.add_subplot(441)
#         # tmp=np.repeat(tf.Session().run(xi)[0],3,axis=2)
#         # tmp = np.repeat(tmp, 3, axis=2)
#         # 必须要转成uint8才能正常显示
#         tmp = np.repeat(np.array((tf.Session().run(xi)[0]+1)*127.5).astype(np.uint8), 3, axis=2)
#         ax.imshow(tmp)
#         ax.set_title('original')
#
#         yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")
#         ax = fig.add_subplot(442)
#         tmp = yi[0,:,:,0:1]
#         max = tf.reduce_max(tmp)
#         min = tf.reduce_min(tmp)
#         # 内存溢出跑不出来
#         tmp = (tf.Session().run((tmp-min)/(max-min))*255)
#         tmp = np.array((tmp)).astype(np.uint8)
#         tmp = np.repeat(tmp, 3, axis=2)
#         ax.imshow(tmp)
#         ax.set_title('yi')
#
#
#         # conv implementation for fuse scores to encourage large patches
#         if fuse:# 目前由于fuse_weight为1.相当于是直接求均值，所以看起来这里主要是转置又转置，是transpose_conv
#             # yi [128,128,128*128]
#             yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
#             ax = fig.add_subplot(443)
#             tmp = yi[0, :, :, 0:1]
#             max = tf.reduce_max(tmp)
#             min = tf.reduce_min(tmp)
#             # 内存溢出跑不出来
#             tmp = (tf.Session().run((tmp - min) / (max - min)) * 255)
#             tmp = np.array((tmp)).astype(np.uint8)
#             tmp = np.repeat(tmp, 3, axis=2)
#             ax.imshow(tmp)
#             ax.set_title('yi_big')
#             # fuse_weight为1
#             yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
#             # 相当于是行列做交换，然后一组是128，分了128个小组，4,3交换的意思就是把小组里面的取出来分别放到128个组中，形成新的128组
#
#             yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
#             ax = fig.add_subplot(444)
#             # ax.imshow(yi[0, :, :, 0:1,0])
#             tmp = yi[0, :, :, 0:1,0]
#             max = tf.reduce_max(tmp)
#             min = tf.reduce_min(tmp)
#             # 内存溢出跑不出来
#             tmp = (tf.Session().run((tmp - min) / (max - min)) * 255)
#             tmp = np.array((tmp)).astype(np.uint8)
#             tmp = np.repeat(tmp, 3, axis=2)
#             ax.imshow(tmp)
#             ax.set_title('yi_fuse1')
#             # 这里有点像转置操作
#             yi = tf.transpose(yi, [0, 2, 1, 4, 3])
#             ax = fig.add_subplot(445)
#             # ax.imshow(yi[0,:,:,0:1,0])
#             tmp = yi[0,:,:,0:1,0]
#             max = tf.reduce_max(tmp)
#             min = tf.reduce_min(tmp)
#             # 内存溢出跑不出来
#             tmp = (tf.Session().run((tmp - min) / (max - min)) * 255)
#             tmp = np.array((tmp)).astype(np.uint8)
#             tmp = np.repeat(tmp, 3, axis=2)
#             ax.imshow(tmp)
#             ax.set_title('yi_fusetranspose')
#             # 这里的reshape就是伸展开，不过因为前面的转置操作，现在这里行列互换，128与128重新排列，图像看来应该会很不一样
#             yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
#             yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
#
#             yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
#             ax = fig.add_subplot(446)
#             tmp = yi[0, :, :, 0:1, 0]
#             max = tf.reduce_max(tmp)
#             min = tf.reduce_min(tmp)
#             # 内存溢出跑不出来
#             tmp = (tf.Session().run((tmp - min) / (max - min)) * 255)
#             tmp = np.array((tmp)).astype(np.uint8)
#             tmp = np.repeat(tmp, 3, axis=2)
#             ax.imshow(tmp)
#             ax.set_title('yi_fuse2')
#             yi = tf.transpose(yi, [0, 2, 1, 4, 3])
#             # ax.set_title('yi_fusetransposefusetranspose')
#         yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])
#         ax = fig.add_subplot(447)
#         tmp = yi[0, :, :, 0:1]
#         max = tf.reduce_max(tmp)
#         min = tf.reduce_min(tmp)
#         # 内存溢出跑不出来
#         tmp = (tf.Session().run((tmp - min) / (max - min)) * 255)
#         tmp = np.array((tmp)).astype(np.uint8)
#         tmp = np.repeat(tmp, 3, axis=2)
#         ax.imshow(tmp)
#         ax.set_title('yi_128x128个')
#
#         # softmax to match
#         # 加掩模
#         yi *=  mm  # mask
#         ax = fig.add_subplot(448)
#         # ax.imshow(yi[0, :, :, 0:1])
#         tmp = yi[0, :, :, 0:1]
#         max = tf.reduce_max(tmp)
#         min = tf.reduce_min(tmp)
#         # 内存溢出跑不出来
#         tmp = (tf.Session().run((tmp - min) / (max - min)) * 255)
#         tmp = np.array((tmp)).astype(np.uint8)
#         tmp = np.repeat(tmp, 3, axis=2)
#         ax.imshow(tmp)
#         ax.set_title('yi_withmask')
#         ax = fig.add_subplot(449)
#         # ax.imshow(yi[0, :, :, 0:1])
#         tmp = mm[0, :, :, 0:1]
#         max = tf.reduce_max(tmp)
#         min = tf.reduce_min(tmp)
#         # 内存溢出跑不出来
#         tmp = (tf.Session().run((tmp - min) / (max - min)) * 255)
#         tmp = np.array((tmp)).astype(np.uint8)
#         tmp = np.repeat(tmp, 3, axis=2)
#         ax.imshow(tmp)
#         ax.set_title('mask')
#         # softmax有放大较大值的作用，因为有exp
#         # 这里是在axis=3上面求和，相当于是在在通道上面归一化到（0,1）
#         yi = tf.nn.softmax(yi*scale, 3)
#
#         ax = fig1.add_subplot(221)
#         # ax.imshow(yi[0, :, :, 0:1])
#         tmp = yi[0, :, :, 0:1]
#         max = tf.reduce_max(tmp)
#         min = tf.reduce_min(tmp)
#         # 内存溢出跑不出来
#         tmp = (tf.Session().run((tmp - min) / (max - min)) * 255)
#         tmp = np.array((tmp)).astype(np.uint8)
#         tmp = np.repeat(tmp, 3, axis=2)
#         ax.imshow(tmp)
#         ax.set_title('yi_withmasksoftmax')
#
#         yi *=  mm  # mask
#         ax = fig1.add_subplot(222)
#         # ax.imshow(yi[0, :, :, 0:1])
#         tmp = yi[0, :, :, 0:1]
#         max = tf.reduce_max(tmp)
#         min = tf.reduce_min(tmp)
#         # 内存溢出跑不出来
#         tmp = (tf.Session().run((tmp - min) / (max - min)) * 255)
#         tmp = np.array((tmp)).astype(np.uint8)
#         tmp = np.repeat(tmp, 3, axis=2)
#         ax.imshow(tmp)
#         ax.set_title('yi_withmasksoftmaxmask')
#         # 找通道上面最大的yi，其实就是返回每个通道上的二维矩阵的行方向上的最大值位置索引
#         offset = tf.argmax(yi, axis=3, output_type=tf.int32)
#         offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
#         # deconv for patch pasting
#         # 3.1 paste center
#         # 然后又用原图的patch做卷积，这样一来又恢复到了原来的通道数
#         wi_center = raw_wi[0]
#         # deconvolution
#         yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
#         ax = fig1.add_subplot(223)
#         # ax.imshow(yi[0, :, :, 0:1])
#         tmp = yi[0, :, :, 0:1]
#         max = tf.reduce_max(tmp)
#         min = tf.reduce_min(tmp)
#         # 内存溢出跑不出来
#         tmp = (tf.Session().run((tmp - min) / (max - min)) * 255)
#         tmp = np.array((tmp)).astype(np.uint8)
#         tmp = np.repeat(tmp, 3, axis=2)
#         ax.imshow(tmp)
#         ax.set_title('yi_deconvo')
#         y.append(yi)
#         offsets.append(offset)
#     # 这里恢复到了输入的通道数
#     y = tf.concat(y, axis=0)
#     y.set_shape(raw_int_fs)
#     offsets = tf.concat(offsets, axis=0)
#     offsets.set_shape(int_bs[:3] + [2])
#     # case1: visualize optical flow: minus current position
#     # 扩张成128*128，包含的是0-127这些数
#     # offset是索引的意思
#     h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
#     w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
#     # 这里第2维第3维都不一样是怎么连接的
#     offsets = offsets - tf.concat([h_add, w_add], axis=3)
#     # to flow image
#     flow = flow_to_image_tf(offsets)
#     # # case2: visualize which pixels are attended
#     # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
#     if rate != 1:
#         # 之前得到的flow是降采样的
#         flow = resize(flow, scale=rate, func=tf.image.resize_bilinear)
#     fig.show()
#     fig1.show()
#     return y, flow
#
# @add_arg_scope
# def gate_conv(x_in, cnum, ksize, stride=1, rate=1, name='conv',
#              padding='SAME', activation='leaky_relu', use_lrn=True,training=True):
#     assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
#     if padding == 'SYMMETRIC' or padding == 'REFELECT':
#         p = int(rate*(ksize-1)/2)
#         x = tf.pad(x_in, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
#         padding = 'VALID'
#     x = tf.layers.conv2d(
#         x_in, cnum, ksize, stride, dilation_rate=rate,
#         activation=None, padding=padding, name=name)
#     if use_lrn:
#         x = tf.nn.lrn(x, bias=0.00005)
#     if activation=='leaky_relu':
#         x = tf.nn.leaky_relu(x)
#
#     g = tf.layers.conv2d(
#         x_in, cnum, ksize, stride, dilation_rate=rate,
#         activation=tf.nn.sigmoid, padding=padding, name=name+'_g')
#
#     x = tf.multiply(x,g)
#     return x, g
#
# @add_arg_scope
# def gate_deconv(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
#        name="deconv", training=True):
#     with tf.variable_scope(name):
#         # filter : [height, width, output_channels, in_channels]
#         w1 = tf.get_variable('w1', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
#                   initializer=tf.random_normal_initializer(stddev=stddev))
#
#         deconv = tf.nn.conv2d_transpose(input_, w1, output_shape=output_shape,
#                     strides=[1, d_h, d_w, 1])
#
#         biases = tf.get_variable('biases1', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
#         deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
#         deconv = tf.nn.leaky_relu(deconv)
#
#         w2 = tf.get_variable('w2', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
#                              initializer=tf.random_normal_initializer(stddev=stddev))
#         g = tf.nn.conv2d_transpose(input_, w2, output_shape=output_shape,
#                     strides=[1, d_h, d_w, 1])
#         b = tf.get_variable('biases2', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
#         g = tf.reshape(tf.nn.bias_add(g, b), deconv.get_shape())
#         g = tf.nn.sigmoid(g)
#
#         deconv = tf.multiply(g,deconv)
#
#         return deconv, g

@add_arg_scope
def gate_conv(x_in, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation='leaky_relu', use_lrn=True,training=True):
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x_in, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    x = tf.layers.conv2d(
        x_in, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name)
    if use_lrn:
        x = tf.nn.lrn(x, bias=0.00005)
    if activation=='leaky_relu':
        x = tf.nn.leaky_relu(x)

    # g = tf.layers.conv2d(
    #     x_in, cnum, ksize, stride, dilation_rate=rate,
    #     activation=tf.nn.sigmoid, padding=padding, name=name+'_g')
    #
    # x = tf.multiply(x,g)
    return x#, g

@add_arg_scope
def gate_deconv(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv", training=True):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w1 = tf.get_variable('w1', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w1, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases1', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.leaky_relu(deconv)

        # w2 = tf.get_variable('w2', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
        #                      initializer=tf.random_normal_initializer(stddev=stddev))
        # g = tf.nn.conv2d_transpose(input_, w2, output_shape=output_shape,
        #             strides=[1, d_h, d_w, 1])
        # b = tf.get_variable('biases2', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        # g = tf.reshape(tf.nn.bias_add(g, b), deconv.get_shape())
        # g = tf.nn.sigmoid(g)
        #
        # deconv = tf.multiply(g,deconv)

        return deconv#, g
def test_contextual_attention(args):
    """Test contextual attention layer with 3-channel image input
    (instead of n-channel feature).

    """
    import cv2
    import os
    # run on cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    rate = 2
    stride = 1
    grid = rate*stride

    b = cv2.imread(args.imageA)
    b = cv2.resize(b, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    h, w, _ = b.shape
    b = b[:h//grid*grid, :w//grid*grid, :]
    b = np.expand_dims(b, 0)
    logger.info('Size of imageA: {}'.format(b.shape))

    f = cv2.imread(args.imageB)
    h, w, _ = f.shape
    f = f[:h//grid*grid, :w//grid*grid, :]
    f = np.expand_dims(f, 0)
    logger.info('Size of imageB: {}'.format(f.shape))

    with tf.Session() as sess:
        bt = tf.constant(b, dtype=tf.float32)
        ft = tf.constant(f, dtype=tf.float32)

        yt, flow = contextual_attention(
            ft, bt, stride=stride, rate=rate,
            training=False, fuse=False)
        y = sess.run(yt)
        cv2.imwrite(args.imageOut, y[0])


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


COLORWHEEL = make_color_wheel()


def compute_color(u,v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img



def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u/(maxrad + np.finfo(float).eps)
        v = v/(maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def flow_to_image_tf(flow, name='flow_to_image'):
    """Tensorflow ops for computing flow to image.
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(flow_to_image, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
        img = img / 127.5 - 1.
        return img


def highlight_flow(flow):
    """Convert flow into middlebury color code image.
    """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h,w]
                vi = v[h,w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def highlight_flow_tf(flow, name='flow_to_image'):
    """Tensorflow ops for highlight flow.
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(highlight_flow, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
        img = img / 127.5 - 1.
        return img


def image2edge(image):
    """Convert image to edges.
    """
    out = []
    for i in range(image.shape[0]):
        img = cv2.Laplacian(image[i, :, :, :], cv2.CV_64F, ksize=3, scale=2)
        out.append(img)
    return np.float32(np.uint8(out))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageA', default='', type=str, help='Image A as background patches to reconstruct image B.')
    parser.add_argument('--imageB', default='', type=str, help='Image B is reconstructed with image A.')
    parser.add_argument('--imageOut', default='result.png', type=str, help='Image B is reconstructed with image A.')
    args = parser.parse_args()
    test_contextual_attention(args)
