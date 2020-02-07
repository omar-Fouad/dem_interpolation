import time
import os
import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel

import torch
import torch.nn.functional as F
from read_visualimg import read_visualimg

import matplotlib.pyplot as plt

from PIL import Image

parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--flist', default=r'H:\latestToPaper\clean_hill_validation_static_view.flist', type=str,
#     help='The filenames of image to be processed: input, mask, output.')
# parser.add_argument(
#     '--flist', default=r'H:\latestToPaper\outside_test_img_tile_clean.flist', type=str,
#     help='The filenames of image to be processed: input, mask, output.')
# parser.add_argument(
#     '--flist', default=r'H:\hill_data_train_and_test\glacier_hill_validation_static_view.flist', type=str,
#     help='The filenames of image to be processed: input, mask, output.')
# parser.add_argument(
#     '--flist', default=r'H:\latestToPaper\visualization.flist', type=str,
#     help='The filenames of image to be processed: input, mask, output.')
parser.add_argument(
    '--flist', default=r'H:\latestToPaper\GDEM data\tile.flist', type=str,
    help='The filenames of image to be processed: input, mask, output.')

# parser.add_argument(
#     '--flist', default=r'H:\test\flist\2_img_test_static_view.flist', type=str,
#     help='The filenames of image to be processed: input, mask, output.')
parser.add_argument(
    '--image_height', default=256, type=int,
    help='The height of images should be defined, otherwise batch mode is not'
    ' supported.')
parser.add_argument(
    '--image_width', default=256, type=int,
    help='The width of images should be defined, otherwise batch mode is not'
    ' supported.')
parser.add_argument(
    '--checkpoint_dir', default=r'H:\latestToPaper\symmetric_nogate_downx3_clean_hill_SCFEge_1202_gloss0.05_full_model_dem_hq_256\snap-310000', type=str,
    help='The directory of tensorflow checkpoint.')
parser.add_argument(
    '--outlist', default=r'C:\Users\Administrator\Desktop\SupplementedData\visualization_tif\\', type=str,
    help='The directory of putting out image.')

def cal_slope(img, w_we, w_sn):
    tensor = torch.Tensor(img)
    w_we = torch.FloatTensor(w_we).unsqueeze_(0).unsqueeze_(0)
    w_we = torch.nn.Parameter(data=w_we, requires_grad=False)
    w_sn = torch.FloatTensor(w_sn).unsqueeze_(0).unsqueeze_(0)
    w_sn = torch.nn.Parameter(data=w_sn, requires_grad=False)
    slope_we = F.conv2d(tensor, w_we, padding=1)
    slope_sn = F.conv2d(tensor, w_sn, padding=1)
    return slope_we, slope_sn

    w_we = [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]
    w_we = w_we/8
    w_sn = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    w_sn = w_sn/8
    slope_we, slope_sn = cal_slope(img, w_we, w_sn)
    slope = math.atan(torch.sqrt(slope_sn*slope_sn + slope_we*slope_we))*180/math.pi #这个是以度为单位

def idw(img_in, p=2 , near=9):
    idw = np.zeros([img_in.shape[0], img_in.shape[1]])
    for i in range(idw.shape[0]):
        for j in range(idw.shape[1]):
            weight = 0
            weighted_value = 0
            for m in range(i-near, i+near):
                for n in range(j - near, j + near):
                    if m >= img_in.shape[0] or m < 0 or n >= img_in.shape[1] or n < 0:
                        continue
                    if img_in[m,n] == 0:
                        continue
                    if m == i and n == j:
                        idw[i, j] = img_in[m, n]
                        continue
                    distance = np.sqrt(pow((i-m), 2) + pow((j-n), 2))
                    temp = 1./pow(distance, p)
                    weight += temp
                    weighted_value += img_in[m, n]*temp
            if idw[i, j] == 0:
                idw[i, j] = weighted_value/weight
    return idw
if __name__ == "__main__":
    downsample_rate = 3
    height = 256
    width = 256
    mask = np.zeros([height, width])
    for i in range(0, height, downsample_rate):
        for j in range(0, width, downsample_rate):
            mask[i, j] = 1

    # flist = open(r"H:\latestToPaper\visualization.flist")
    i = 0
    meanMSE = 0
    # for line in flist:
    #     line = line.strip("\n")
    #     img = cv2.imread(line, -1)
    #     filename = line.split('\\')[-1]
    #     img_in = img * mask
    #     # img = torch.tensor(img)
    #     # img_in = torch.tensor(img_in)
    #     img_idw = idw(img_in, 2, 6)
    #     MSE = np.sqrt(np.mean(np.square(img - img_idw)))
    #     # MSE = torch.sqrt(torch.mean(torch.square(img - img_idw)))
    #     # out = Image.fromarray(img_idw)
    #     # out.save(r'H:\latestToPaper\visualization\\idw\\idw_%d'%i +'_'+ filename)
    #     print("MSE is %f" % MSE)
    #     i = i + 1
    #     meanMSE = (meanMSE * (i - 1) + MSE) / i
    #     print("mean MSE is %f" % meanMSE + "\n")
    # i = 0
    # flist = open(r'H:\latestToPaper\clean_hill_validation_static_view.flist')
    # for line in flist:
    #     line = line.strip('\n')
    #     filename = line.split('\\')[-1]
    #     img = cv2.imread(line, -1)
    #     out = np.zeros([img.shape[0], img.shape[1]])
    #     out = cv2.normalize(img,out, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #     out = out.astype(np.uint8)
    #     cv2.imwrite(r"H:\latestToPaper\png_validation\\%d"%i + '_'+filename, out)
    #     i = i + 1

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
    # output,x_inputca1, x_outca1, x_hallu1 = model.build_server_graph(FLAGS, input_image_ph)
    output = model.build_server_graph(FLAGS, input_image_ph)
    # output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    # output = tf.saturate_cast(output, tf.uint8)
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
    # while i * downsample_rate < 256 and j * downsample_rate < 256:
    #     mask[i * downsample_rate, j * downsample_rate] = 1
    #     i = i + 1
    #     j = j + 1
    for i in range(0,255,downsample_rate):
        for j in range(0,255,downsample_rate):
            mask[i,j] = 1
    mask = 1-mask
    mask = mask[np.newaxis, ..., np.newaxis]
    with open(args.flist, 'r') as f:
        lines = f.read().splitlines()
    t = time.time()
    i = 0
    meanh=0
    filenames = read_visualimg()
    # for line in lines:
    for line in filenames:
    # for i in range(100):
        image = line.split()
        # base = os.path.basename(mask)

        image = cv2.imread(line, -1)
        filename = line.split('\\')[-1].strip("\n")
        # mask = cv2.imread(mask)
        image = cv2.resize(image, (args.image_width, args.image_height))
        # mask = cv2.resize(mask, (args.image_width, args.image_height))
        # cv2.imwrite(out, image*(1-mask/255.) + mask)
        # # continue
        # image = np.zeros((128, 256, 3))
        # mask = np.zeros((128, 256, 3))

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
        # x_inputca = sess.run(x_inputca1, feed_dict={input_image_ph: input_image})
        # x_outca = sess.run(x_outca1, feed_dict={input_image_ph: input_image})
        # x_hallu = sess.run(x_hallu1, feed_dict={input_image_ph: input_image})

        MSE = np.sqrt(np.mean(np.square((result[0] - image) / 2.0 * (max - min))))
        print(MSE)
        print(MSE*2/(max-min))
        print(max-min)
        print('\n')
        # result = (result + 1) / 2. * 255. + 0
        # image = (image + 1) / 2. * 255. + 0
        result = (result + 1) / 2. * (max-min) + min
        image = (image + 1) / 2. * (max-min) + min
        # print('Processed: {}'.format(out))
        # if i==15:
    #         #     print(meanMSE)
    #         #     print(trueMSE)

        # fig = plt.figure()
        # ax = fig.add_subplot(221)
        # ax.set_title('x_inputca')
        # max = np.max(x_inputca[0, :, :, 1])
        # min = np.min(x_inputca[0, :, :, 1])
        # tmp = (x_inputca[0, :, :, 1] - min) * 255 / (max - min)
        # # tmp = np.array(tf.Session().run(tmp)).astype(np.uint8)
        # ax.imshow(np.array(tmp).astype(np.uint8), cmap='gray')
        #
        # ax = fig.add_subplot(222)
        # ax.set_title('x_outca')
        # max = np.max(x_outca[0, :, :, 1])
        # min = np.min(x_outca[0, :, :, 1])
        # tmp = (x_outca[0, :, :, 1] - min) * 255 / (max - min)
        # # tmp = np.array(tf.Session().run(tmp)).astype(np.uint8)
        # ax.imshow(np.array(tmp).astype(np.uint8), cmap='gray')
        #
        # ax = fig.add_subplot(223)
        # ax.set_title('original')
        # # max = np.max(x_outca[0, :, :, 1])
        # # min = np.min(x_outca[0, :, :, 1])
        # # tmp = image* 255 / 2
        # # tmp = np.array(tf.Session().run(tmp)).astype(np.uint8)
        # ax.imshow(np.array(image[0][:, :, 0]).astype(np.uint8), cmap='gray')
        #
        # ax = fig.add_subplot(224)
        # ax.set_title('x_hallu')
        # max = np.max(x_hallu[0, :, :, 1])
        # min = np.min(x_hallu[0, :, :, 1])
        # tmp = (x_hallu[0, :, :, 1] - min) * 255 / (max - min)
        # # tmp = np.array(tf.Session().run(tmp)).astype(np.uint8)
        # ax.imshow(np.array(tmp).astype(np.uint8), cmap='gray')
        #
        # fig.show()

        #
        # o1 = Image.fromarray(np.array(image[0, :, :, 0]))
        # o1.save(args.outlist+'%d' % i + filename)

        o1 = Image.fromarray(np.array(result[0, :, :, 0]))
        o1.save(args.outlist+'nogate_%d' % i + filename)

        # cv2.imwrite(args.outlist + '463000_gen_%d' % i + '.tif', result[0][:, :, ::-1])
        # # cv2.imwrite(args.outlist + 'input_%d' % i + '.png', result[2][0][:, :, ::-1])
        # cv2.imwrite(args.outlist + 'png_%d' % i + '.png', image[0][:, :, ::-1])
        i = i+1
        meanh = (meanh*(i-1)+max-min)/i
        meanMSE =((meanMSE*(i-1)+MSE*2/(max-min))/i)
        trueMSE = (trueMSE * (i - 1) + MSE) / i


    print('Time total: {}'.format(time.time() - t))
    print(meanMSE)
    print(trueMSE)
    print(meanh)
