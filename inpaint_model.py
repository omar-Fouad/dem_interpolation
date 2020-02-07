""" common model for DCGAN """
import logging
import time
import cv2
import neuralgym as ng
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

import torchvision.utils as vutils

from neuralgym.models import Model
from neuralgym.ops.summary_ops import scalar_summary, images_summary
from neuralgym.ops.summary_ops import gradients_summary
from neuralgym.ops.layers import flatten, resize
from neuralgym.ops.gan_ops import gan_hinge_loss
from neuralgym.ops.gan_ops import random_interpolates

from inpaint_ops import gen_conv, gen_deconv, dis_conv, gate_conv, gate_deconv
from inpaint_ops import random_bbox, bbox2mask, local_patch, brush_stroke_mask
from inpaint_ops import resize_mask_like, contextual_attention
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image



logger = logging.getLogger()


class InpaintCAModel(Model):
    def __init__(self):
        super().__init__('InpaintCAModel')
        self.mask = []
        self.downsample_rate = []
        self.step = 0
        self.batch_size = 1
        # image_dims = [256, 256, 1]
        # masks_dims = [256, 256, 1]
        # self.dtype = tf.float32
        # self.images = tf.placeholder(
        #     self.dtype, [self.batch_size] + image_dims, name='real_images')
        # self.masks = tf.placeholder(
        #     self.dtype, [self.batch_size] + masks_dims, name='masks')

    def build_inpaint_net(self, x, reuse=False,
                          training=True, padding='SAME', name='inpaint_net'):
        """Inpaint network.

        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        # def gate_conv(x_in, cnum, ksize, stride=1, rate=1, name='conv',
        #              padding='SAME', activation='leaky_relu', use_lrn=True,training=True):

        cnum = 64
        self.input_size, self.input_size = 256, 256
        s_h, s_w = self.input_size, self.input_size
        s_h2, s_w2 = int(self.input_size / 2), int(self.input_size / 2)
        s_h4, s_w4 = int(self.input_size / 4), int(self.input_size / 4)
        s_h8, s_w8 = int(self.input_size / 8), int(self.input_size / 8)
        s_h16, s_w16 = int(self.input_size / 16), int(self.input_size / 16)
        s_h32, s_w32 = int(self.input_size / 32), int(self.input_size / 32)
        s_h64, s_w64 = int(self.input_size / 64), int(self.input_size / 64)
        with tf.variable_scope(name, reuse=reuse):
            # encoder
            x_now = x
            x1 = gate_conv(x, cnum, 7, 2, use_lrn=False, name='gconv1_ds')
            x2 = gate_conv(x1, 1 * cnum, 5, 2, name='gconv2_ds')
            x3 = gate_conv(x2, 2 * cnum, 5, 2, name='gconv3_ds')
            x4 = gate_conv(x3, 4 * cnum, 3, 2, name='gconv4_ds')
            x5 = gate_conv(x4, 4 * cnum, 3, 2, name='gconv5_ds')
            x6 = gate_conv(x5, 4 * cnum, 3, 2, name='gconv6_ds')
            x7 = gate_conv(x6, 4 * cnum, 3, 2, name='gconv7_ds')

            # dilated conv
            x7 = gate_conv(x7, 4 * cnum, 3, 1, rate=2, name='co_conv1_dlt')
            x7 = gate_conv(x7, 4 * cnum, 3, 1, rate=4, name='co_conv2_dlt')
            x7 = gate_conv(x7, 4 * cnum, 3, 1, rate=8, name='co_conv3_dlt')
            # x7 = gate_conv(x7, 4 * cnum, 3, 1, rate=16, name='co_conv4_dlt')
            x7 = gate_conv(x7, 4 * cnum, 3, 1, rate=8, name='co_conv4_dlt')
            x7 = gate_conv(x7, 4 * cnum, 3, 1, rate=4, name='co_conv5_dlt')
            x7 = gate_conv(x7, 4 * cnum, 3, 1, rate=2, name='co_conv6_dlt')

            # decoder
            x8 = gate_deconv(x7, [self.batch_size, s_h64, s_w64, 4 * cnum], name='deconv1')
            x8 = tf.concat([x6, x8], axis=3)
            x8 = gate_conv(x8, 8 * cnum, 3, 1, name='gconv8')

            x9 = gate_deconv(x8, [self.batch_size, s_h32, s_w32, 4 * cnum], name='deconv2')
            x9 = tf.concat([x5, x9], axis=3)
            x9 = gate_conv(x9, 8 * cnum, 3, 1, name='gconv9')

            x10 = gate_deconv(x9, [self.batch_size, s_h16, s_w16, 4 * cnum], name='deconv3')
            x10 = tf.concat([x4, x10], axis=3)
            x10 = gate_conv(x10, 8 * cnum, 3, 1, name='gconv10')

            x11 = gate_deconv(x10, [self.batch_size, s_h8, s_w8, 2 * cnum], name='deconv4')
            x11 = tf.concat([x3, x11], axis=3)
            x11 = gate_conv(x11, 4 * cnum, 3, 1, name='gconv11')

            x12 = gate_deconv(x11, [self.batch_size, s_h4, s_w4, 1 * cnum], name='deconv5')
            x12 = tf.concat([x2, x12], axis=3)
            x12 = gate_conv(x12, 2 * cnum, 3, 1, name='gconv12')

            x13 = gate_deconv(x12, [self.batch_size, s_h2, s_w2, cnum], name='deconv6')
            x13 = tf.concat([x1, x13], axis=3)
            x13 = gate_conv(x13, cnum, 3, 1, name='gconv13')

            x14 = gate_deconv(x13, [self.batch_size, s_h, s_w, 3], name='deconv7')
            x14 = tf.concat([x_now, x14], axis=3)
            x14 = gate_conv(x14, 1, 3, 1, activation=None, use_lrn=False, name='gconv14')

            output = tf.tanh(x14)

            return output

        # cnum = 64
        # self.input_size, self.input_size = 256, 256
        # s_h, s_w = self.input_size, self.input_size
        # s_h2, s_w2 = int(self.input_size / 2), int(self.input_size / 2)
        # s_h4, s_w4 = int(self.input_size / 4), int(self.input_size / 4)
        # s_h8, s_w8 = int(self.input_size / 8), int(self.input_size / 8)
        # s_h16, s_w16 = int(self.input_size / 16), int(self.input_size / 16)
        # s_h32, s_w32 = int(self.input_size / 32), int(self.input_size / 32)
        # s_h64, s_w64 = int(self.input_size / 64), int(self.input_size / 64)
        # with tf.variable_scope(name, reuse=reuse):
        #     # encoder
        #     x_now = x
        #     x1, mask1 = gate_conv(x, cnum, 7, 2, use_lrn=False, name='gconv1_ds')
        #     x2, mask2 = gate_conv(x1, 1 * cnum, 5, 2, name='gconv2_ds')
        #     x3, mask3 = gate_conv(x2, 2 * cnum, 5, 2, name='gconv3_ds')
        #     x4, mask4 = gate_conv(x3, 4 * cnum, 3, 2, name='gconv4_ds')
        #     x5, mask5 = gate_conv(x4, 4 * cnum, 3, 2, name='gconv5_ds')
        #     x6, mask6 = gate_conv(x5, 4 * cnum, 3, 2, name='gconv6_ds')
        #     x7, mask7 = gate_conv(x6, 4 * cnum, 3, 2, name='gconv7_ds')
        #
        #     # dilated conv
        #     x7, _ = gate_conv(x7, 4 * cnum, 3, 1, rate=2, name='co_conv1_dlt')
        #     x7, _ = gate_conv(x7, 4 * cnum, 3, 1, rate=4, name='co_conv2_dlt')
        #     x7, _ = gate_conv(x7, 4 * cnum, 3, 1, rate=8, name='co_conv3_dlt')
        #     x7, _ = gate_conv(x7, 4 * cnum, 3, 1, rate=16, name='co_conv4_dlt')
        #     # x7, _ = gate_conv(x7, 4 * cnum, 3, 1, rate=8, name='co_conv4_dlt')
        #     # x7, _ = gate_conv(x7, 4 * cnum, 3, 1, rate=4, name='co_conv5_dlt')
        #     # x7, _ = gate_conv(x7, 4 * cnum, 3, 1, rate=2, name='co_conv6_dlt')
        #
        #     # decoder
        #     x8, _ = gate_deconv(x7, [self.batch_size, s_h64, s_w64, 4 * cnum], name='deconv1')
        #     x8 = tf.concat([x6, x8], axis=3)
        #     x8, mask8 = gate_conv(x8, 8 * cnum, 3, 1, name='gconv8')
        #
        #     x9, _ = gate_deconv(x8, [self.batch_size, s_h32, s_w32, 4 * cnum], name='deconv2')
        #     x9 = tf.concat([x5, x9], axis=3)
        #     x9, mask9 = gate_conv(x9, 8 * cnum, 3, 1, name='gconv9')
        #
        #     x10, _ = gate_deconv(x9, [self.batch_size, s_h16, s_w16, 4 * cnum], name='deconv3')
        #     x10 = tf.concat([x4, x10], axis=3)
        #     x10, mask10 = gate_conv(x10, 8 * cnum, 3, 1, name='gconv10')
        #
        #     x11, _ = gate_deconv(x10, [self.batch_size, s_h8, s_w8, 2 * cnum], name='deconv4')
        #     x11 = tf.concat([x3, x11], axis=3)
        #     x11, mask11 = gate_conv(x11, 4 * cnum, 3, 1, name='gconv11')
        #
        #     x12, _ = gate_deconv(x11, [self.batch_size, s_h4, s_w4, 1 * cnum], name='deconv5')
        #     x12 = tf.concat([x2, x12], axis=3)
        #     x12, mask12 = gate_conv(x12, 2 * cnum, 3, 1, name='gconv12')
        #
        #     x13, _ = gate_deconv(x12, [self.batch_size, s_h2, s_w2, cnum], name='deconv6')
        #     x13 = tf.concat([x1, x13], axis=3)
        #     x13, mask13 = gate_conv(x13, cnum, 3, 1, name='gconv13')
        #
        #     x14, _ = gate_deconv(x13, [self.batch_size, s_h, s_w, 3], name='deconv7')
        #     x14 = tf.concat([x_now, x14], axis=3)
        #     x14, mask14 = gate_conv(x14, 1, 3, 1, activation=None, use_lrn=False, name='gconv14')
        #
        #     output = tf.tanh(x14)
        #
        #     return output, mask14

    def build_sn_patch_gan_discriminator(self, x, reuse=False, training=True):
        with tf.variable_scope('sn_patch_gan', reuse=reuse):
            cnum = 64
            x = dis_conv(x, cnum, name='conv1', training=training)
            x = dis_conv(x, cnum*2, name='conv2', training=training)
            x = dis_conv(x, cnum*4, name='conv3', training=training)
            x = dis_conv(x, cnum*4, name='conv4', training=training)
            x = dis_conv(x, cnum*4, name='conv5', training=training)
            x = dis_conv(x, cnum*4, name='conv6', training=training)
            x = flatten(x, name='flatten')
            return x

    def build_gan_discriminator(
            self, batch, reuse=False, training=True):
        with tf.variable_scope('discriminator', reuse=reuse):
            d = self.build_sn_patch_gan_discriminator(
                batch, reuse=reuse, training=training)
            return d


    def build_graph_with_losses_down(
            self, FLAGS, batch_data, mask, downsample_rate, training=True, summary=False,
            reuse=False):
        self.mask = mask
        self.downsample_rate = downsample_rate
        if FLAGS.guided:
            batch_data, edge = batch_data
            edge = edge[:, :, :, 0:1] / 255.
            edge = tf.cast(edge > FLAGS.edge_threshold, tf.float32)
        # modified
        # batch_pos = batch_data / 127.5 - 1.
        # mean = tf.reduce_mean(batch_data * (1. - self.mask), 1:3)

        # mean = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(batch_data*(1.-self.mask), 1), 1), 1)*downsample_rate*downsample_rate
        # mean = mean[..., np.newaxis, np.newaxis, np.newaxis]
        # batch_pos = batch_data / mean - 1.

        # generate mask, 1 represents masked point
        # bbox = random_bbox(FLAGS)
        # regular_mask = bbox2mask(FLAGS, bbox, name='mask_c')
        # irregular_mask = brush_stroke_mask(FLAGS, name='mask_c')
        # mask = tf.cast(
        #     tf.logical_or(
        #         tf.cast(irregular_mask, tf.bool),
        #         tf.cast(regular_mask, tf.bool),
        #     ),
        #     tf.float32
        # )
        max = tf.reduce_max(batch_data, 1)
        max = tf.reduce_max(max, 1)
        max = max[...,np.newaxis,np.newaxis]
        min = tf.reduce_min(batch_data, 1)
        min = tf.reduce_min(min, 1)
        min = min[..., np.newaxis, np.newaxis]
        batch_pos = (batch_data-min)*2/(max-min)-1.
        batch_incomplete = batch_pos*(1.-self.mask)
        if FLAGS.guided:
            edge = edge * self.mask
            xin = tf.concat([batch_incomplete, edge], axis=3)
        else:
            xin = batch_incomplete
        # x1, x2, offset_flow = self.build_inpaint_net(
        #     xin, self.mask, reuse=reuse, training=training,
        #     padding=FLAGS.padding)
        mask = np.repeat(mask, 16, axis=0)
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        xin = tf.concat([batch_incomplete, mask], axis=3)
        x2, gen_mask = self.build_inpaint_net(
            xin, reuse=reuse, training=training,
            padding=FLAGS.padding)
        batch_predicted = x2
        losses = {}
        # apply mask and complete image
        batch_complete = batch_predicted*self.mask + batch_incomplete*(1.-self.mask)
        # local patches
        # x1 = batch_pos
        # losses['ae_loss'] = FLAGS.l1_loss_alpha * tf.reduce_mean(tf.abs(batch_pos - x1))
        losses['ae_loss'] = FLAGS.l1_loss_alpha * tf.reduce_mean(tf.abs(batch_pos - x2))
        if summary:
            scalar_summary('losses/ae_loss', losses['ae_loss'])
            if FLAGS.guided:
                viz_img = [
                    batch_pos,
                    batch_incomplete + edge,
                    batch_complete]
            else:
                viz_img = [batch_pos, batch_incomplete, batch_complete]
            # if offset_flow is not None:
            #     viz_img.append(
            #         resize(offset_flow, scale=4,
            #                func=tf.image.resize_bilinear))

            # if self.step % FLAGS.train_spe == 0:
            #     epoch = FLAGS.max_iters / FLAGS.train_spe
            #     # fake = tf.constant(batch_predicted)
            #     # sess = tf.Session()
            #     # sess.run(tf.global_variables_initializer())
            #     # fake = batch_predicted.eval(session=sess)
            #     # fake = tf.Session().run(batch_predicted)
            #     # a = Image.fromarray(fake[0,:,:,0])
            #     # vutils.save_image(batch_predicted,'./images/epoch_%03d_fake.png' % epoch,normalize=True)
            #     # vutils.save_image(batch_data, './images/epoch_%03d_fake.png' % epoch,
            #     #                   normalize=True)
            #     step = '%d' % self.step
            #     name = "fake_" + step
            #     # img = tf.clip_by_value(batch_predicted * 255., 0, 255)
            #     summary_image1 = tf.summary.image(name, batch_predicted, max_outputs=2)
            #     name = "real_" + step
            #     # img = tf.clip_by_value(batch_data * 255., 0, 255)
            #     summary_image2 = tf.summary.image(name, batch_data, max_outputs=2)
            #     summary_image = tf.summary.merge([summary_image1, summary_image2])
            #     with tf.Session() as sess:
            #         summary_write = tf.summary.FileWriter('./images', sess.graph)  # out_dir 为输出路径
            #         summary_out = sess.run(summary_image)
            #         summary_write.add_summary(summary_out)
            images_summary(
                tf.concat(viz_img, axis=2),
                'raw_incomplete_predicted_complete', FLAGS.viz_max_out)

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)
        if FLAGS.gan_with_mask:
            batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(self.mask, [FLAGS.batch_size*2, 1, 1, 1])], axis=3)
        if FLAGS.guided:
            # conditional GANs
            batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(edge, [2, 1, 1, 1])], axis=3)
        # wgan with gradient penalty
        if FLAGS.gan == 'sngan':
            pos_neg = self.build_gan_discriminator(batch_pos_neg, training=training, reuse=reuse)
            pos, neg = tf.split(pos_neg, 2)
            g_loss, d_loss = gan_hinge_loss(pos, neg)
            losses['g_loss'] = g_loss
            losses['d_loss'] = d_loss
        else:
            raise NotImplementedError('{} not implemented.'.format(FLAGS.gan))
        if summary:
            # summary the magnitude of gradients from different losses w.r.t. predicted image
            gradients_summary(losses['g_loss'], batch_predicted, name='g_loss')
            gradients_summary(losses['g_loss'], x2, name='g_loss_to_x2')
            # gradients_summary(losses['ae_loss'], x1, name='ae_loss_to_x1')
            gradients_summary(losses['ae_loss'], x2, name='ae_loss_to_x2')
        losses['g_loss'] = FLAGS.gan_loss_alpha * losses['g_loss']
        if FLAGS.ae_loss:
            losses['g_loss'] += losses['ae_loss']
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

    def build_graph_with_losses(
            self, FLAGS, batch_data, training=True, summary=False,
            reuse=False):
        if FLAGS.guided:
            batch_data, edge = batch_data
            edge = edge[:, :, :, 0:1] / 255.
            edge = tf.cast(edge > FLAGS.edge_threshold, tf.float32)
        batch_pos = batch_data / 127.5 - 1.
        # generate mask, 1 represents masked point
        bbox = random_bbox(FLAGS)
        regular_mask = bbox2mask(FLAGS, bbox, name='mask_c')
        irregular_mask = brush_stroke_mask(FLAGS, name='mask_c')
        mask = tf.cast(
            tf.logical_or(
                tf.cast(irregular_mask, tf.bool),
                tf.cast(regular_mask, tf.bool),
            ),
            tf.float32
        )

        batch_incomplete = batch_pos*(1.-mask)
        if FLAGS.guided:
            edge = edge * mask
            xin = tf.concat([batch_incomplete, edge], axis=3)
        else:
            xin = batch_incomplete
        x1, x2, offset_flow = self.build_inpaint_net(
            xin, mask, reuse=reuse, training=training,
            padding=FLAGS.padding)
        batch_predicted = x2
        losses = {}
        # apply mask and complete image
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        # local patches
        losses['ae_loss'] = FLAGS.l1_loss_alpha * tf.reduce_mean(tf.abs(batch_pos - x1))
        losses['ae_loss'] += FLAGS.l1_loss_alpha * tf.reduce_mean(tf.abs(batch_pos - x2))
        if summary:
            scalar_summary('losses/ae_loss', losses['ae_loss'])
            if FLAGS.guided:
                viz_img = [
                    batch_pos,
                    batch_incomplete + edge,
                    batch_complete]
            else:
                viz_img = [batch_pos, batch_incomplete, batch_complete]
            if offset_flow is not None:
                viz_img.append(
                    resize(offset_flow, scale=4,
                           func=tf.image.resize_bilinear))
            images_summary(
                tf.concat(viz_img, axis=2),
                'raw_incomplete_predicted_complete', FLAGS.viz_max_out)

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)
        if FLAGS.gan_with_mask:
            batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(mask, [FLAGS.batch_size*2, 1, 1, 1])], axis=3)
        if FLAGS.guided:
            # conditional GANs
            batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(edge, [2, 1, 1, 1])], axis=3)
        # wgan with gradient penalty
        if FLAGS.gan == 'sngan':
            pos_neg = self.build_gan_discriminator(batch_pos_neg, training=training, reuse=reuse)
            pos, neg = tf.split(pos_neg, 2)
            g_loss, d_loss = gan_hinge_loss(pos, neg)
            losses['g_loss'] = g_loss
            losses['d_loss'] = d_loss
        else:
            raise NotImplementedError('{} not implemented.'.format(FLAGS.gan))
        if summary:
            # summary the magnitude of gradients from different losses w.r.t. predicted image
            gradients_summary(losses['g_loss'], batch_predicted, name='g_loss')
            gradients_summary(losses['g_loss'], x2, name='g_loss_to_x2')
            # gradients_summary(losses['ae_loss'], x1, name='ae_loss_to_x1')
            gradients_summary(losses['ae_loss'], x2, name='ae_loss_to_x2')
        losses['g_loss'] = FLAGS.gan_loss_alpha * losses['g_loss']
        if FLAGS.ae_loss:
            losses['g_loss'] += losses['ae_loss']
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses
    # modified
    def downsampleMask_graph(self, FLAGS, batch_data, mask, downsample_rate, name='val'):
        if FLAGS.guided:
            batch_data, edge = batch_data
            edge = edge[:, :, :, 0:1] / 255.
            edge = tf.cast(edge > FLAGS.edge_threshold, tf.float32)

        # mask = brush_stroke_mask(name='mask_c')
        # regular_mask = bbox2mask(bbox, name='mask_c')
        # irregular_mask = brush_stroke_mask(name='mask_c')
        # mask = tf.cast(
        #     tf.logical_or(
        #         tf.cast(irregular_mask, tf.bool),
        #         tf.cast(regular_mask, tf.bool),
        #     ),
        #     tf.float32
        # )

        # 事实上还是对输入的图像做了归一化 这里要改
        # modified
        # batch_pos = batch_data / 127.5 - 1.
        mean = tf.reduce_mean(tf.reduce_mean(batch_data*(1.-mask), 1), 2)*downsample_rate*downsample_rate
        batch_pos = batch_data/mean -1
        batch_incomplete = batch_pos*(1.-mask)

        if FLAGS.guided:
            edge = edge * mask
            xin = tf.concat([batch_incomplete, edge], axis=3)
        else:
            xin = batch_incomplete
        # inpaint
        x1, x2, offset_flow = self.build_inpaint_net(
            xin, mask, reuse=True,
            training=False, padding=FLAGS.padding)
        batch_predicted = x2
        # apply mask and reconstruct
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        # global image visualization
        if FLAGS.guided:
            viz_img = [
                batch_pos,
                batch_incomplete + edge,
                batch_complete]
        else:
            viz_img = [batch_pos, batch_incomplete, batch_complete]
        if offset_flow is not None:
            viz_img.append(
                resize(offset_flow, scale=4,
                       func=tf.image.resize_bilinear))
        images_summary(
            tf.concat(viz_img, axis=2),
            name+'_raw_incomplete_complete', FLAGS.viz_max_out)
        return batch_complete

    def build_infer_graph(self, FLAGS, batch_data, bbox=None, name='val'):
        """
        """
        if FLAGS.guided:
            batch_data, edge = batch_data
            edge = edge[:, :, :, 0:1] / 255.
            edge = tf.cast(edge > FLAGS.edge_threshold, tf.float32)
        mask = brush_stroke_mask(name='mask_c')
        regular_mask = bbox2mask(bbox, name='mask_c')
        irregular_mask = brush_stroke_mask(name='mask_c')
        mask = tf.cast(
            tf.logical_or(
                tf.cast(irregular_mask, tf.bool),
                tf.cast(regular_mask, tf.bool),
            ),
            tf.float32
        )

        batch_pos = batch_data / 127.5 - 1.
        batch_incomplete = batch_pos*(1.-mask)
        if FLAGS.guided:
            edge = edge * mask
            xin = tf.concat([batch_incomplete, edge], axis=3)
        else:
            xin = batch_incomplete
        # inpaint
        x1, x2, offset_flow = self.build_inpaint_net(
            xin, mask, reuse=True,
            training=False, padding=FLAGS.padding)
        batch_predicted = x2
        # apply mask and reconstruct
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        # global image visualization
        if FLAGS.guided:
            viz_img = [
                batch_pos,
                batch_incomplete + edge,
                batch_complete]
        else:
            viz_img = [batch_pos, batch_incomplete, batch_complete]
        if offset_flow is not None:
            viz_img.append(
                resize(offset_flow, scale=4,
                       func=tf.image.resize_bilinear))
        images_summary(
            tf.concat(viz_img, axis=2),
            name+'_raw_incomplete_complete', FLAGS.viz_max_out)
        return batch_complete

    def build_static_infer_graph(self, FLAGS, batch_data, name):
        """
        """
        # generate mask, 1 represents masked point
        bbox = (tf.constant(FLAGS.height//2), tf.constant(FLAGS.width//2),
                tf.constant(FLAGS.height), tf.constant(FLAGS.width))
        return self.build_infer_graph(batch_data, bbox, name)


    def build_server_graph(self, FLAGS, batch_data, reuse=False, is_training=False):
        """
        """
        # generate mask, 1 represents masked point
        if FLAGS.guided:
            batch_raw, edge, masks_raw = tf.split(batch_data, 3, axis=2)
            edge = edge[:, :, :, 0:1] / 255.
            edge = tf.cast(edge > FLAGS.edge_threshold, tf.float32)
        else:
            batch_raw, masks_raw = tf.split(batch_data, 2, axis=2)
        # masks = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)
        masks = masks_raw

        batch_pos = batch_raw #/ 127.5 - 1.
        batch_incomplete = batch_pos * (1. - masks)
        if FLAGS.guided:
            edge = edge * masks[:, :, :, 0:1]
            xin = tf.concat([batch_incomplete, edge], axis=3)
        else:
            xin = batch_incomplete
        # inpaint
        # x1, x2, flow = self.build_inpaint_net(
        #     xin, masks, reuse=reuse, training=is_training)
        # batch_predict = x2
        # # apply mask and reconstruct
        # batch_complete = batch_predict*masks + batch_incomplete*(1-masks)
        # return batch_predict
        mask = tf.convert_to_tensor(masks_raw, dtype=tf.float32)
        xin = tf.concat([xin, mask], axis=3)
        # x2, gen_mask= self.build_inpaint_net(
        #     xin, reuse=reuse, training=is_training)
        x2 = self.build_inpaint_net(
            xin, reuse=reuse, training=is_training)
        batch_predict = x2
        batch_predict = x2*(1-masks) + batch_raw*masks
        # apply mask and reconstruct
        # batch_complete = batch_predict*masks + batch_incomplete*(1-masks)
        # return batch_complete
        return batch_predict

    def downsample_build_server_graph(self, FLAGS, batch_image, batch_mask, reuse=False, is_training=False):
        """
        """
        # generate mask, 1 represents masked point
        # if FLAGS.guided:
        #     batch_raw, edge, masks_raw = tf.split(batch_data, 3, axis=2)
        #     edge = edge[:, :, :, 0:1] / 255.
        #     edge = tf.cast(edge > FLAGS.edge_threshold, tf.float32)
        # else:
        batch_raw = batch_image
        masks = batch_mask
        # 下面这个感觉是一个二值化过程
        # masks = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)

        # batch_pos = batch_raw / 127.5 - 1.
        batch_incomplete = batch_raw * (1. - masks)
        # if FLAGS.guided:
        #     edge = edge * masks[:, :, :, 0:1]
        #     xin = tf.concat([batch_incomplete, edge], axis=3)
        # else:
        xin = batch_incomplete
        # inpaint
        # x_stage1, x_stage2, offset_flow, x_inputca, x_outca
        x1, x2, flow,x_inputca, x_outca = self.build_inpaint_net(
            xin, masks, reuse=reuse, training=is_training)
        batch_predict = x2
        # apply mask and reconstruct
        # batch_complete = batch_predict*masks + batch_incomplete*(1-masks)
        # return batch_complete
        return batch_predict,x_inputca, x_outca
