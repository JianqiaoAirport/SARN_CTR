# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from rnn import dynamic_rnn
from utils import *
from Dice import dice
import math


class Model():

    """
    SARN
    """

    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, matrix_width=36, use_negsampling=True):

        self.EMBEDDING_DIM = EMBEDDING_DIM
        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')  # item id list
            self.cat_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cat_his_batch_ph')  # cate id list

            tmp_n = 4.0 / 4.0
            discard_index = int(n_cat * tmp_n)
            tensor_2d = tf.ones_like(self.cat_his_batch_ph) * (int(n_cat * tmp_n) + 1)
            self.cat_his_batch_ph = tf.where(self.cat_his_batch_ph > discard_index, tensor_2d, self.cat_his_batch_ph)

            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')  # user id
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')  # target item id
            self.cat_batch_ph = tf.placeholder(tf.int32, [None, ], name='cat_batch_ph')  # target item  cateid

            tensor_1d = tf.ones_like(self.cat_batch_ph) * (int(n_cat * tmp_n) + 1)
            self.cat_batch_ph = tf.where(self.cat_batch_ph > discard_index, tensor_1d, self.cat_batch_ph)

            self.mask = tf.placeholder(tf.float32, [None, None], name='mask')
            self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])  #
            self.use_negsampling = use_negsampling
            if use_negsampling:
                # generate 3 item IDs from negative sampling.
                self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_mid_batch_ph')
                self.noclk_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_cat_batch_ph')

                tensor_3d = tf.ones_like(self.noclk_cat_batch_ph) * (int(n_cat * tmp_n) + 1)
                self.noclk_cat_batch_ph = tf.where(self.noclk_cat_batch_ph > discard_index, tensor_3d, self.noclk_cat_batch_ph)

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var",
                                                      [n_uid, EMBEDDING_DIM])  # shape : (543080,18)

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var",
                                                      [n_mid, EMBEDDING_DIM])  # shape : (367983,18)

            self.cat_embeddings_var = tf.get_variable("cat_embedding_var",
                                                      [n_cat + 1, EMBEDDING_DIM])  # shape : (1601,18)

            tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var,
                                                             self.uid_batch_ph)  # shape : (?,18)
            tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                             self.mid_batch_ph)  # shape : (?,18)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                                 self.mid_his_batch_ph)  # shape : (?,?,18)
            if self.use_negsampling:  # noclk_mid_his_batch_embedded shape : (?,?,?,18)
                self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                                           self.noclk_mid_batch_ph)  # noclk_mid_batch_ph shape :(?,?,?)

            self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var,
                                                             self.cat_batch_ph)  # shape : (?,18)
            self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var,
                                                                 self.cat_his_batch_ph)  # shape : (?,?,18) number of sample, number of history, length of embedding
            if self.use_negsampling:
                self.noclk_cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var,
                                                                           self.noclk_cat_batch_ph)
            tf.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)

            self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)  # shape : (?,36)
            self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
            # (b, T, 1)
            self.his_mask = tf.expand_dims(self.mask, 2)

            self.item_his_eb = self.item_his_eb * self.his_mask
            # shape : (?,?,36) number of sample, number of history, length of embedding
            self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb,
                                                 1)  # shape : (?,36)  number of sample, length(embedding)
            if self.use_negsampling:
                with tf.name_scope('negsampling'):
                    # 0 means only using the first negative item ID. 3 item IDs are inputed in the line 24.
                    self.noclk_item_his_eb = tf.concat(  # noclk_item_his_eb shape : (?,?,36)
                        [self.noclk_mid_his_batch_embedded[:, :, 0, :], self.noclk_cat_his_batch_embedded[:, :, 0, :]],
                        -1)
                    # cat embedding 18 concate item embedding 18.
                    self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb,  # noclk_item_his_eb shape : (?,?,36)
                                                        [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1],
                                                         EMBEDDING_DIM * 2])
                    self.noclk_item_his_eb = self.noclk_item_his_eb * self.his_mask

        with tf.name_scope('base_rn'):
            self.rn_outputs_1, self.rn_outputs_2 = self.build_base_relational_net(self.item_eb, self.item_his_eb, matrix_width, i_net=0)

        with tf.name_scope('aux_structure'):

            self.aux_loss = self.relational_reg(self.rn_outputs_1, self.mask)

            aux_loss_neg = self.auxiliary_loss_neg(self.rn_outputs_1,
                                                   self.item_his_eb,
                                                   self.noclk_item_his_eb,
                                                   self.mask, stag="aux_loss")
            self.aux_loss += aux_loss_neg

        with tf.name_scope('relational_inductive_interest_net'):

            seq_len_ph = tf.expand_dims(tf.cast(self.seq_len_ph, tf.float32), 1)
            seq_len_ph = tf.tile(seq_len_ph, [1, EMBEDDING_DIM * 2])
            item_his_eb_mean = self.item_his_eb_sum / seq_len_ph
            item_his_eb_max = tf.reduce_max(self.item_his_eb, axis=[1])

            self.interest = self.rn_outputs_2

            inp = tf.concat(
                [self.item_eb, self.interest, self.item_eb * item_his_eb_mean, item_his_eb_max, item_his_eb_mean], -1)
        print(tf.trainable_variables())
        self.build_fcn_net(inp, use_dice=True)

    def build_base_relational_net(self, item_eb, item_his_eb, matrix_width, i_net):
        with tf.variable_scope('subnet_' + str(i_net)):
            Q = tf.layers.dense(item_his_eb, matrix_width, activation=tf.nn.relu, name='Q_dense',
                                use_bias=False, reuse=tf.AUTO_REUSE)
            Q = Q / (tf.sqrt(tf.reduce_sum(Q * Q, axis=-1)[:, :, tf.newaxis]) + 1e-10)

            K = tf.layers.dense(item_his_eb, matrix_width, activation=tf.nn.relu, name='K_dense',
                                use_bias=False, reuse=tf.AUTO_REUSE)
            K = K / (tf.sqrt(tf.reduce_sum(K * K, axis=-1)[:, :, tf.newaxis]) + 1e-10)

            V = item_his_eb
            K_tmp = tf.transpose(K, perm=[0, 2, 1])

            relation = tf.matmul(Q, K_tmp)
            self.relation_save = relation

            relation_cube = relation[:, :, :, tf.newaxis]

            his_len = tf.shape(K)[1]
            V_cube_1 = V[:, tf.newaxis, :, :]
            V_cube_1 = tf.tile(V_cube_1, [1, his_len, 1, 1])
            V_cube_2 = V[:, :, tf.newaxis, :]
            V_cube_2 = tf.tile(V_cube_2, [1, 1, his_len, 1])

            mixed_cube = V_cube_1 + relation_cube * V_cube_2

            target = item_eb[:, tf.newaxis, tf.newaxis, :]
            target = tf.tile(target, [1, his_len, his_len, 1])

            att_mlp_input = tf.concat([target, mixed_cube], axis=-1)

            att_mlp_hid = tf.layers.dense(att_mlp_input, 36, activation=tf.nn.relu)
            att_mlp_output = tf.layers.dense(att_mlp_hid, 1, activation=tf.nn.sigmoid)

            att_out = att_mlp_output * mixed_cube

            mask = self.his_mask[:, :, tf.newaxis, :]
            mask = tf.tile(mask, [1, 1, his_len, 1])

            att_out = att_out * mask

            relational_output_max = tf.reduce_max(att_out, axis=[1, 2])

            return mixed_cube, relational_output_max

    def build_fcn_net(self, inp, use_dice=False):
        with tf.name_scope('fcn_net'):
            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
            dnn1 = tf.layers.dense(bn1, 50, activation=None, use_bias=True, name='f1')
            if use_dice:
                dnn1 = dice(dnn1, name='dice_1')
            else:
                dnn1 = prelu(dnn1, 'prelu1')
            dnn2 = tf.layers.dense(dnn1, 10, activation=None, use_bias=False, name='f2')
            if use_dice:
                dnn2 = dice(dnn2, name='dice_2')
            else:
                dnn2 = prelu(dnn2, 'prelu2')
            dnn3 = tf.layers.dense(dnn2, 2, activation=None, use_bias=False, name='f3')
            self.y_hat = tf.nn.softmax(dnn3) + 1e-8

            with tf.name_scope('Metrics'):
                ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
                self.loss = ctr_loss
                tf.summary.scalar("loss without aux_loss", self.loss)
                if self.use_negsampling:
                    self.loss += self.aux_loss
                    pass
                tf.summary.scalar('loss', self.loss)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

                # Accuracy metric
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)

            self.merged = tf.summary.merge_all()

    def relational_reg(self, relations, mask, stag='auxiliary_loss'):

        batch_size = tf.shape(relations)[0]
        his_len = tf.shape(relations)[1]

        mask = mask[:, :, tf.newaxis, tf.newaxis]
        mask = tf.tile(mask, [1, 1, his_len, 1])
        mask = tf.cast(mask, tf.float32)

        coord_tmp_1 = tf.range(his_len)[tf.newaxis, :, tf.newaxis, tf.newaxis]
        coord_tmp_1 = tf.tile(coord_tmp_1, [batch_size, 1, his_len, 1])

        coord_tmp_2 = tf.range(his_len)[tf.newaxis, tf.newaxis, :, tf.newaxis]
        coord_tmp_2 = tf.tile(coord_tmp_2, [batch_size, his_len, 1, 1])

        label_aux = coord_tmp_2-coord_tmp_1

        label_aux = tf.cast(label_aux, dtype=tf.float32) / tf.cast(tf.reshape(his_len, shape=[1, 1, 1, 1]), dtype=tf.float32)

        indicator = tf.where(tf.greater(label_aux, 0), tf.ones_like(label_aux) * 1.0, tf.ones_like(label_aux) * -1.0)

        label_aux = tf.sqrt(label_aux * indicator)

        label_aux = label_aux * indicator

        distance = self.auxiliary_net_reg(relations)
        distance_loss = tf.square(distance-label_aux) * mask
        loss = tf.reduce_mean(distance_loss)
        return loss

    def auxiliary_net_reg(self, in_, stag='auxiliary_net_4'):
        with tf.name_scope('auxiliary_net_4'):
            dnn1 = tf.layers.dense(in_, 10, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
            dnn1 = tf.nn.relu(dnn1)
            y_hat = tf.layers.dense(dnn1, 1, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
            return y_hat


    def auxiliary_loss_neg(self, rn_outputs, click_seq, noclick_seq, mask, stag=None):

        mask = tf.cast(mask, tf.float32)

        h_states = tf.reduce_mean(rn_outputs, axis=1) * ((tf.cast(tf.shape(rn_outputs)[1], tf.float32) / tf.cast(self.seq_len_ph, tf.float32))[:, tf.newaxis, tf.newaxis])

        his_len = tf.shape(h_states)[1]
        index_shuffled = tf.random_shuffle(tf.range(his_len))[:, tf.newaxis]
        item_his_eb_transposed = tf.transpose(h_states, perm=[1, 0, 2])
        shuffled_items = tf.gather_nd(item_his_eb_transposed, index_shuffled)
        h_states = tf.transpose(shuffled_items, perm=[1, 0, 2])

        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net_neg(click_input_)[:, :, 0]
        noclick_prop_ = self.auxiliary_net_neg(noclick_input_)[:, :, 0]
        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net_neg(self, in_, stag='ori_auxiliary_net'):
        with tf.name_scope('ori_auxiliary_net'):
            bn1 = tf.layers.batch_normalization(inputs=in_, name='ori_bn1' + stag, reuse=tf.AUTO_REUSE)
            dnn1 = tf.layers.dense(bn1, 72, activation=None, name='ori_f1' + stag, reuse=tf.AUTO_REUSE)
            dnn1 = tf.nn.sigmoid(dnn1)
            dnn2 = tf.layers.dense(dnn1, 36, activation=None, name='ori_f2' + stag, reuse=tf.AUTO_REUSE)
            dnn2 = tf.nn.sigmoid(dnn2)
            dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='ori_f3' + stag, reuse=tf.AUTO_REUSE)
            y_hat = tf.nn.softmax(dnn3) + 0.00000001
            return y_hat

    def train(self, sess, inps):
        if self.use_negsampling:
            loss, accuracy, aux_loss, _ = sess.run([self.loss, self.accuracy, self.aux_loss, self.optimizer],
                                                   feed_dict={
                                                       self.uid_batch_ph: inps[0],
                                                       self.mid_batch_ph: inps[1],
                                                       self.cat_batch_ph: inps[2],
                                                       self.mid_his_batch_ph: inps[3],
                                                       self.cat_his_batch_ph: inps[4],
                                                       self.mask: inps[5],
                                                       self.target_ph: inps[6],
                                                       self.seq_len_ph: inps[7],
                                                       self.lr: inps[8],
                                                       self.noclk_mid_batch_ph: inps[9],
                                                       self.noclk_cat_batch_ph: inps[10],
                                                       # self.user_attr: inps[11],
                                                   })
            # return loss, accuracy
            return loss, accuracy, aux_loss
        else:
            loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7],
                self.lr: inps[8],
            })
            return loss, accuracy, 0

    def calculate(self, sess, inps):
        if self.use_negsampling:
            probs, loss, accuracy, aux_loss = sess.run([
                self.y_hat, self.loss, self.accuracy, self.aux_loss], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7],
                self.noclk_mid_batch_ph: inps[8],
                self.noclk_cat_batch_ph: inps[9],
            })
            return probs, loss, accuracy, aux_loss
        else:
            probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7]
            })
            return probs, loss, accuracy, 0

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)

