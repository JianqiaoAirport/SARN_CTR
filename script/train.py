# -*- coding: utf-8 -*-
import numpy
from data_iterator import DataIterator
from model import *
import time
import random
import sys
import os
from utils import *
import logging

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0


def prepare_data_DRIIN(input, target, maxlen=None, return_neg=False):
    lengths_x = [len(s[4]) for s in input]
    seqs_mid = [inp[3] for inp in input]
    seqs_cat = [inp[4] for inp in input]
    noclk_seqs_mid = [inp[5] for inp in input]
    noclk_seqs_cat = [inp[6] for inp in input]
    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_noclk_seqs_mid.append(inp[5])
                new_noclk_seqs_cat.append(inp[6])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat
        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_mid)
    maxlen_x = numpy.max(lengths_x)
    neg_samples = len(noclk_seqs_mid[0][0])

    mid_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    cat_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    noclk_mid_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    noclk_cat_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    mid_mask = numpy.zeros((n_samples, maxlen_x)).astype('float32')

    for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy

    uids = numpy.array([inp[0] for inp in input])
    mids = numpy.array([inp[1] for inp in input])
    cats = numpy.array([inp[2] for inp in input])

    if return_neg:

        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(
            lengths_x), noclk_mid_his, noclk_cat_his

    else:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x)


def eval_DRIIN(sess, test_data, model, model_path, maxlen=None):
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    for src, tgt in test_data:
        nums += 1
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = \
            prepare_data_DRIIN(src, tgt, maxlen, return_neg=True)
        prob, loss, acc, aux_loss = model.calculate(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl,
                                                           noclk_mids, noclk_cats])

        loss_sum += loss
        aux_loss_sum += aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum = aux_loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        model.save(sess, model_path)
    return test_auc, loss_sum, accuracy_sum, aux_loss_sum, loss_sum - aux_loss_sum

def train(
    datasetdir="data/Electronics/",
    train_file="/training_set",
    test_file="/test_set",
    uid_voc="/uid_voc.pkl",
    mid_voc="/mid_voc.pkl",
    cat_voc="/cat_voc.pkl",
    batch_size=128,
    maxlen=30,
    matrix_width=36,
    test_iter=100,
    save_iter=4000000,
    model_type='DRIIN',
    seed=2,
):
    train_file = datasetdir + train_file
    test_file = datasetdir + test_file
    uid_voc = datasetdir + uid_voc
    mid_voc = datasetdir + mid_voc
    cat_voc = datasetdir + cat_voc

    model_path = datasetdir + "/dnn_save_path/ckpt_noshuff" + model_type + str(seed)
    best_model_path = datasetdir + "/dnn_best_model/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, shuffle_each_epoch=True,
                                  datasetdir=datasetdir)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, datasetdir=datasetdir)
        n_uid, n_mid, n_cat = train_data.get_n()

        model = Model(n_uid, n_mid, n_cat, EMBEDDING_DIM, matrix_width=matrix_width)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        start_time = time.time()

        file1 = logging.FileHandler(
            filename=datasetdir + '/my_logs/' + "model_" + str(
                time.asctime(time.localtime(start_time))) + '.txt',
            mode='a', encoding='utf-8')
        logger_accuracy = logging.Logger(name='name_accuracy', level=logging.INFO)
        logger_accuracy.addHandler(file1)

        logger_accuracy.info("start_time:" + time.asctime(time.localtime(start_time)) + "\r\n")
        logger_accuracy.info(
            model_type + " " + datasetdir + " maxlen:" + str(maxlen) + " batch_size:" + str(batch_size) + "\r\n")

        file2 = logging.FileHandler(
            filename=datasetdir + '/loss_logs/' + "model_test_" + str(
                time.asctime(time.localtime(start_time))) + '.txt',
            mode='a', encoding='utf-8')
        logger_test_loss = logging.Logger(name='name_loss', level=logging.INFO)
        logger_test_loss.addHandler(file2)

        file3 = logging.FileHandler(
            filename=datasetdir + '/loss_logs/' + "model_train_" + str(
                time.asctime(time.localtime(start_time))) + '.txt',
            mode='a', encoding='utf-8')
        logger_train_loss = logging.Logger(name='name_loss', level=logging.INFO)
        logger_train_loss.addHandler(file3)

        iter = 0
        lr = 0.01
        global best_auc
        breakflag = False

        test_auc_log, loss_sum_log, accuracy_sum_log, aux_loss_sum_log, loss_without_aux = eval_DRIIN(sess,
                                                                                                      test_data,
                                                                                                      model,
                                                                                                      best_model_path,
                                                                                                      maxlen)
        logger_accuracy.info(
            'test_auc: %.4f - test_loss: %.4f - test_accuracy: %.4f - test_aux_loss: %.4f - loss_without_aux: %.4f *best_auc: %.4f \r\n' % (
                test_auc_log, loss_sum_log, accuracy_sum_log, aux_loss_sum_log, loss_without_aux, best_auc))
        # writer.add_summary(summary, iter)
        print(
            'test_auc: %.4f - test_loss: %.4f - test_accuracy: %.4f - test_aux_loss: %.4f - loss_without_aux: %.4f  *best_auc: %.4f' %
            (test_auc_log, loss_sum_log, accuracy_sum_log, aux_loss_sum_log, loss_without_aux, best_auc))
        logger_test_loss.info(
            '%d,%.4f,%.4f,%.4f' % \
            (iter, loss_sum_log, aux_loss_sum_log, loss_without_aux))

        logger_train_loss.info(
            '%d,%.4f,%.4f,%.4f' % \
            (iter, loss_sum_log, aux_loss_sum_log, loss_without_aux))

        for epoch in range(5):
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            if breakflag:
                break
            print("epoch:", epoch)
            logger_accuracy.info('epoch: %d\r\n' % epoch)
            for src, tgt in train_data:
                iter += 1
                uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data_DRIIN(
                    src, tgt, maxlen, return_neg=True)
                loss, acc, aux_loss = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, lr,
                                                         noclk_mids, noclk_cats])
                loss_sum += loss
                accuracy_sum += acc
                aux_loss_sum += aux_loss

                sys.stdout.flush()
                if (iter % test_iter) == 0:
                    logger_accuracy.info(
                        'iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- tran_aux_loss: %.4f \r\n' % \
                        (iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter))
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- tran_aux_loss: %.4f' % \
                          (iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter))

                    logger_train_loss.info(
                        '%d,%.4f,%.4f,%.4f' % \
                        (iter, loss_sum / test_iter, aux_loss_sum / test_iter, (loss_sum - aux_loss_sum) / test_iter, ))

                    test_auc_log, loss_sum_log, accuracy_sum_log, aux_loss_sum_log, loss_without_aux = eval_DRIIN(sess,
                                                                                                test_data,
                                                                                                model,
                                                                                                best_model_path,
                                                                                                maxlen)
                    logger_accuracy.info(
                        'test_auc: %.4f -test_loss: %.4f -test_accuracy: %.4f -test_aux_loss: %.4f -loss_without_aux: %.4f *best_auc: %.4f \r\n' % (
                            test_auc_log, loss_sum_log, accuracy_sum_log, aux_loss_sum_log, loss_without_aux, best_auc))
                    print(
                        'test_auc: %.4f - test_loss: %.4f - test_accuracy: %.4f - test_aux_loss: %.4f - loss_without_aux: %.4f  *best_auc: %.4f' %
                        (test_auc_log, loss_sum_log, accuracy_sum_log, aux_loss_sum_log, loss_without_aux, best_auc))

                    logger_test_loss.info(
                        '%d,%.4f,%.4f,%.4f' % \
                        (iter, loss_sum_log, aux_loss_sum_log, loss_without_aux))

                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                    # if test_auc_log > 0.87:
                    #     test_iter = 10
                    # if iter >= test_iter:
                    #     test_iter = 10
                # if iter == 2500:
                #     test_iter = 100
                # if iter == 6000:
                #     breakflag = True
                #     break


                if (iter % save_iter) == 0:
                    print('save model iter: %d' % (iter))
                    model.save(sess, model_path + "--" + str(iter))
                # if iter == 3000:
                #     lr *= 2

            test_time = time.time()
            print("test interval: " + str((test_time - start_time) / 60.0) + " min")
            logger_accuracy.info("test interval: " + str((test_time - start_time) / 60.0) + " min" + "\r\n")

        logger_accuracy.info("end_time:" + time.asctime(time.localtime(time.time())) + "\r\n")


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
    SEED = 4
    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    train(seed=SEED, datasetdir=sys.argv[1])

