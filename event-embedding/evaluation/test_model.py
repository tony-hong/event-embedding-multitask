# test_model.py 

import re
import os
import sys
import time
import cPickle
import gzip
import random

import numpy as np
from keras.models import Model

# Configuration of environment
SRC_DIR = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))
sys.path.append(SRC_DIR)

import config, utils, model_builder
from batcher import generator, get_minibatch

DATA_PATH = config.DATA_VERSION
MODEL_PATH = config.MODEL_VERSION


def stats(net, confusionM):
    # print ("Confusion Matrix: ")
    # print "    A0,  A1, LOC, TMP, MNR,   V, <UNKNOWN>"
    # print (confusionM)
    vid = net.role_vocabulary['V']
    matrix = confusionM
    # matrix = np.delete(confusionM, vid, 0)
    # matrix = np.delete(matrix, vid, 1)

    predicted = matrix.sum(axis=0) * 1.
    gold = matrix.sum(axis=1) * 1.
    TP = np.diag(matrix) * 1.
    FP = predicted - TP
    FN = gold - TP

    precision = TP / predicted * 100
    recall = TP / gold * 100
    F1 = (2 * precision * recall) / (precision + recall)
    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0
    F1[np.isnan(F1)] = 0

    # print (precision)
    # print (recall)
    # print (F1)

    dir_P = TP[:-1].sum() / predicted[:-1].sum() * 100
    dir_R = TP[:-1].sum() / gold.sum() * 100
    dir_F1 = (2 * dir_P * dir_R) / (dir_P + dir_R)

    # avg_P = (precision[:-1] * gold[:-1] / gold.sum()).sum()
    # avg_R = (recall[:-1] * gold[:-1] / gold.sum()).sum()
    # avg_F1 = (2 * avg_P * avg_R) / (avg_P + avg_R)

    print "Dir: %.2f \t %.2f \t %.2f" % (dir_P, dir_R, dir_F1)
    # print "Avg: %.2f \t %.2f \t %.2f" % (avg_P, avg_R, avg_F1)

    return dir_P, dir_R, dir_F1, precision, recall, F1


def evaluate(model_name, experiment_name, batch_size):
    MODEL_NAME = experiment_name
    repr_file = os.path.join(MODEL_PATH, 'confusionM_' + MODEL_NAME)

    description = model_builder.load_description(MODEL_PATH, MODEL_NAME)
    net = model_builder.build_model(model_name, description)
    net.load(MODEL_PATH, MODEL_NAME, description)

    n_roles = len(net.role_vocabulary)
    print net.role_vocabulary
    print("unk_word_id", net.unk_word_id)
    print("missing_word_id", net.missing_word_id)

    net.model.summary()
    print net.model.metrics_names
    reverse_role_vocabulary = utils.get_reverse_map(net.role_vocabulary)

    test_sample_size = config.OCT_TEST_SIZE
    test_steps = test_sample_size / batch_size
    # # DEBUG
    # test_steps = 10

    print 'Testing...'
    test_start = time.clock()

    # Always use generator in Keras
    if re.search('NAME_WHICH_YOU_NEED_OLD_BATCHER', experiment_name):
        test_gen = get_minibatch(DATA_PATH + "NN_test", net.unk_word_id, net.unk_role_id, net.missing_word_id, 
                n_roles, random=False, batch_size=batch_size)
    else:
        test_gen = generator(DATA_PATH + "NN_test", model_name, net.unk_word_id, net.unk_role_id, net.missing_word_id, 
                n_roles, random=False, batch_size=batch_size)
    
    # Test the model
    test_result = net.model.evaluate_generator(
            generator = test_gen,
            steps = test_steps, 
            max_q_size = 1, 
            workers = 1, 
            pickle_safe = False
        )
    print ('test_result', test_result)

    # Compute confusion matrix
    metrics_names = net.model.metrics_names
    result_dict = {(x, 0) for x in metrics_names}
    batch_n = 0
    confusionM = np.zeros((n_roles, n_roles), dtype='int32')
    ppl_role_list = dict()
    ppl_role = dict()

    result_list = []
    for ([i_w, i_r, t_w, t_r], _) in generator(DATA_PATH + "NN_test", model_name, 
            net.unk_word_id, net.unk_role_id, net.missing_word_id, n_roles, random=False, batch_size=batch_size):
        result_role = net.predict_role(i_w, i_r, t_w, t_r, batch_size)
        result_word_likelihood = net.predict(i_w, i_r, t_w, t_r, batch_size)[0]
        neg_log_likelihoods = -np.log(result_word_likelihood)

        for i, row in enumerate(neg_log_likelihoods, start=0):
            target_word = t_w[i][0]
            target_role = t_r[i][0]
            neg_log_likelihood = row[target_word]
            ppl_role_list.setdefault(target_role, []).append(neg_log_likelihood)

        for i, true_r in enumerate(t_r, start=0):                
            confusionM[true_r, result_role[i]] += 1
            if true_r == result_role[i]:
                result_list.append(1)
        batch_n += 1
        print batch_n
        if batch_n >= test_steps:
            break

    for k, v in ppl_role_list.items():
        neg_log_likelihood_role = np.mean(np.array(v))
        ppl_role[k] = np.exp(neg_log_likelihood_role)

    print "Confusion Matrix: "
    print "    A0,  A1, LOC, TMP, MNR,   V, <UNKNOWN>"
    print confusionM
    np.savetxt('confusionM_' + experiment_name + '.csv', confusionM, delimiter = ',')
    np.savetxt('result_list_' + experiment_name + '.csv', result_list, delimiter = ',')

    stats(net, confusionM)

    print "Loss(neg_log_likelihood) by role: "
    for r in ppl_role.keys():
        print (reverse_role_vocabulary[r], np.log(ppl_role[r]))

    print "PPL by role: "
    for r in ppl_role.keys():
        print (reverse_role_vocabulary[r], ppl_role[r])

    with open(repr_file, 'w') as f_out:
        f_out.write('[')
        for i in range(n_roles): 
            f_out.write('[')
            for j in range(n_roles):
                f_out.write(str(confusionM[i][j]) + ", ")
            f_out.write('] \n')
        f_out.write(']')

    test_end = time.clock()
    print 'test time: %f, sps: %f' % (test_end - test_start, test_steps * batch_size / (test_end - test_start))


if __name__ == "__main__":
    batch_size = config.BATCH_SIZE
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        sys.exit("Model name input argument missing missing")
    
    if len(sys.argv) > 2:
        experiment = sys.argv[2]
    else:
        sys.exit("Experiment input argument missing missing")

    if len(sys.argv) > 3:
        batch_size = int(sys.argv[3])

    experiment_name = model_name + '_' + experiment

    evaluate(model_name, experiment_name, batch_size)
