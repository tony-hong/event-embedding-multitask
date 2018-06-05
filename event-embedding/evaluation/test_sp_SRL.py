import re
import os
import sys
import time
import cPickle
import gzip
import random

import numpy as np
import scipy.stats as st
from keras.models import Model

# Configuration of environment
SRC_DIR = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))
sys.path.append(SRC_DIR)

import config, utils, model_builder
from batcher import get_MT_batch, random_lines

DATA_PATH = config.DATA_VERSION
MODEL_PATH = config.MODEL_VERSION
EVAL_PATH = os.path.join(config.EVAL_PATH, 'CoNLL05/')



def data_gen(file_name, model_name, net, batch_size, VR_SP_SRL=True, random_gen=False, rng=None):
    """Generates k noise samples for target role + 1 positive sample from data. Noise and positive samples share inputs"""
    unk_word_id, unk_role_id, missing_word_id = net.unk_word_id, net.unk_role_id, net.missing_word_id
    word_vocabulary, role_vocabulary = net.word_vocabulary, net.role_vocabulary

    n_roles = len(role_vocabulary)
    get_batch = get_MT_batch

    while True:
        with open(file_name, 'r') as f:
            x_w_i = []
            x_r_i = []
            y_w_i = []
            y_r_i = []
            n_total_samples = 0
            n_neg_samples = 0

            if random_gen:
                file_length = os.stat(file_name).st_size
                lines = random_lines(f, file_length, batch_size, rng)
            else:
                lines = f

            for line in lines:
                d = eval(line)
                roles, words = map(list, zip(*d.items()))
                vid = role_vocabulary['V']
                
                # SRC for verb-head pair
                if VR_SP_SRL:
                    target_role_name = [r for r in roles if r != 'V'][0]
                    target_word_name = d.get(target_role_name, '<unknown>')

                    input_role = vid
                    target_role = role_vocabulary.get(target_role_name, unk_role_id)

                    event = dict()
                    for r, rid in role_vocabulary.items():
                        if rid != target_role:
                            input_word = d.get(r, '<MISSING>')
                            event[rid] = word_vocabulary.get(input_word, missing_word_id)
                    event_roles, event_words = map(list, zip(*event.items()))

                    # Positive sample
                    # Remove current role-word pair from context ...
                    input_words = event_words[:]
                    input_roles = event_roles[:]

                    # ... and set it as target
                    target_roles = [target_role]
                    target_words = [word_vocabulary.get(target_word_name, unk_word_id)]

                    x_w_i.append(input_words)
                    x_r_i.append(input_roles)
                    y_w_i.append(target_words)
                    y_r_i.append(target_roles)
                    n_total_samples += 1

                else:
                    # PRD only for SRC
                    if re.search('PRD', model_name):
                        target_role_names = [r for r in roles if r != 'V']
                        target_word_idxs = [i for i, r in enumerate(target_role_names) if d.get(r)]

                        # print target_role_names, target_word_idxs

                        for i in target_word_idxs:
                            target_role_name = target_role_names[i]
                            target_role = role_vocabulary.get(target_role_name, unk_role_id)

                            target_word_name = d.get(target_role_name, '<unknown>')
                            target_word = word_vocabulary.get(target_word_name, unk_word_id)

                            # print target_word_name, target_role_name

                            event = dict()
                            for r, rid in role_vocabulary.items():
                                if rid not in [target_role]: 
                                    input_word = d.get(r, '<MISSING>')
                                    event[rid] = word_vocabulary.get(input_word, missing_word_id)
                            event_roles, event_words = map(list, zip(*event.items()))

                            input_words = event_words[:]
                            input_roles = event_roles[:]

                            # ... and set it as target
                            target_roles = [target_role]
                            target_words = [target_word]

                            x_w_i.append(input_words)
                            x_r_i.append(input_roles)
                            y_w_i.append(target_words)
                            y_r_i.append(target_roles)
                            n_total_samples += 1

                    else:
                        target_role_names = [r for r in roles if r != 'V']
                        target_word_idxs = [i for i, r in enumerate(target_role_names) if d.get(r)]

                        # print target_role_names, target_word_idxs

                        for i in target_word_idxs:
                            target_role_name = target_role_names[i]
                            target_role = role_vocabulary.get(target_role_name, unk_role_id)

                            target_word_name = d.get(target_role_name, '<unknown>')
                            target_word = word_vocabulary.get(target_word_name, unk_word_id)

                            prd_role = vid
                            prd_word_name = d.get('V', '<unknown>')
                            prd_word = word_vocabulary.get(prd_word_name, unk_word_id)

                            # print target_word_name, target_role_name

                            event = dict()
                            for r, rid in role_vocabulary.items():
                                if rid not in [target_role, vid]: 
                                    input_word = d.get(r, '<MISSING>')
                                    event[rid] = word_vocabulary.get(input_word, missing_word_id)
                            event_roles, event_words = map(list, zip(*event.items()))

                            input_words = event_words[:]
                            input_roles = event_roles[:]

                            input_words.append(prd_word)
                            input_roles.append(prd_role)

                            # ... and set it as target
                            target_roles = [target_role]
                            target_words = [target_word]

                            x_w_i.append(input_words)
                            x_r_i.append(input_roles)
                            y_w_i.append(target_words)
                            y_r_i.append(target_roles)
                            n_total_samples += 1

                # print x_w_i, x_r_i, y_w_i, y_r_i

                if len(x_w_i) >= batch_size:
                    # print x_w_i[-1]
                    # print (os.getpid(), x_w_i[batch_size-1])

                    yield (get_batch(x_w_i, x_r_i, y_w_i, y_r_i))

                    x_w_i = []
                    x_r_i = []
                    y_w_i = []
                    y_r_i = []
                    n_total_samples = 0
                    n_neg_samples = 0



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

    avg_P = (precision[:-1] * gold[:-1] / gold.sum()).sum()
    avg_R = (recall[:-1] * gold[:-1] / gold.sum()).sum()
    avg_F1 = (2 * avg_P * avg_R) / (avg_P + avg_R)


    print ("gold: ", gold)
    print ("gold_sum: ", gold.sum())
    print "Dir: %.2f \t %.2f \t %.2f" % (dir_P, dir_R, dir_F1)
    print "Avg: %.2f \t %.2f \t %.2f" % (avg_P, avg_R, avg_F1)

    return dir_P, dir_R, dir_F1, precision, recall, F1



def evaluate(model_name, experiment_name, test_name, batch_size, VR_SP_SRL=True, bootstrapping=False, majority_baseline=False):
    MODEL_NAME = experiment_name
    # repr_file = os.path.join(MODEL_PATH, 'confusionM_' + MODEL_NAME)

    description = model_builder.load_description(MODEL_PATH, MODEL_NAME)
    net = model_builder.build_model(model_name, description)
    net.load(MODEL_PATH, MODEL_NAME, description)

    n_roles = len(net.role_vocabulary)
    reverse_word_vocabulary = utils.get_reverse_map(net.word_vocabulary)
    reverse_role_vocabulary = utils.get_reverse_map(net.role_vocabulary)
    # net.set_0_bias()

    print net.role_vocabulary
    print("unk_word_id", net.unk_word_id)
    print("missing_word_id", net.missing_word_id)

    net.model.summary()

    # print net.model.metrics_names
    
    test_sample_size = 0
    with open(EVAL_PATH + test_name, 'r') as lines:
        for l in lines:
            test_sample_size += 1
    print (test_sample_size)

    test_steps = test_sample_size / float(batch_size)
    # test_steps = test_sample_size
    # # DEBUG
    # test_steps = 10

    print 'Testing ' + test_name + ' ...'
    print 'VR_SP_SRL: ' + str(VR_SP_SRL)
    test_start = time.clock()

    # if re.search('NNRF_1e8', experiment_name) or re.search('MTRF_dev', experiment_name):
    #     test_gen = get_minibatch(DATA_PATH + "NN_test", net.unk_word_id, net.unk_role_id, net.missing_word_id, 
    #             n_roles, random=False, batch_size=batch_size)
    # else:
    #     test_gen = generator(DATA_PATH + "NN_test", model_name, net.unk_word_id, net.unk_role_id, net.missing_word_id, 
    #             n_roles, random=False, batch_size=batch_size)
    
    # # Test the model
    # test_result = net.model.evaluate_generator(
    #         generator = test_gen,
    #         steps = test_steps, 
    #         max_q_size = 1, 
    #         workers = 1, 
    #         pickle_safe = False
    #     )
    # print ('test_result', test_result)

    # Compute confusion matrix
    metrics_names = net.model.metrics_names
    result_dict = {(x, 0) for x in metrics_names}
    batch_n = 0
    confusionM = np.zeros((n_roles, n_roles), dtype='int32')
    ppl_role_list = dict()

    result_list = []
    output_list = []
    for ([i_w, i_r, t_w, t_r], _) in data_gen(EVAL_PATH + test_name, model_name, 
            net, batch_size, VR_SP_SRL=VR_SP_SRL):
        # zeros = np.zeros(t_r.shape)
        result_role = net.predict_role(i_w, i_r, t_w, t_r, batch_size)
        
        # word_emb, avg_emb, event_emb = net.avg_emb.predict([i_w, i_r, t_w, t_r], batch_size)
        # print word_emb.shape, avg_emb.shape, event_emb.shape
        # assert np.multiply(word_emb[0][0], avg_emb[0])[0] == event_emb[0][0][0]
        # assert np.multiply(word_emb[0][0], avg_emb[0])[1] == event_emb[0][0][1]


        # test role prediction of MTRF_dev, result: role prediction is useless
        # print i_r
        # print t_r.reshape(-1)
        # print result_role

        # result_word_likelihood = net.predict(i_w, i_r, t_w, t_r, batch_size)[0]
        # neg_log_likelihoods = -np.log(result_word_likelihood)

        # for i, row in enumerate(neg_log_likelihoods, start=0):
        #     target_word = t_w[i][0]
        #     target_role = t_r[i][0]
        #     neg_log_likelihood = row[target_word]
        #     ppl_role_list.setdefault(target_role, []).append(neg_log_likelihood)

        # print i_w, i_r, t_w, t_r

        for i, true_r in enumerate(t_r, start=0):
            # if reverse_role_vocabulary.get(t_r[0][0], '<unknown>') == 'AM-LOC':
            #     print ("input words", [reverse_word_vocabulary.get(w, '<unknown>') for w in i_w[0]])    
            #     print ("input roles", [reverse_role_vocabulary.get(r, '<unknown>') for r in i_r[0]])
            #     print ("target word", [reverse_word_vocabulary.get(w, '<unknown>') for w in t_w[0]])
            #     print ("target role", [reverse_role_vocabulary.get(r, '<unknown>') for r in t_r[0]])
            #     print ("predicted role", [reverse_role_vocabulary.get(result_role[i], '<unknown>') for r in t_r[0]])
            #     print ''

            confusionM[true_r, result_role[i]] += 1
            if true_r == result_role[i]:
                result_list.append(1)
            output_list.append((true_r, result_role[i]))
        batch_n += 1
        if batch_n % 100 == 0:
            print (batch_n)
        if batch_n >= test_steps:
            break

    # ppl_role = dict()
    # for k, v in ppl_role_list.items():
    #     neg_log_likelihood_role = np.mean(np.array(v))
    #     ppl_role[k] = np.exp(neg_log_likelihood_role)

    # obtain ZeroR baseline
    print confusionM
    majority = 1
    if majority_baseline == True:
        for i in range(7):
            confusionM[i][majority] = confusionM[i][:].sum()
            confusionM[i][majority-1] = 0 
            confusionM[i][majority+1:] = 0 
    print confusionM

    dir_P, dir_R, dir_F1, precision, recall, F1 = stats(net, confusionM)
    print "Dir: %.2f \t %.2f \t %.2f" % (dir_P, dir_R, dir_F1)

    # np.savetxt('confusionM_' + experiment_name + '.' + test_name.strip('.dat') + '.csv', confusionM, delimiter = ',')
    # np.savetxt('output_' + experiment_name + '.' + test_name.strip('.dat') + '.csv', output_list, delimiter = ',')

    # with open(repr_file, 'w') as f_out:
    #     f_out.write('[')
    #     for i in range(n_roles): 
    #         f_out.write('[')
    #         for j in range(n_roles):
    #             f_out.write(str(confusionM[i][j]) + ", ")
    #         f_out.write('] \n')
    #     f_out.write(']')

    # print "Loss(neg_log_likelihood) by role: "
    # for r in ppl_role.keys():
    #     print (reverse_role_vocabulary[r], np.log(ppl_role[r]))

    print ("Result by role: ")
    for r in range(len(precision)):
        print ('%s: \t %.2f \t %.2f \t %.2f' % (reverse_role_vocabulary[r], precision[r], recall[r], F1[r]))



    test_end = time.clock()
    print 'test time: %f, sps: %f' % (test_end - test_start, test_steps * batch_size / (test_end - test_start))

    if bootstrapping:
        P_mean, P_std, R_mean, R_std, F1_mean, F1_std = bootstrap(experiment_name, test_name, net, n_roles, output_list=output_list)

        return P_mean, P_std, R_mean, R_std, F1_mean, F1_std



def bootstrap(experiment_name, test_name, net, n_roles, output_list=None, alpha=0.99, n_b=100):
    print ('Bootstrap method: ')
    if not output_list:
        output_list = np.loadtxt('output_' + experiment_name + '.' + test_name.strip('.dat') + '.csv', delimiter = ',')

    rng = np.random
    P_list, R_list, F1_list = [], [], []
    for i in range(n_b):
        rand_len = len(output_list)
        indices = rng.randint(0, rand_len, rand_len)
        conMat = np.zeros((n_roles, n_roles))
        for idx in indices:
            x = int(output_list[idx][0])
            y = int(output_list[idx][1])
            conMat[x, y] += 1
        dir_P, dir_R, dir_F1, precision, recall, F1 = stats(net, conMat)
        P_list.append(dir_P)
        R_list.append(dir_R)
        F1_list.append(dir_F1)

    P_array, R_array, F1_array = np.array(P_list), np.array(R_list), np.array(F1_list)
    P_mean, P_std = np.mean(P_array), np.std(P_array)
    R_mean, R_std = np.mean(R_array), np.std(R_array)
    F1_mean, F1_std = np.mean(F1_array), np.std(F1_array)
    P_t_int = st.t.interval(alpha, len(P_array)-1, loc=np.mean(P_array), scale=st.sem(P_array))
    R_t_int = st.t.interval(alpha, len(R_array)-1, loc=np.mean(R_array), scale=st.sem(R_array))
    F1_t_int = st.t.interval(alpha, len(F1_array)-1, loc=np.mean(F1_array), scale=st.sem(F1_array))

    print ('Precision: \t %.2f \t %.2f \t %.2f \t %.2f' % (P_mean, P_std, P_t_int[0], P_t_int[1]))
    print ('Recall:    \t %.2f \t %.2f \t %.2f \t %.2f' % (R_mean, R_std, R_t_int[0], R_t_int[1]))
    print ('F1:        \t %.2f \t %.2f \t %.2f \t %.2f' % (F1_mean, F1_std, F1_t_int[0], F1_t_int[1]))
    
    return P_mean, P_std, R_mean, R_std, F1_mean, F1_std



if __name__ == "__main__":
    batch_size = 1
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

    # print ('SP_SRL: ')
    evaluate(model_name, experiment_name, 'SP_SRL.wsj.unique_pairs.dat', batch_size, VR_SP_SRL=True, bootstrapping=False)
    evaluate(model_name, experiment_name, 'SP_SRL.brown.unique_pairs.dat', batch_size, VR_SP_SRL=True, bootstrapping=False)

    # print ('SRL full input: ')
    # config.SRC = True
    # evaluate(model_name, experiment_name, 'SP_SRL.wsj.plain.dat', batch_size, VR_SP_SRL=not config.SRC)
    # evaluate(model_name, experiment_name, 'SP_SRL.brown.plain.dat', batch_size, VR_SP_SRL=not config.SRC)
    # evaluate(model_name, experiment_name, 'SP_SRL.wsj_brown.plain.dat', batch_size, VR_SP_SRL=not config.SRC)
    # config.SRC = False
