import os
import sys
import time
import cPickle
import gzip
import random

import numpy
from scipy.stats import spearmanr

# Configuration of environment
SRC_DIR = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))
sys.path.append(SRC_DIR)

import model_builder
import config
import utils

MODEL_PATH = config.MODEL_VERSION
RF_EVAL_PATH = os.path.join(config.EVAL_PATH, 'rf/')



def eval_bicknell_switch(model_name, experiment_name, evaluation, model=None, print_result=True, switch_test=False):
    MODEL_NAME = experiment_name

    if model:
        net = model
    else:
        description = model_builder.load_description(MODEL_PATH, MODEL_NAME)        
        net = model_builder.build_model(model_name, description)
        net.load(MODEL_PATH, MODEL_NAME, description)

    bias = net.set_0_bias()

    if print_result:
        print net.role_vocabulary

    eval_data_file = os.path.join(RF_EVAL_PATH, evaluation + '.txt')

    result_file = os.path.join(MODEL_PATH, MODEL_NAME + '_' + evaluation + '.txt')

    probs = []
    baseline = []
    oov_count = 0

    if print_result:
        print eval_data_file
        print "="*60

    dataset = numpy.genfromtxt(eval_data_file, dtype=str, delimiter='\t', usecols=[0,1,2,3,4])

    samples = []
    i = 0
      
    while True:
        d = dataset[i]
        d2 = dataset[i+1]
        
        A0 = d[0][:-2]
        V = d[1][:-2]
        assert d2[0][:-2] == A0
        assert d2[1][:-2] == V
        
        if d[3] == 'yes':
            assert d2[3] == 'no'
            A1_correct = d[2][:-2]
            A1_incorrect = d2[2][:-2]
            b_correct = d[4]
            b_incorrect = d2[4]
        else:
            assert d[3] == 'no'
            A1_correct = d2[2][:-2]
            A1_incorrect = d[2][:-2]
            b_correct = d2[4]
            b_incorrect = d[4]
        
        if A1_correct not in net.word_vocabulary or A1_incorrect not in net.word_vocabulary:
            if A1_correct not in net.word_vocabulary and print_result:
                print "%s MISSING FROM VOCABULARY. SKIPPING..." % A1_correct
            if A1_incorrect not in net.word_vocabulary and print_result:
                print "%s MISSING FROM VOCABULARY. SKIPPING..." % A1_incorrect
        else:
            roles = net.role_vocabulary.values()
            del roles[net.unk_role_id]

            input_roles_words = dict((r, net.missing_word_id) for r in (roles))
        
            input_roles_words[net.role_vocabulary["A0"]] = utils.input_word_index(net.word_vocabulary, A0, net.unk_word_id, warn_unk=True)
            input_roles_words[net.role_vocabulary["V"]] = utils.input_word_index(net.word_vocabulary, V, net.unk_word_id, warn_unk=True)
        
            sample = (
                numpy.asarray([input_roles_words.values(), input_roles_words.values()], dtype=numpy.int64),     # x_w_i
                numpy.asarray([input_roles_words.keys(), input_roles_words.keys()], dtype=numpy.int64),         # x_r_i
                numpy.asarray([net.word_vocabulary[A1_correct], net.word_vocabulary[A1_incorrect]], dtype=numpy.int64),   # y_i (1st is correct and 2nd is incorrect
                numpy.asarray([net.role_vocabulary["A1"], net.role_vocabulary["A1"]], dtype=numpy.int64),                           # y_r_i
                [b_correct, b_incorrect],                                                                       # bicknell scores
                "\"" + A0 + " " + V + "\"", # context
                [A1_correct, A1_incorrect]
            )

            samples.append(sample)
        
        i += 2
        if i > len(dataset) - 2:
            break

    num_samples = len(samples)
    num_correct = 0
    num_total = 0
    
    if print_result:
        print "context", "correct", "incorrect", "P(correct)", "P(incorrect)", "bicnell_correct", "bicnell_incorrect"
    
    result_list = []
    
    for x_w_i, x_r_i, y_w_i, y_r_i, bicknell, context, a1 in samples:
        
        p = net.p_words(x_w_i, x_r_i, y_w_i, y_r_i)
                
        p_correct = p[0]
        p_incorrect = p[1]
            
        if print_result:
            print context, a1[0], a1[1], p_correct, p_incorrect, bicknell[0], bicknell[1]
        
        if p_correct > p_incorrect:
            result_list.append(1)
        else:
            result_list.append(0)

        num_correct += p_correct > p_incorrect
        num_total += 1
    
    assert num_total == num_samples
    
    accuracy = float(num_correct)/float(num_samples)

    if print_result:
        print "Number of lines %d" % num_samples
        print "Baseline Lenci11 is 43/64=0.671875"
        print "Final score of theano model is %d/%d=%.6f" % (num_correct, num_samples, accuracy)
        
    print result_list


    if switch_test and print_result:
        print "\nSwitch A0/A1 TEST"

        input_words = []
        input_roles = []
        for i in range(1):
            roles = net.role_vocabulary.values()
            print net.unk_role_id
            roles.remove(net.unk_role_id)

            input_role_word_pairs = dict((r, net.missing_word_id) for r in roles)
            input_role_word_pairs[net.role_vocabulary["V"]] = utils.input_word_index(net.word_vocabulary, "buy", net.unk_word_id, warn_unk=True)

            input_words.append(input_role_word_pairs.values())
            input_roles.append(input_role_word_pairs.keys())

        man = utils.input_word_index(net.word_vocabulary, "man", net.unk_word_id, warn_unk=True)
        car = utils.input_word_index(net.word_vocabulary, "car", net.unk_word_id, warn_unk=True)
        a1 = net.role_vocabulary["A1"]
        a0 = net.role_vocabulary["A0"]
        
        a0_test = (
                    numpy.asarray(input_words, dtype=numpy.int64),
                    numpy.asarray(input_roles, dtype=numpy.int64),
                    numpy.asarray([man, car], dtype=numpy.int64),
                    numpy.asarray([a0], dtype=numpy.int64),
        )
        x_w_i, x_r_i, y_w_i, y_r_i = a0_test
        p0 = net.p_words(x_w_i, x_r_i, y_w_i, y_r_i)
        print p0

        a1_test = (
                    numpy.asarray(input_words, dtype=numpy.int64),
                    numpy.asarray(input_roles, dtype=numpy.int64),
                    numpy.asarray([man, car], dtype=numpy.int64),
                    numpy.asarray([a1], dtype=numpy.int64),
        )
        x_w_i, x_r_i, y_w_i, y_r_i = a1_test
        p1 = net.p_words(x_w_i, x_r_i, y_w_i, y_r_i)
        print p1

        print "man buy", p0[0]
        print "buy man", p1[0]
        print "car buy", p0[1]
        print "buy car", p1[1]

    net.set_bias(bias)

    return num_correct, num_samples, accuracy


if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        sys.exit("Model name input argument missing missing")
    
    if len(sys.argv) > 2:
        experiment_version = sys.argv[2]
    else:
        sys.exit("Experiment input argument missing missing")

    experiment_name = model_name + '_' + experiment_version

    eval_bicknell_switch(model_name, experiment_name, 'bicknell')



