'''This module is used to perform a query with the model. 

    This part is very messy but still runnable. 

    @author: Tony Hong
'''
# coding: utf-8

import os
import sys
import time
import cPickle
import gzip
import random

import numpy

# Configuration of environment
SRC_DIR = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))
sys.path.append(SRC_DIR)

import model_builder
import config
import utils

MODEL_PATH = config.MODEL_VERSION


def query(model_name, experiment_name, inputs, target):
    MODEL_NAME = experiment_name
    description = model_builder.load_description(MODEL_PATH, MODEL_NAME)

    net = model_builder.build_model(model_name, description)
    net.load(MODEL_PATH, MODEL_NAME, description)

    # net.model.summary()
    # print net.model.get_layer(name="embedding_2").get_weights()[0]

    print net.role_vocabulary
    print("unk_word_id", net.unk_word_id)
    print("missing_word_id", net.missing_word_id)
    # net.set_0_bias()

    net.model.summary()

    propbank_map = {
        "subj"  :   "A0",
        "obj"   :   "A1",
        "ARG0"  :   "A0",
        "ARG1"  :   "A1",
        "ARG2"  :   "A2",
    }

    # tr_map = {
    #     "A0": numpy.asarray([[net.role_vocabulary["A0"]]], dtype=numpy.int64),
    #     "A1": numpy.asarray([[net.role_vocabulary["A1"]]], dtype=numpy.int64),
    #     "A2": numpy.asarray([[net.role_vocabulary["<UNKNOWN>"]]], dtype=numpy.int64)
    # }

    # net.word_vocabulary["<NOTHING>"] = net.missing_word_id
    # net.role_vocabulary["<UNKNOWN>"] = net.unk_role_id    

    reverse_vocabulary = utils.get_reverse_map(net.word_vocabulary)
    reverse_role_vocabulary = utils.get_reverse_map(net.role_vocabulary)    

    print reverse_role_vocabulary

    raw_words = dict((reverse_role_vocabulary[r], reverse_vocabulary[net.missing_word_id]) for r in net.role_vocabulary.values())

    # print raw_words

    raw_words.update(inputs)
    
    # print raw_words
    # print len(raw_words)
    assert len(raw_words) == len(net.role_vocabulary)
    # print repr(raw_words)

    # n = int(sys.argv[3])    
    t_r = [net.role_vocabulary.get(r, net.unk_role_id) for r in target.keys()]
    t_w = [net.word_vocabulary.get(w, net.unk_word_id) for w in target.values()]

    input_roles_words = {}
    for r, w in raw_words.items():
        input_roles_words[net.role_vocabulary[r]] = utils.input_word_index(net.word_vocabulary, w, net.unk_word_id, warn_unk=True)

    print input_roles_words, t_r
    input_roles_words.pop(t_r[0])

    # default_roles_words = dict((r, net.missing_word_id) for r in (net.role_vocabulary.values()))
    # default_roles_words.update(input_roles_words)
    # input_roles_words = default_roles_words
        
    x_w_i = numpy.asarray([input_roles_words.values()], dtype=numpy.int64)
    x_r_i = numpy.asarray([input_roles_words.keys()], dtype=numpy.int64)
    y_w_i = numpy.asarray(t_w, dtype=numpy.int64)
    y_r_i = numpy.asarray(t_r, dtype=numpy.int64)

    topN=20
    predicted_word_indices = net.top_words(x_w_i, x_r_i, y_w_i, y_r_i, topN)
    # print predicted_word_indices
    # print len(predicted_word_indices)

    print(x_w_i, x_r_i, y_w_i, y_r_i)

    p_w = net.p_words(x_w_i, x_r_i, y_w_i, y_r_i, batch_size=1, verbose=0)[0]
    print ('p_t_w: ', p_w)

    resultlist = predicted_word_indices
    # print resultlist

    for i, t_w_i in enumerate(resultlist):
        t_w = net.word_vocabulary.get(t_w_i, net.unk_word_id)
        y_w_i = numpy.asarray([t_w_i], dtype=numpy.int64)
        p = net.p_words(x_w_i, x_r_i, y_w_i, y_r_i, batch_size=1, verbose=0)[0]
        n = numpy.round(p / 0.005)
        fb = numpy.floor(n)
        hb = n % 2
        print u"{:<5} {:7.6f} {:<20} ".format(i+1, float(p), reverse_vocabulary[int(t_w_i)]) + u"\u2588" * int(fb) + u"\u258C" * int(hb)
    

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        sys.exit("Model name input argument missing missing")
    
    if len(sys.argv) > 2:
        experiment = sys.argv[2]
    else:
        sys.exit("Experiment input argument missing missing")

    experiment_name = model_name + '_' + experiment

    if len(sys.argv) > 3:
        inputs = eval(sys.argv[3])
    else:
        sys.exit('Inputs argument missing missing. Example: ' +  "{'V':'eat', 'A1':'cat'}")

    if len(sys.argv) > 4:
        target = eval(sys.argv[4])
    else:
        sys.exit('Target argument missing missing. Example: ' +  "{'A0':'tiger'}")

    query(model_name, experiment_name, inputs, target)
