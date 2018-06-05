# coding: utf-8

import os
import sys
import time
import cPickle
import gzip
import random

import numpy
from scipy.stats import spearmanr
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

# Configuration of environment
SRC_DIR = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))
sys.path.append(SRC_DIR)

import model_builder
import config

MODEL_PATH = config.MODEL_VERSION
EVAL_PATH = os.path.join(config.EVAL_PATH, 'single/')
RV_EVAL_PATH = os.path.join(config.EVAL_PATH, 'rv/')

wnl = WordNetLemmatizer()


def eval_MNR_LOC(model_name, experiment_name, evaluation, model=None, print_result=True, skip_header=False):
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

    tr_map = {
        "ARG2": "A2",
        "ARG3": "A3",
        "ARGM-MNR": "AM-MNR",
        "ARGM-LOC": "AM-LOC",
    }

    if evaluation == "AM-MNR":    
        eval_data_file = os.path.join(RV_EVAL_PATH, 'McRaeInstr-fixed' + '.txt')
        remove_suffix=False
    elif evaluation == 'AM-LOC':
        eval_data_file = os.path.join(RV_EVAL_PATH, 'McRaeLoc-fixed' + '.txt')
        remove_suffix=True
    else:
        sys.exit('No such evaluation!!!')


    result_file = os.path.join(MODEL_PATH, MODEL_NAME + '_' + evaluation + '.txt')

    
    probs = []
    baseline = []
    oov_count = 0

    r_i = net.role_vocabulary["V"]

    if print_result:
        print eval_data_file, evaluation
        print "="*60

    with open(eval_data_file, 'r') as f, \
         open(result_file, 'w') as f_out:
        for i, line in enumerate(f):
            if i == 0 and skip_header:
                # print line.strip() + "\tP(instrument|verb)"
                continue #skip header
            line = line.strip()
            w, tw, temp1, temp2 = line.split()[:4] # input word, target word, other stuff
            w = w[:-2] if remove_suffix else w
            tw = tw[:-2] if remove_suffix else tw

            w = wnl.lemmatize(w.lower(), wn.VERB)
            tw = wnl.lemmatize(tw.lower(), wn.NOUN)

            w_i = net.word_vocabulary.get(w, net.unk_word_id)
            tw_i = net.word_vocabulary.get(tw, net.unk_word_id)

            if evaluation == "AM-MNR":
                r = temp2
            else: 
                r = temp1

            # tr_i = net.role_vocabulary.get(evaluation, net.unk_role_id)
            tr_i = net.role_vocabulary.get(tr_map[r], net.unk_role_id)
            y_r_i = numpy.asarray([tr_i], dtype=numpy.int64)

            if tw_i == net.unk_word_id:
                oov_count += 1
                print w, tw
                f_out.write(line + "\tnan\n")
                continue

            b = float(line.split()[-1 if remove_suffix else -2])
            baseline.append(b)

            input_roles_words = dict((r, net.missing_word_id) for r in (net.role_vocabulary.values() + [net.unk_role_id]))
            input_roles_words[r_i] = w_i
            input_roles_words.pop(tr_i, None)

            x_w_i = numpy.asarray([input_roles_words.values()], dtype=numpy.int64)
            x_r_i = numpy.asarray([input_roles_words.keys()], dtype=numpy.int64)

            y_w_i = numpy.asarray([tw_i], dtype=numpy.int64)

            p = net.p_words(x_w_i, x_r_i, y_w_i, y_r_i)
            # pr = net.p_roles(x_w_i, x_r_i, y_w_i, y_r_i)

            probs.append(p)
        
            f_out.write(line + "\t%s\n" % p)

    rho, p_value = spearmanr(baseline, probs)
    rating = len(probs)

    if print_result:
        print "Spearman correlation: %f; 2-tailed p-value: %f" % (rho, p_value)
        print "Num ratings: %d (%d out of vocabulary)" % (rating, oov_count)
    
    net.set_bias(bias)
    
    return rho, p_value, oov_count, probs, baseline


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        sys.exit("Model name input argument missing missing")
    
    if len(sys.argv) > 2:
        experiment_version = sys.argv[2]
    else:
        sys.exit("Experiment input argument missing missing")

    experiment_name = model_name + '_' + experiment_version

    r, p, oov_count, probs, baseline = eval_MNR_LOC(model_name, experiment_name, 'AM-MNR', skip_header=True)
    r2, p2, oov_count2, probs2, baseline2 = eval_MNR_LOC(model_name, experiment_name, 'AM-LOC')

    print "="*60 + "\nTOTAL:"
    rho, p_value = spearmanr(baseline + baseline2, probs + probs2)
    print "Spearman correlation: %f; 2-tailed p-value: %f" % (rho, p_value)
    print "Num ratings: %d (%d out of vocabulary)" % (len(probs + probs2), oov_count + oov_count2)

