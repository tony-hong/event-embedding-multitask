
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


word_fix = {
    'omelet' : 'omelette'
}

def eval_greenberg(model_name, experiment_name, evaluation, model=None, print_result=True):
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

    eval_data_file = os.path.join(RV_EVAL_PATH, evaluation + '.txt')

    result_file = os.path.join(MODEL_PATH, MODEL_NAME + '_' + evaluation + '.txt')

    probs = []
    baseline = []
    oov_count = 0

    tr = "A1"
    r_i = net.role_vocabulary["V"]
    tr_i = net.role_vocabulary["A1"]
    y_r_i = numpy.asarray([tr_i], dtype=numpy.int64)

    if print_result:
        print eval_data_file
        print "="*60

    with open(eval_data_file, 'r') as f, \
         open(result_file, 'w') as f_out:
        for line in f:
            line = line.strip().lower()
            w, tw = line.split()[:2] # input word, target word, other stuff
            w = w[:-2].strip()
            tw = tw[:-2].strip()

            w = wnl.lemmatize(w, wn.VERB)
            tw = wnl.lemmatize(tw, wn.NOUN)

            # a hack to fix some words
            # tw = word_fix.get(tw, tw)

            w_i = net.word_vocabulary.get(w, net.unk_word_id)
            tw_i = net.word_vocabulary.get(tw, net.unk_word_id)

            if tw_i == net.unk_word_id:
                oov_count += 1
                print w, tr, tw
                f_out.write(line + "\tnan\n")
                continue
            
            b = float(line.split()[-1])

            sample = dict((r, net.missing_word_id) for r in (net.role_vocabulary.values() + [net.unk_role_id]))
            sample[r_i] = w_i
            sample.pop(tr_i, None)

            x_w_i = numpy.asarray([sample.values()], dtype=numpy.int64)
            x_r_i = numpy.asarray([sample.keys()], dtype=numpy.int64)
            y_w_i = numpy.asarray([tw_i], dtype=numpy.int64)

            p = net.p_words(x_w_i, x_r_i, y_w_i, y_r_i)
            # pr = net.p_roles(x_w_i, x_r_i, y_w_i, y_r_i)
           
            if tw_i == net.unk_word_id:
                print('OOV: %s' % w, b, p)
            baseline.append(b)
            probs.append(p)            

            f_out.write(line + "\t%s\n" % p)
    
    rho, p_value = spearmanr(baseline, probs)
    if print_result:
        print "Spearman correlation of %s: %f; 2-tailed p-value: %f" % (evaluation, rho, p_value)
        print "Num ratings: %d (%d out of vocabulary)" % (len(probs), oov_count)
    
    net.set_bias(bias)
    
    return rho, p_value, oov_count, probs, baseline


def eval_greenberg_all(model_name, experiment_name, model=None, print_result=True):
    r, p, oov_count, probs, baseline = eval_greenberg(model_name, experiment_name, 'GreenbergEtAl2015-fillers', model, print_result)
    r2, p2, oov_count2, probs2, baseline2 = eval_greenberg(model_name, experiment_name, 'GreenbergEtAl2015', model, print_result)

    rho, p_value = spearmanr(baseline + baseline2, probs + probs2)
    rating = len(probs + probs2)
    oov = oov_count + oov_count2

    if print_result:
        print "="*60 + "\nTOTAL:"
        print "Spearman correlation: %f; 2-tailed p-value: %f" % (rho, p_value)
        print "Num ratings: %d (%d out of vocabulary)" % (rating, oov)

    return (r2, p2, oov_count2, r, p, oov_count, rho, p_value, oov)


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
    
    eval_greenberg_all(model_name, experiment_name)

