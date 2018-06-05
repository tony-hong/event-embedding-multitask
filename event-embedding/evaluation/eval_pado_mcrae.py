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
COMP_EVAL_PATH = os.path.join(config.EVAL_PATH, 'pado_verify/')

wnl = WordNetLemmatizer()


def eval_pado_mcrae(model_name, experiment_name, evaluation, model=None, print_result=True):
    MODEL_NAME = experiment_name

    if model:
        net = model
    else:
        description = model_builder.load_description(MODEL_PATH, MODEL_NAME)
        net = model_builder.build_model(model_name, description)
        net.load(MODEL_PATH, MODEL_NAME, description)

    bias = net.set_0_bias()

    # net.model.summary()
    # print net.model.get_layer(name="embedding_2").get_weights()[0]

    # If no <UNKNOWN> role in the role vocabulary, add it.
    if net.role_vocabulary.get("<UNKNOWN>", -1) == -1:
        net.role_vocabulary["<UNKNOWN>"] = len(net.role_vocabulary) - 1

    if print_result:
        print net.role_vocabulary
        print("unk_word_id", net.unk_word_id)
        print("missing_word_id", net.missing_word_id)

    propbank_map = {
        "subj"  :   "A0",
        "obj"   :   "A1",
        "ARG0"  :   "A0",
        "ARG1"  :   "A1",
    }

    tr_map = {
        "A0": numpy.asarray([net.role_vocabulary["A0"]], dtype=numpy.int64),
        "A1": numpy.asarray([net.role_vocabulary["A1"]], dtype=numpy.int64),
        "<UNKNOWN>": numpy.asarray([net.role_vocabulary["<UNKNOWN>"]], dtype=numpy.int64)
    }

    if "A2" not in net.role_vocabulary.keys():
        propbank_map["ARG2"] = "<UNKNOWN>"
        tr_map["A2"] = numpy.asarray([net.role_vocabulary["<UNKNOWN>"]], dtype=numpy.int64)
    else:
        propbank_map["ARG2"] = "A2"
        tr_map["A2"] = numpy.asarray([net.role_vocabulary["A2"]], dtype=numpy.int64)

    fixed = False
    if evaluation == "pado":    
        eval_data_file = os.path.join(EVAL_PATH, 'pado_plausibility_pb.txt')
    elif evaluation == 'mcrae':
        eval_data_file = os.path.join(EVAL_PATH, 'mcrae_agent_patient_more.txt')
    else:
        fixed = True
        if evaluation == 'pado_fixed':
            eval_data_file = os.path.join(RV_EVAL_PATH, 'Pado-AsadFixes.txt')
        elif evaluation == 'mcrae_fixed':
            eval_data_file = os.path.join(RV_EVAL_PATH, 'McRaeNN-fixed.txt')
        else:
            eval_data_file = os.path.join(COMP_EVAL_PATH, 'compare-pado.txt')

    result_file = os.path.join(MODEL_PATH, MODEL_NAME + '_' + evaluation + '.txt')

    r_i = net.role_vocabulary["V"]

    probs = {}
    baseline = {}
    oov_count = {}
    blist=[]
    plist = []

    if print_result:
        print eval_data_file
        print "="* 60

    with open(eval_data_file, 'r') as f, \
            open(result_file, 'w') as f_out:
        for i, line in enumerate(f):

            line = line.strip()
            if line == "":
                continue

            w, tw, tr = line.split()[:3]  # input word, target word, role
            w = w[:-2] if fixed else w
            tw = tw[:-2] if fixed else tw

            w = wnl.lemmatize(w, wn.VERB)
            tw = wnl.lemmatize(tw, wn.NOUN)

            w_i = net.word_vocabulary.get(w, net.unk_word_id)
            tw_i = net.word_vocabulary.get(tw, net.unk_word_id)
            tr_i = net.role_vocabulary.get(propbank_map[tr], net.unk_role_id)

            if tw_i == net.unk_word_id:
                print w, tr, tw
                oov_count[tr] = oov_count.get(tr, 0) + 1
                f_out.write(line + "\tnan\n")
                continue

            b = float(line.split()[3])
            baseline.setdefault(tr, []).append(b)
            blist.append(b)

            sample = dict((r, net.missing_word_id) for r in (net.role_vocabulary.values() + [net.unk_role_id]))
            sample[r_i] = w_i

            sample.pop(net.role_vocabulary[propbank_map[tr]], None)

            x_w_i = numpy.asarray([sample.values()], dtype=numpy.int64)
            x_r_i = numpy.asarray([sample.keys()], dtype=numpy.int64)
            y_w_i = numpy.asarray([tw_i])
            y_r_i = tr_map[propbank_map[tr]]

            s = net.p_words(x_w_i, x_r_i, y_w_i, y_r_i)
            # pr = net.p_roles(x_w_i, x_r_i, y_w_i, y_r_i)

            plist.append(s)

            probs.setdefault(tr, []).append(s)

            f_out.write(line + "\t%s\n" % s)

    result = dict()
    for r, b in baseline.iteritems():
        p = probs[r]
        rho, p_value = spearmanr(b, p)
        rating = len(p)
        oov = oov_count.get(r, 0)

        result[r] = round(rho, 4)
        if print_result:
            print "=" * 60
            print "ROLE: %s" % r
            print "-" * 60
            print "Spearman correlation: %f; 2-tailed p-value: %f" % (rho, p_value)
            print "Num ratings: %d (%d out of vocabulary)" % (rating, oov)

    rho, p_value = spearmanr(blist, plist)
    
    result['all'] = round(rho, 4)
    if print_result:
        print "Spearman correlation of %s: %f; 2-tailed p-value: %f" % (evaluation, rho, p_value)


    net.set_bias(bias)
    
    return result, plist, blist


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

    if len(sys.argv) > 3:
        evaluation = sys.argv[3].lower()
        eval_pado_mcrae(model_name, experiment_name, evaluation)
    else:
        eval_pado_mcrae(model_name, experiment_name, 'pado')
        # eval_pado_mcrae(model_name, experiment_name, 'compare-pado')

        eval_pado_mcrae(model_name, experiment_name, 'mcrae')

        # eval_pado_mcrae(model_name, experiment_name, 'pado_fixed')
        # eval_pado_mcrae(model_name, experiment_name, 'mcrae_fixed')    

