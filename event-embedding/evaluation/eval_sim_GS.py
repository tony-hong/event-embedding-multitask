import os
import sys
import cPickle
import re

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from keras.models import Model
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

# Configuration of environment
SRC_DIR = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))
sys.path.append(SRC_DIR)

import model_builder
import config
import utils

MODEL_PATH = config.MODEL_VERSION
EVAL_PATH = os.path.join(config.EVAL_PATH, 'GS/')

wnl = WordNetLemmatizer()


def get_en_uk(word):
    if word == "favour":
        word = "favor"
    if word == "behaviour":
        word = "behavior"
    if word == "offence":
        word = "offense"
    return word


def eval_GS(model_name, experiment_name, eval_file_name, model=None, print_result=True, verb_baseline=False):
    MODEL_NAME = experiment_name
    eval_file = os.path.join(EVAL_PATH, eval_file_name)
    result_file = os.path.join(MODEL_PATH, MODEL_NAME + '_' + eval_file_name)
    
    if model:
        net = model
    else:
        description = model_builder.load_description(MODEL_PATH, MODEL_NAME)
        net = model_builder.build_model(model_name, description)
        net.load(MODEL_PATH, MODEL_NAME, description)

    sent_layer = 'context_embedding'

    sent_model = Model(inputs=net.model.input,
                                 outputs=net.model.get_layer(sent_layer).output)
    
    # if print_result:
    #     sent_model.summary()

    n_input_length = len(net.role_vocabulary) - 1

    print net.role_vocabulary

    scores = []
    similarities = []
    original_sim_f = []
    similarities_f = []
    lo_similarities = []
    hi_similarities = []
    records = []

    print("Embedding: " + experiment_name)
    print("="*60)
    print("\n")
    print("sentence1\tsentence2\taverage_score\tembedding_cosine")
    print("-"*60)

    with open(eval_file, 'r') as f, \
        open(result_file, 'w') as f_out:


        first = True
        for line in f:
            # skip header
            if first:
                first = False
                continue

            s = line.split()
            sentence = " ".join(s[1:5])
            score = float(s[5])
            hilo = s[6].upper()

            # verb subject object landmark
            # A1 - object; A0 - subject
            V1, A0, A1, V2 = sentence.split()

            V1 = wnl.lemmatize(V1, wn.VERB)
            A0 = wnl.lemmatize(A0, wn.NOUN)
            A1 = wnl.lemmatize(A1, wn.NOUN)
            V2 = wnl.lemmatize(V2, wn.VERB)

            V1_i = net.word_vocabulary.get(V1, net.unk_word_id)
            A0_i = net.word_vocabulary.get(A0, net.unk_word_id)
            A1_i = net.word_vocabulary.get(A1, net.unk_word_id)
            V2_i = net.word_vocabulary.get(V2, net.unk_word_id)

            # if np.array([V1_i, A0_i, A1_i, V2_i]).any() == net.unk_word_id:
            #     print 'OOV: ', A0, A1, V1, V2

            V_ri = net.role_vocabulary['V']
            A0_ri = net.role_vocabulary['A0']
            A1_ri = net.role_vocabulary['A1']

            sent1_x = dict((r, net.missing_word_id) for r in (net.role_vocabulary.values()))
            sent2_x = dict((r, net.missing_word_id) for r in (net.role_vocabulary.values()))

            sent1_x.pop(n_input_length)
            sent2_x.pop(n_input_length)

            sent1_x[V_ri] = V1_i
            sent2_x[V_ri] = V2_i

            if not verb_baseline: 
                sent1_x[A0_ri] = A0_i
                sent1_x[A1_ri] = A1_i
                sent2_x[A0_ri] = A0_i
                sent2_x[A1_ri] = A1_i

            zeroA = np.array([0])

            s1_w = np.array(sent1_x.values()).reshape((1, n_input_length))
            s1_r = np.array(sent1_x.keys()).reshape((1, n_input_length))
            s2_w = np.array(sent2_x.values()).reshape((1, n_input_length))
            s2_r = np.array(sent2_x.keys()).reshape((1, n_input_length))

            if re.search('NNRF', model_name):
                sent1_emb = sent_model.predict([s1_w, s1_r, zeroA])
                sent2_emb = sent_model.predict([s2_w, s2_r, zeroA])
            else:
                sent1_emb = sent_model.predict([s1_w, s1_r, zeroA, zeroA])
                sent2_emb = sent_model.predict([s2_w, s2_r, zeroA, zeroA])

            # Baseline
            #sent1_emb = V1_i
            #sent2_emb = V2_i
            # Compositional
            # sent1_emb = V1_i + A0_i + A1_i
            # sent2_emb = V2_i + A0_i + A1_i
            #sent1_emb = V1_i * A0_i * A1_i
            #sent2_emb = V2_i * A0_i * A1_i

            similarity = -(cosine(sent1_emb, sent2_emb) - 1.0) # convert distance to similarity

            if hilo == "HIGH": 
                hi_similarities.append(similarity)
            elif hilo == "LOW":
                lo_similarities.append(similarity)
            else:
                raise Exception("Unknown hilo value %s" % hilo)

            if (V1, A0, A1, V2) not in records: 
                records.append((V1, A0, A1, V2))
                # print "\"%s %s %s\"\t\"%s %s %s\"\t%.2f\t%.2f \n" % (A0, V1, A1, A0, V2, A1, score, similarity)

            scores.append(score)
            similarities.append(similarity)
            
            f_out.write("\"%s %s %s\"\t\"%s %s %s\"\t %.2f \t %.2f \n" % (A0, V1, A1, A0, V2, A1, score, similarity))

    print("-"*60)

    correlation, pvalue = spearmanr(scores, similarities)

    if print_result:
        print "Total number of samples: %d" % len(scores)
        print "Spearman correlation: %.4f; 2-tailed p-value: %.10f" % (correlation, pvalue)
        print "High: %.2f; Low: %.2f" % (np.mean(hi_similarities), np.mean(lo_similarities))

        # import pylab
        # pylab.scatter(scores, similarities)
        # pylab.show()

    return correlation


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

    file2011 = 'GS2011data.txt'
    file2013 = 'GS2013data.txt'

    # eval_GS(model_name, experiment_name, file2011, verb_baseline=True)
    # eval_GS(model_name, experiment_name, file2013, verb_baseline=True)

    # eval_GS(model_name, experiment_name, file2011, verb_baseline=False)
    eval_GS(model_name, experiment_name, file2013, verb_baseline=False)
