# model_builder.py
import re
import os
import cPickle

from model import *

def load_description(file_dir, model_name):
        description_file = os.path.join(file_dir, model_name + "_description")        

        with open(description_file, 'rb') as f:
            description = cPickle.load(f)

        return description

def build_model(model_name, description):
    if re.search('NNRF', model_name):
        net = NNRF(
            n_word_vocab = len(description["word_vocabulary"]),
            n_role_vocab = len(description["role_vocabulary"]),
            n_factors_emb = description["n_factors_emb"],
            n_hidden = description["n_hidden"],
            word_vocabulary = description["word_vocabulary"],
            role_vocabulary = description["role_vocabulary"],
            unk_word_id = description["unk_word_id"],
            unk_role_id = description["unk_role_id"],
            missing_word_id = description["missing_word_id"],
            using_dropout = description["using_dropout"],
            dropout_rate = description["dropout_rate"]
            )
    else:
        net = eval(model_name)(
            n_word_vocab = len(description["word_vocabulary"]),
            n_role_vocab = len(description["role_vocabulary"]),
            n_factors_emb = description["n_factors_emb"],
            n_hidden = description["n_hidden"],
            word_vocabulary = description["word_vocabulary"],
            role_vocabulary = description["role_vocabulary"],
            unk_word_id = description["unk_word_id"],
            unk_role_id = description["unk_role_id"],
            missing_word_id = description["missing_word_id"],
            using_dropout = description["using_dropout"],
            dropout_rate = description["dropout_rate"]
            )

    return net