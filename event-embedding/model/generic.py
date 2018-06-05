''' This module contains generic model of role-filler.

    Author: Tony Hong
'''
import os
import re
import cPickle

from keras.models import load_model

import config
from utils import get_reverse_map


class GenericModel(object):
    """ Generic model for role filler
    """
    def __init__(self, n_word_vocab, n_role_vocab, n_factors_emb, n_hidden, word_vocabulary, role_vocabulary, 
        unk_word_id, unk_role_id, missing_word_id, using_dropout, dropout_rate, optimizer, loss, metrics):

        self.n_word_vocab = n_word_vocab
        self.n_role_vocab = n_role_vocab
        self.n_factors_emb = n_factors_emb
        self.n_hidden = n_hidden

        self.word_vocabulary = word_vocabulary
        self.role_vocabulary = role_vocabulary
        self.word_decoder = get_reverse_map(word_vocabulary)
        self.role_decoder = get_reverse_map(role_vocabulary)

        self.unk_role_id = unk_role_id
        self.unk_word_id = unk_word_id
        self.missing_word_id = missing_word_id
        self.using_dropout = using_dropout
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.loss = loss

        # assert missing_word_id == len(word_vocabulary)
        # assert unk_word_id == len(word_vocabulary) + 1
        # assert unk_role_id == len(role_vocabulary)
        
    def save(self, file_dir, file_name, model_name, learning_rate=None, validation_cost_history=None, 
        best_validation_cost=None, best_epoch=None, epoch=None):
        ''' Save current model
        '''
        description_file = os.path.join(file_dir, file_name + "_description")
        model_file = os.path.join(file_dir, file_name + '.h5')

        description = {
            "n_word_vocab":             self.n_word_vocab,
            "n_role_vocab":             self.n_role_vocab,
            "n_factors_emb":            self.n_factors_emb,
            "n_hidden":                 self.n_hidden,
            "word_vocabulary":          self.word_vocabulary,
            "role_vocabulary":          self.role_vocabulary,
            "unk_role_id":              self.unk_role_id,
            "unk_word_id":              self.unk_word_id,
            "missing_word_id":          self.missing_word_id,
            "using_dropout":            self.using_dropout,
            "dropout_rate":             self.dropout_rate,

            "learning_rate":            learning_rate,
            "validation_cost_history":  validation_cost_history,
            "best_validation_cost":     best_validation_cost,
            "best_epoch":               best_epoch,
            "epoch":                    epoch,
        }
        with open(description_file, 'wb') as f:
            cPickle.dump(description, f, protocol=cPickle.HIGHEST_PROTOCOL)

        if re.search('final', file_name) or re.search('A2', file_name):
            self.model.save_weights(model_file)
        else:
            self.model.save(model_file)


    def load(self, file_dir, file_name, description):
        """ Load model from description
        """
        model_file = os.path.join(file_dir, file_name + '.h5')

        learning_rate = description["learning_rate"]
        validation_cost_history = description["validation_cost_history"]
        best_validation_cost = description["best_validation_cost"]
        best_epoch = description["best_epoch"]
        epoch = description["epoch"]

        if config.SRC:
            print ("Load model directly for: ", file_name)
            self.model = load_model(model_file)
        else:
            print ("Load weights only for: ", file_name)
            self.model.load_weights(model_file)

        return learning_rate, validation_cost_history, best_validation_cost, best_epoch, epoch
        
