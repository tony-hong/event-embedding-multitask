''' This module contains example runner of non-incremental model of role-filler. 

    Author: Tony Hong

    Ref: Ottokar Tilk, Event participant modelling with neural networks, EMNLP 2016
'''

import os, sys, time, cPickle, re

import numpy as np
from keras.optimizers import Adagrad
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, LambdaCallback, TerminateOnNaN, ReduceLROnPlateau

# Configuration of environment
# SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(SRC_DIR, 'evaluation/'))
# sys.path.append(os.path.join(SRC_DIR, 'model/'))

import config, utils, model_builder
from batcher import generator

from model import *
from evaluation import *

DATA_PATH = config.DATA_VERSION
MODEL_PATH = config.MODEL_VERSION



def run(experiment_name, model_name, load_previous, learning_rate, batch_size, samples_per_epoch, epochs, print_after_batches, 
    save_after_steps, learning_rate_decay, L1_reg, L2_reg, n_factors_emb, n_hidden, using_dropout, dropout_rate, loss_weights):
    
    print 'Meta parameters: '
    print('experiment_name: ', experiment_name)
    print('learning_rate: ', learning_rate)
    print('batch_size: ', batch_size)
    print('samples_per_epoch: ', samples_per_epoch)
    print('epochs: ', epochs)
    print('n_factors_emb: ', n_factors_emb)
    print('n_hidden: ', n_hidden)
    print('using_dropout: ', using_dropout)
    print('dropout_rate: ', dropout_rate)
    print('loss_weights: ', loss_weights)
    print ''

    start_time = time.clock()

    experiment_name_prefix = "%s_" % experiment_name
    tmp_exp_name = experiment_name_prefix + "temp"
    final_exp_name = experiment_name_prefix + "final"
    e27_exp_name = experiment_name_prefix + "e27"

    with open(DATA_PATH + "description", 'rb') as data_file:
        description = cPickle.load(data_file)
        print description.keys()
        word_vocabulary = description['word_vocabulary']
        role_vocabulary = description['role_vocabulary']
        unk_word_id = description['NN_unk_word_id']
        unk_role_id = description['unk_role_id']
        missing_word_id = description['NN_missing_word_id']

        print (unk_word_id, unk_role_id, missing_word_id)
    
    print '... building the model'

    rng = np.random
    
    word_vocabulary['<NULL>'] = missing_word_id
    word_vocabulary['<UNKNOWN>'] = unk_word_id
    role_vocabulary['<UNKNOWN>'] = unk_role_id
    n_word_vocab = len(word_vocabulary)
    n_role_vocab = len(role_vocabulary)

    adagrad = Adagrad(lr=learning_rate, epsilon=1e-08, decay=0.0)

    if re.search('NNRF', model_name):
        model = NNRF(n_word_vocab, n_role_vocab, 
            n_factors_emb, 512, n_hidden, word_vocabulary, role_vocabulary, unk_word_id, unk_role_id, missing_word_id, 
            using_dropout, dropout_rate, optimizer=adagrad, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:        
        model = eval(model_name)(n_word_vocab, n_role_vocab, 
            n_factors_emb, n_hidden, word_vocabulary, role_vocabulary, unk_word_id, unk_role_id, missing_word_id, 
            using_dropout, dropout_rate, optimizer=adagrad, loss='sparse_categorical_crossentropy', metrics=['accuracy'], loss_weights=loss_weights)
    # else:
    #     sys.exit('No such model!!!')
        
    model.summary()

    print model.model.metrics_names

    epoch = 0
    max_output_length = 0
    validation_cost_history = []
    best_validation_cost = np.inf
    best_epoch = 0
    
    valid_sample_size = config.OCT_VALID_SIZE
    test_sample_size = config.OCT_TEST_SIZE
    train_steps = samples_per_epoch / batch_size
    valid_steps = valid_sample_size / batch_size
    test_steps = test_sample_size / batch_size
    # # DEBUG
    # valid_steps = 10
    # test_steps = 10

    save_after_steps = save_after_steps + 1

    training_verbose = 2
    max_q_size = 10
    workers = 2
    pickle_safe = True


    def thematic_fit_evaluation(model_name, experiment_name, model, print_result):
        result = dict()
        # Pado, Mcrae A0/A1/A2
        tempdict = dict()
        tempdict['pado'], _, _ = eval_pado_mcrae(model_name, experiment_name, 'pado', model, print_result)
        tempdict['mcrae'], _, _ = eval_pado_mcrae(model_name, experiment_name, 'mcrae', model, print_result)
        # tempdict['pado_fixed'] = eval_pado_mcrae(model_name, experiment_name, 'pado_fixed', model=model, print_result=False)
        # tempdict['mcrad_fixed'] = eval_pado_mcrae(model_name, experiment_name, 'mcrad_fixed', model=model, print_result=False)
        for k, v in tempdict.items():
            for sk, sv in v.items():
                result[k + '-' + sk] = sv

        r, _, _, _, _ = eval_MNR_LOC(model_name, experiment_name, 'AM-MNR', model, print_result, skip_header=True)
        result['mcrae-MNR'] = round(r, 4)
        r2, _, _, _, _ = eval_MNR_LOC(model_name, experiment_name, 'AM-LOC', model, print_result)
        result['mcrae-LOC'] = round(r2, 4)

        rho_obj, _, _, rho_fil, _, _, rho_gre, _, _ = eval_greenberg_all(model_name, experiment_name, model, print_result)
        result['GObject'] = round(rho_obj, 4)
        result['GFiller'] = round(rho_fil, 4)
        result['greenberg'] = round(rho_gre, 4)

        correct, _, acc = eval_bicknell_switch(model_name, experiment_name, 'bicknell', model, print_result, switch_test=False)
        result['bicknell'] = (acc, correct)

        correlation = eval_GS(model_name, experiment_name, 'GS2013data.txt', model, print_result)
        result['GS'] = round(correlation, 4)

        return result


    class CallbackContainer(Callback):
        """Callback that records events into a `History` object.
        """
        def on_train_begin(self, logs=None):
            self.epoch = []
            self.history = {}
            self.best_validation_cost = -1
            self.best_epoch = -1

        def on_epoch_begin(self, epoch, logs):
            self.epoch_start = time.clock()

        def on_batch_end(self, batch, logs):
            batch_n = batch + 1
            epoch_n = len(self.epoch)
            if batch_n % print_after_batches == 0:
                elapsed_time = time.clock() - self.epoch_start
                output = "batch %d; %d samples; %.1f sps; " % (
                    batch_n, 
                    batch_n * batch_size, 
                    batch_n * batch_size / (elapsed_time + 1e-32))
                print output
            if batch_n % save_after_steps == 0:
                model.save(MODEL_PATH, tmp_exp_name, model_name, learning_rate, self.history, self.best_validation_cost, self.best_epoch, epoch_n)
                print "Temp model saved! "

        def on_epoch_end(self, epoch, logs=None):
            epoch_n = epoch + 1
            logs = logs or {}
            self.epoch.append(epoch_n)
            
            print 'Validating...'
            valid_result = model.model.evaluate_generator(
                    generator = generator(DATA_PATH + "NN_dev", model_name, unk_word_id, unk_role_id, missing_word_id, 
                        role_vocabulary, random=False, batch_size=batch_size),
                    steps = test_steps, 
                    max_q_size = 1, 
                    workers = 1, 
                    pickle_safe = False
                )
            print ('validate_result', valid_result)

            for i, m in enumerate(model.model.metrics_names):
                logs['valid_' + m] = valid_result[i]

            # print model.model.get_layer("softmax_word_output").get_weights()[1]

            result = thematic_fit_evaluation(model_name, experiment_name, model, False)
            for k, v in result.items():
                logs[k] = v

            # print model.model.get_layer("softmax_word_output").get_weights()[1]

            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

            if epoch_n > 1 and self.history['valid_loss'][-1] < self.history['valid_loss'][-2]:
                print "Best model saved! "
                self.best_validation_cost = np.min(np.array(self.history['valid_loss']))
                self.best_epoch = np.argmin(np.array(self.history['valid_loss'])) + 1
                model.save(MODEL_PATH, final_exp_name, model_name, learning_rate, self.history, self.best_validation_cost, self.best_epoch, epoch_n)
                print ('best_validation_cost, best_epoch, epoch_n', self.best_validation_cost, self.best_epoch, epoch_n)
                for k, v in self.history.items():
                    print k, v

            if epoch_n ==  27:
                model.save(MODEL_PATH, experiment_name, model_name, learning_rate, self.history, self.best_validation_cost, self.best_epoch, epoch_n)

            print "Current model saved! "
            model.save(MODEL_PATH, experiment_name, model_name, learning_rate, self.history, self.best_validation_cost, self.best_epoch, epoch_n)


    callback_container = CallbackContainer()

    # saves the backup model weights after each epoch if the validation loss decreased
    # backup_checkpointer = ModelCheckpoint(filepath='backup_' + experiment_name + '.hdf5', verbose=1, save_best_only=True)

    stopper = EarlyStopping(monitor='valid_loss', min_delta=1e-3, patience=5, verbose=1)
    naNChecker = TerminateOnNaN()
    reduce_lr = ReduceLROnPlateau(monitor='valid_loss', factor=0.1,
              patience=3, min_lr=0.001)

    print 'Training...'
    train_start = time.clock()

    model.model.fit_generator(
        generator = generator(DATA_PATH + "NN_train", model_name, unk_word_id, unk_role_id, missing_word_id, 
            role_vocabulary, random=True, rng=rng, batch_size=batch_size),
        steps_per_epoch = train_steps, 
        epochs = epochs, 
        verbose = training_verbose,
        workers = workers,
        max_q_size = max_q_size,
        pickle_safe = pickle_safe,
        callbacks = [callback_container, stopper, naNChecker, reduce_lr]
    )
    print callback_container.epoch
    for k, v in callback_container.history.items():
        print k, v
        
    train_end = time.clock()
    print 'train and validate time: %f, sps: %f' % (train_end - train_start, train_steps * batch_size / (train_end - train_start))


    print 'Testing...'
    test_start = time.clock()

    description_best = model_builder.load_description(MODEL_PATH, experiment_name)
    model.load(MODEL_PATH, experiment_name, description_best)

    test_result = model.model.evaluate_generator(
            generator = generator(DATA_PATH + "NN_test", model_name, unk_word_id, unk_role_id, missing_word_id, 
                role_vocabulary, random=False, batch_size=batch_size),
            steps = test_steps, 
            max_q_size = 1, 
            workers = 1, 
            pickle_safe = False
        )
    print ('test_result', test_result)

    test_end = time.clock()
    print 'test time: %f, sps: %f' % (test_end - test_start, test_steps * batch_size / (test_end - test_start))


    end_time = time.clock()
    print "Total running time %.2fh" % ((end_time - start_time) / 3600.)

    print 'Optimization complete. Best validation cost of %f obtained at epoch %i' % (callback_container.best_validation_cost, callback_container.best_epoch)



if __name__ == '__main__':
    '''Check model/__init__.py for the classname of each model
    '''
    learning_rate = config.LEARNING_RATE
    batch_size = config.BATCH_SIZE
    samples_per_epoch = config.SAMPLES_EPOCH
    epochs = config.EPOCHS
    print_after_batches = config.PRINT_AFTER
    save_after_steps = config.SAVE_AFTER
    n_factors_emb = config.FACTOR_NUM
    n_hidden = config.HIDDEN_NUM
    loss_weight_role = config.LOSS_WEIGHT_ROLE

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        sys.exit("Model name input argument missing missing")
    
    if len(sys.argv) > 2:
        experiment_version = sys.argv[2]
    else:
        sys.exit("Experiment name input argument missing missing")
    
    if len(sys.argv) > 3:
        load_previous = bool(int(sys.argv[3]))
    else:
        sys.exit("Load previous (0 = No / 1 = Yes) input argument missing")

    if len(sys.argv) > 4:
        learning_rate = float(sys.argv[4])

    if len(sys.argv) > 5:
        batch_size = int(sys.argv[5])

    if len(sys.argv) > 6:
        samples_per_epoch = int(sys.argv[6])

    if len(sys.argv) > 7:
        epochs = int(sys.argv[7])

    if len(sys.argv) > 8:
        print_after_batches = int(sys.argv[8])

    if len(sys.argv) > 9:
        loss_weight_role = float(sys.argv[9])

    if len(sys.argv) > 10:
        save_after_steps = int(sys.argv[10])

    experiment_name = model_name + '_' + experiment_version

    run(experiment_name, model_name, load_previous, learning_rate, batch_size, samples_per_epoch, epochs, print_after_batches, 
        save_after_steps, learning_rate_decay=1.0, L1_reg=0.00, L2_reg=0.00, n_factors_emb=n_factors_emb, n_hidden=n_hidden, 
        using_dropout=False, dropout_rate=0.2, loss_weights=[1., loss_weight_role])

