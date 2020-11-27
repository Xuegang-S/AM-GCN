# Copyright (c) 2016 Thomas Kipf
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
import random
from gcn.utils import *
from gcn.models import MLP, GCN, Deep_GCN

def get_train_test_masks(labels, idx_train, idx_val, idx_test):
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    sxg_mask1 = np.ones(labels.shape[0])  
    sxg_mask = np.array(sxg_mask1, dtype=np.bool)
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask, sxg_mask


def run_training(adj, features, labels, idx_train, idx_val, idx_test,
                 params):

    random.seed(params['seed'])
    np.random.seed(params['seed'])
    tf.set_random_seed(params['seed'])

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('model', params['model'], 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', params['lrate'], 'Initial learning rate.')
    flags.DEFINE_integer('epochs', params['epochs'], 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', params['hidden'], 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', params['dropout'], 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', params['decay'], 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', params['early_stopping'], 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', params['max_degree'], 'Maximum Chebyshev polynomial degree.')
    flags.DEFINE_integer('depth', params['depth'], 'Depth of Deep GCN')

    # Create test, val and train masked variables
    y_train, y_val, y_test, train_mask, val_mask, test_mask, sxg_mask = get_train_test_masks(labels, idx_train, idx_val, idx_test)
    features = preprocess_features(features)

    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for GCN model ')
    
    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'phase_train': tf.placeholder_with_default(False, shape=()),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    model1 = model_func(placeholders, input_dim=features[2][1], logging=True)

    # Initialize session
    sess = tf.Session()

    def evaluate(feats, graph, label, mask, placeholder):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(feats, graph, label, mask, placeholder)
        feed_dict_val.update({placeholder['phase_train'].name: False})
        outs_val = sess.run([model1.loss, model1.accuracy, model1.predict()], feed_dict=feed_dict_val)

        pred = outs_val[2]

        pred = pred[np.squeeze(np.argwhere(mask == 1)), :]
        pred1 = np.round(pred)
        lab = label
        lab = lab[np.squeeze(np.argwhere(mask == 1)), :]
        pred11 = np.zeros(len(pred1))
        for ii in range(0, len(pred1)):
            pred11[ii] = pred1[ii][0]
        lab11 = np.zeros(len(lab))
        for ii in range(0, len(lab)):
            lab11[ii] = lab[ii][0]
        return outs_val[0], outs_val[1], (time.time() - t_test), pred, lab  # used to output roc curve (pred and lab)

    # Init variables
    sess.run(tf.global_variables_initializer())
    
    cost_val = []
    # Train model

    for epoch in range(params['epochs']):

        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout, placeholders['phase_train']: True})
        outs = sess.run([model1.opt_op, model1.loss, model1.accuracy, model1.predict(),model1.outputs], feed_dict=feed_dict)

        pred = outs[4]
        pred = pred[np.squeeze(np.argwhere(train_mask == 1)), :]
        labs = y_train
        labs = labs[np.squeeze(np.argwhere(train_mask == 1)), :]

        # Validation
        cost, acc, duration, spe, bac = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)
        print("Epoch:", '%04d' % (epoch + 1))
        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")
    
    # Testing
    sess.run(tf.local_variables_initializer())
    test_cost, test_acc, test_duration, test_spe, test_bac = evaluate(features, support, y_test, sxg_mask, placeholders) 

    del flags
    del FLAGS
    del model1
    del placeholders
    return test_acc, test_spe, test_bac
