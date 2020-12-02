# code for the paper "Augmented Multi-center Graph Convolutional Network for COVID-19 Diagnosis"
# Time : 2020-11-27
# Author : Xuegang Song, shenzhen university, china
# Software: PyCharm
# Our code bases on the public code https://github.com/parisots/population-gcn

import time
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import sparse
from scipy.spatial import distance
import scipy.io as sio
import Code_pub.Parser as Reader2
import Code_pub.Train_GCN1 as Train1
import Code_pub.Train_GCN2 as Train11


def train_fold(train_ind, test_ind, feature, y, y_data, train_ind1, test_ind1, feature1, y1, y_data1, params,graph_feat0, graph_feat1, mode):

    train_ind, val_ind = train_test_split(train_ind, test_size=0.2, random_state=1)

    x_data = Reader2.feature_selection(feature, y, train_ind, params['num_features'])
    x_data1 = Reader2.feature_selection(feature1, y1, train_ind, params['num_features'])

    num_0 = len(train_ind)+len(val_ind)
    num_1 = len(y1)
    distv = distance.pdist(x_data, metric='correlation')
    dist0 = distance.squareform(distv)
    sigma = np.mean(dist0)
    sparse_graph = np.exp(- dist0 ** 2 / (2 * sigma ** 2))
    if mode == 1:
        final_graph = np.eye(len(y), dtype=int)
    if mode == 2:
        final_graph = sparse_graph
    if mode == 3:
        final_graph =  graph_feat0 * sparse_graph
    if mode == 4:
        final_graph = graph_feat0 * sparse_graph

    feat_num = 50
    final_graph = Reader2.final_graph_sparse(final_graph, feat_num)
    for k in range(len(y)):
        final_graph[k, :] = final_graph[k, :] / np.sum(final_graph[k, :])

    ################# Adaptive Adjancency Matrix #########

    test_acc1, test_pred1, test_bac1 = Train1.run_training(final_graph, sparse.coo_matrix(x_data).tolil(), y_data,
                                                          train_ind, val_ind,
                                                          test_ind, params)
    score = test_pred1[:, 1]
    xiangsi = np.random.normal(size=(len(y), len(y)))

    for i1 in [len(y) - 1]:
        for j1 in [len(y) - 1]:
            xiangsi[i1, j1] = score[i1] - score[j1]
    xiangsi = 1 - np.abs(xiangsi)
    dist = xiangsi
    sigma = np.mean(dist)
    sparse_graph0 = np.exp(- dist ** 2 / (2 * sigma ** 2))
    final_graph0 = graph_feat0 * sparse_graph0
    for k in range(len(y)):
        final_graph0[k, :] = final_graph0[k, :] / np.sum(final_graph0[k, :])

    ########### Augmentated Adjancency Matrix  ########

    final_graph1 = np.zeros([num_1, num_1])
    for i in range(num_0):
        for j in range(num_0):
            final_graph1[i, i] = 1
    for i in range(num_0,2*num_0):
        for j in range(num_0,2*num_0):
            final_graph1[i, j] = final_graph[i-num_0, j-num_0]
    for i in range(2*num_0,num_1):
        for j in range(2*num_0,num_1):
            final_graph1[i, j] = final_graph0[i-2*num_0, j-2*num_0]

    final_graph1 = graph_feat1 * final_graph1
    final_graph1 = Reader2.final_graph_sparse(final_graph1, feat_num)
    for k in range(len(y1)):
        final_graph1[k, :] = final_graph1[k, :] / np.sum(final_graph1[k, :])

    ###########  Augmented Multi-center GCN   ###########
    train_ind1, val_ind1 = train_test_split(train_ind1, test_size=0.2, random_state=1)
    test_acc1, test_pred1, test_bac1 = Train11.run_training(final_graph1, sparse.coo_matrix(x_data1).tolil(), y_data1,
                                                          train_ind1, val_ind1,
                                                          test_ind1, params)
    time.sleep(2)
    lab = y_data1[test_ind1, 1]
    pred = test_pred1[test_ind1, 1]

    return pred, lab


def main():
    parser = argparse.ArgumentParser(description='Graph CNNs for population graphs: '
                                                 'classification of the ADNI dataset')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout rate (1 - keep probability) (default: 0.3)')
    parser.add_argument('--decay', default=0.0005, type=float,
                        help='Weight for L2 loss on embedding matrix (default: 5e-4)')
    parser.add_argument('--hidden', default= 32, type=int, help='Number of filters in hidden layers (default: 16)')
    parser.add_argument('--lrate', default=0.005, type=float, help='Initial learning rate (default: 0.005)')
    # parser.add_argument('--atlas', default='ho', help='atlas for network construction (node definition) (default: ho, '
    #                                                   'see preprocessed-connectomes-project.org/abide/Pipelines.html '
    #                                                   'for more options )')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train')
    parser.add_argument('--num_features', default=20, type=int, help='Number of features to keep for '
                                                                       'the feature selection step (default: 2000)')
    parser.add_argument('--num_training', default=1, type=float, help='Percentage of training set used for '
                                                                        'training (default: 1.0)')
    parser.add_argument('--depth', default=0, type=int, help='Number of additional hidden layers in the GCN. '
                                                             'Total number of hidden layers: 1+depth (default: 0)')
    parser.add_argument('--model', default='gcn_cheby', help='gcn model used (default: gcn_cheby, '
                                                             'uses chebyshev polynomials, '
                                                             'options: gcn, gcn_cheby, dense )')
    parser.add_argument('--seed', default=123, type=int, help='Seed for random initialisation (default: 123)')
    parser.add_argument('--folds', default=3, type=int, help='For cross validation, specifies which fold will be '
                                                             'used. All folds are used if set to 11 (default: 11)')
    parser.add_argument('--save', default=1, type=int, help='Parameter that specifies if results have to be saved. '
                                                            'Results will be saved if set to 1 (default: 1)')
    parser.add_argument('--connectivity', default='correlation', help='Type of connectivity used for network '
                                                                      'construction (default: correlation, '
                                                                      'options: correlation, partial correlation, '
                                                                      'tangent)')

    args = parser.parse_args()
    start_time = time.time()

    # GCN Parameters
    params = dict()
    params['model'] = args.model                    # gcn model using chebyshev polynomials
    params['lrate'] = args.lrate                    # Initial learning rate
    params['epochs'] = args.epochs                  # Number of epochs to train
    params['dropout'] = args.dropout                # Dropout rate (1 - keep probability)
    params['hidden'] = args.hidden                  # Number of units in hidden layers
    params['decay'] = args.decay                    # Weight for L2 loss on embedding matrix.
    params['early_stopping'] = params['epochs']     # Tolerance for early stopping (# of epochs). No early stopping if set to param.epochs
    params['max_degree'] = 3                        # Maximum Chebyshev polynomial degree.
    params['depth'] = args.depth                    # number of additional hidden layers in the GCN. Total number of hidden layers: 1+depth
    params['seed'] = args.seed                      # seed for random initialisation

    # GCN Parameters
    params['num_features'] = args.num_features      # number of features for feature selection step
    params['num_training'] = args.num_training      # percentage of training set used for training

  ############################################################################

    # data1 = sio.loadmat("D:\\code-new\\population-gcn-master\\coruvorus\\corodatasci\\KT1-417-1.mat")
    # task = 1
    # data1 = sio.loadmat("D:\\code-new\\population-gcn-master\\coruvorus\\corodatasci\\KT2-178-1.mat")
    # task = 2
    # data1 = sio.loadmat("D:\\code-new\\population-gcn-master\\coruvorus\\corodatasci\\哈医大-1.mat")
    # task = 3
    # data1 = sio.loadmat("D:\\code-new\\population-gcn-master\\coruvorus\\corodatasci\\七医院-104-11.mat")
    # task = 4
    data1 = sio.loadmat("D:\\code-new\\population-gcn-master\\coruvorus\\corodatasci\\武汉方舱-130-1.mat")
    task = 5
    # data1 = sio.loadmat("D:\\code-new\\population-gcn-master\\coruvorus\\corodatasci\\中南-205-1.mat")
    # task = 6
    # data1 = sio.loadmat("C:\\Users\\admin\\Desktop\\新冠文章修改\\resultrevised\\data-onine1000_fold1weight111-3.mat")
    # task = 7

    feature_train = data1['x_train_1']
    labels_train = data1['y_train1']
    feature_test = data1['x_test_1']
    labels_test = data1['y_test1']
    sex = data1['sex']
    site = data1['site']
    equipnum = data1['equip']

   ############

    feature = np.vstack((feature_train, feature_test))
    labels = np.vstack((labels_train, labels_test))
    total_train = len(labels_train[:, 1])
    total_test = len(labels_test[:, 1])
    y_data = labels
    y = labels[:, 1]
    total_num = total_train + total_test
    graph_feat0 = np.zeros([total_num, total_num])
    for i in range(total_train):
        for j in range(total_train):
            if labels[i, 1] == labels[j, 1] and sex[i] == sex[j] and site[i] == site[j] and equipnum[i] == equipnum[j]:
                graph_feat0[i, j] = 1
    for i in range(total_train, total_num):
        for j in range(total_train, total_num):
            if site[i] == site[j]:   # if the number of test samples is small, it is better to use this.
            # if sex[i] == sex[j] and site[i] == site[j] and equipnum[i] == equipnum[j]:
                graph_feat0[i, j] = 1
    train_ind = list(np.zeros(total_train))
    test_ind = list(np.zeros(total_test))
    for i in range(total_train):
        train_ind[i] = i
    for i in range(total_test):
        test_ind[i] = total_train + i

   ####  augmentation mechanism ######

    feature1 = np.vstack((feature_train, feature_train, feature_train, feature_test))
    labels1 = np.vstack((labels_train, labels_train,labels_train, labels_test))
    y_data1 = labels1
    y1 = labels1[:,1]
    total_num1 = 3*total_train + total_test
    graph_feat1 = np.zeros([total_num1, total_num1])

    for i in range(total_train):
        for j in range(total_train):
            if labels1[i, 1] == labels1[j, 1] and sex[i] == sex[j] and site[i] == site[j] and equipnum[i] == equipnum[j]:
                graph_feat1[i, j] = 1
    for i in range(total_train,2*total_train ):
        for j in range(total_train, 2*total_train):
            if labels1[i, 1] == labels1[j, 1] and sex[i-total_train] == sex[j-total_train] and site[i-total_train] == site[j-total_train] and equipnum[i-total_train] == equipnum[j-total_train]:
                graph_feat1[i, j] = 1
    for i in range(2*total_train,3*total_train):
        for j in range(2*total_train, 3*total_train):
            if labels1[i, 1] == labels1[j, 1] and sex[i-2*total_train] == sex[j-2*total_train] and site[i-2*total_train] == site[j-2*total_train] and equipnum[i-2*total_train] == equipnum[j-2*total_train]:
                graph_feat1[i, j] = 1
    for i in range(3*total_train,total_num1):
        for j in range(3*total_train,total_num1):
            if site[i-3*total_train] == site[j-3*total_train]: # if the number of test samples is small, it is better to use this.
            # if sex[i-3*total_train] == sex[j-3*total_train] and site[i-3*total_train] == site[j-3*total_train] and equipnum[i-3*total_train] == equipnum[j-3*total_train]:
                graph_feat1[i, j] = 1

    train_ind1 = list(np.zeros(3*total_train))
    test_ind1 = list(np.zeros(total_test))
    for i in range(3*total_train):
        train_ind1[i] = i
    for i in range(total_test):
        test_ind1[i] = 3*total_train+i

    ####### Training and Test
    mode = 4
    gcn_pred, lab= train_fold(train_ind, test_ind, feature, y, y_data, train_ind1, test_ind1, feature1, y1, y_data1, params,graph_feat0, graph_feat1, mode)

    sio.savemat("C:\\Users\\admin\\Desktop\\resultrevised\\" + str(task) + 'pred' +'mode'+ str(mode) + '.mat',
                {'pred': gcn_pred})
    sio.savemat("C:\\Users\\admin\\Desktop\\resultrevised\\" + str(task) +'lab' + '.mat',
                {'lab':lab})

if __name__ == "__main__":
    main()
