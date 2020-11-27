

from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE


def feature_selection(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    estimator = RidgeClassifier()
    selector = RFE(estimator, fnum, step=5, verbose=0)

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)

    print("Number of labeled samples %d" % len(train_ind))
    print("Number of features selected %d" % x_data.shape[1])

    return x_data

def final_graph_sparse(final_graph, feat_num):
    for i in range(len(final_graph[1])):
        ai = sorted(final_graph[i, :])  # 第i行从小到大排序，
        bi = ai[len(final_graph[1]) - feat_num]  # 去掉最小相似值，保留feat_num个大的相似值
        ci = ai[len(final_graph[1]) - 2]  # 去掉2个最大相似值
        for j in range(len(final_graph[1])):
            if final_graph[i, j] <= bi:
                final_graph[i, j] = 0

    return final_graph
