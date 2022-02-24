import scipy.io as scio
import random
import numpy as np

# from DLass1.utils1 import get_dataset_random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# Import matplotlib library
import matplotlib.pyplot as plt

# Import scikit-learn library
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import numpy as np
# random seed
random.seed(436)

# load data
dataFile = 'Data/YaleB_32x32.mat'
data = scio.loadmat(dataFile)
# print(data['fea'].shape)
# print(data['gnd'].shape)

def get_acc(m=20, k=2, p=1):
    def get_dataset_random(data, m=m):
        features = data['fea']
        labels = data['gnd']
        # print("labels")
        # print(labels)

        remove_duplicates_labels_set = set()
        for label in labels:
            remove_duplicates_labels_set.add(label[0])

        # print(remove_duplicates_labels_set)

        label_featureindex_dict = {}
        for label in remove_duplicates_labels_set:
            label_featureindex_dict[str(label)] = []

        for feature_index, feature in enumerate(features):
            label_featureindex_dict[str(labels[feature_index][0])].append(feature)

        # just for validation
        # for label in remove_duplicates_labels_set:
        #     print(len(label_featureindex_dict[str(label)]))

        train_feature = []
        test_feature = []
        train_label = []
        test_label = []

        for label in remove_duplicates_labels_set:
            temp_list = label_featureindex_dict[str(label)].copy()
            random.shuffle(temp_list)
            for i in temp_list[:m]:
                train_feature.append(i)
            for i in temp_list[m:]:
                test_feature.append(i)
            for i in range(m):
                train_label.append(label)
            for i in range(len(temp_list)-m):
                test_label.append(label)

        return train_feature, test_feature, train_label, test_label

    # generate train val test
    train_feature, test_feature, train_label, test_label = get_dataset_random(data=data, m=m)


    ############################## PCA #################################################################

    n_components = 150

    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(train_feature)

    eigenfaces = pca.components_.reshape((n_components, 32, 32))

    train_feature_pca = pca.transform(train_feature)
    test_feature_pca = pca.transform(test_feature)


    # knn

    knn = KNeighborsClassifier(n_neighbors=1, p=2)
    knn.fit(train_feature_pca, train_label)
    y_predict = knn.predict(test_feature_pca)

    acc = accuracy_score(test_label, y_predict)


    return acc


if __name__ == '__main__':

    print(get_acc())




