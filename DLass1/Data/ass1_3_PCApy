import scipy.io as scio
import random
import numpy as np

# from DLass1.utils1 import get_dataset_random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# random seed
random.seed(436)

# load data
dataFile = 'Data/YaleB_32x32.mat'
data = scio.loadmat(dataFile)
# print(data['fea'].shape)
# print(data['gnd'].shape)

def get_acc(m=10, k=2, p=1):
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
            for i in temp_list[:-20]:
                train_feature.append(i)
            for i in temp_list[-20:]:
                test_feature.append(i)
            for i in range(len(temp_list)-20):
                train_label.append(label)
            for i in range(20):
                test_label.append(label)

        return train_feature, test_feature, train_label, test_label

    # generate train val test
    train_feature, test_feature, train_label, test_label = get_dataset_random(data=data, m=m)

    from sklearn.model_selection import GridSearchCV

    parameters = {'n_neighbors': [2, 3, 5, 10],
                  'p': [1, 2, 3]}
    knn = KNeighborsClassifier(n_neighbors=k, p=p)
    clf = GridSearchCV(knn, parameters, scoring="accuracy")
    clf.fit(train_feature, train_label)
    print(clf.best_params_)
    print(clf.best_score_)

    return 1


if __name__ == '__main__':

    get_acc()




