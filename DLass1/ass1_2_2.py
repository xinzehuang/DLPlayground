import scipy.io as scio
import random
import numpy as np

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

def get_acc(m=10, k=2):
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
    print(len(train_feature))
    knn = KNeighborsClassifier(n_neighbors=k, p=2)
    knn.fit(train_feature, train_label)
    y_predict = knn.predict(test_feature)

    acc = accuracy_score(test_label, y_predict)
    return acc

m_list = [10, 20, 30, 40, 50]
k_list = [2, 3, 5, 10]
err_list = []
for m in m_list:
    err_list_sub = []
    for k in k_list:
        err_list_sub.append(1-get_acc(m, k))
    err_list.append(err_list_sub)

print(err_list)

x = np.asarray(k_list)
y0 = np.asarray(err_list[0])
y1 = np.asarray(err_list[1])
y2 = np.asarray(err_list[2])
y3 = np.asarray(err_list[3])
y4 = np.asarray(err_list[4])

plt.plot(x, y0, '.-', label='m=10')
plt.plot(x, y1, '.-', label='m=20')
plt.plot(x, y2, '.-', label='m=30')
plt.plot(x, y3, '.-', label='m=40')
plt.plot(x, y4, '.-', label='m=50')

plt.xlabel('k')
plt.ylabel('error rate')
plt.legend()
plt.show()

