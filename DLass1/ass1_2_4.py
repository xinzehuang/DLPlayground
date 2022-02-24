import scipy.io as scio
import random
import numpy as np

# from DLass1.utils1 import get_dataset_random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.feature import hog

# random seed
random.seed(436)

# load data
dataFile = 'Data/YaleB_32x32.mat'
data = scio.loadmat(dataFile)
# print(data['fea'].shape)
# print(data['gnd'].shape)

def get_acc(m=10, k=2, p=1, LBP=True):
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

    print("length of train feature{}".format(len(train_feature)))
    if LBP:
        ## LBP
        ####################################################################################################################################################

        def get_pixel(img, center, x, y):
            new_value = 0

            try:
                # If local neighbourhood pixel
                # value is greater than or equal
                # to center pixel values then
                # set it to 1
                if img[x][y] >= center:
                    new_value = 1

            except:
                # Exception is required when
                # neighbourhood value of a center
                # pixel value is null i.e. values
                # present at boundaries.
                pass

            return new_value

        # Function for calculating LBP
        def lbp_calculated_pixel(img, x, y):
            center = img[x][y]

            val_ar = []

            # top_left
            val_ar.append(get_pixel(img, center, x - 1, y - 1))

            # top
            val_ar.append(get_pixel(img, center, x - 1, y))

            # top_right
            val_ar.append(get_pixel(img, center, x - 1, y + 1))

            # right
            val_ar.append(get_pixel(img, center, x, y + 1))

            # bottom_right
            val_ar.append(get_pixel(img, center, x + 1, y + 1))

            # bottom
            val_ar.append(get_pixel(img, center, x + 1, y))

            # bottom_left
            val_ar.append(get_pixel(img, center, x + 1, y - 1))

            # left
            val_ar.append(get_pixel(img, center, x, y - 1))

            # Now, we need to convert binary
            # values to decimal
            power_val = [1, 2, 4, 8, 16, 32, 64, 128]

            val = 0

            for i in range(len(val_ar)):
                val += val_ar[i] * power_val[i]

            return val

        ############################################################################################

        img_gray = train_feature[3].reshape((32, 32))

        height, width = 32, 32

        img_lbp = np.zeros((height, width),
                           np.uint8)

        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

        plt.imshow(img_gray.T, cmap="gray")
        plt.show()

        plt.imshow(img_lbp.T, cmap="gray")
        plt.show()

        #########################################################################
        # construct LBP features
        train_feature_LBP = []
        test_feature_LBP = []

        for fea in train_feature:
            img_gray = fea.reshape((-1, 32))
            height, width = 32, 32
            img_lbp = np.zeros((height, width),
                               np.uint8)
            for i in range(0, height):
                for j in range(0, width):
                    img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
            train_feature_LBP.append(img_lbp.flatten())
        for fea in test_feature:
            img_gray = fea.reshape((-1, 32))
            height, width = 32, 32
            img_lbp = np.zeros((height, width),
                               np.uint8)
            for i in range(0, height):
                for j in range(0, width):
                    img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
            test_feature_LBP.append(img_lbp.flatten())

        ###
        knn = KNeighborsClassifier(n_neighbors=k, p=p)
        knn.fit(train_feature_LBP, train_label)
        y_predict = knn.predict(test_feature_LBP)

        acc = accuracy_score(test_label, y_predict)
        return acc

    else:
        #########################################################################
        ## show of HOG
        img_hog = train_feature[4].reshape((32, 32))

        fd, hog_image = hog(img_hog, orientations=9, pixels_per_cell=(2, 2),
                            cells_per_block=(2, 2), visualize=True, multichannel=False, channel_axis=None)

        plt.imshow(img_hog.T, cmap="gray")
        plt.show()
        plt.axis("on")
        plt.imshow(hog_image.T, cmap="gray")
        plt.show()
        #########################################################################
        ## Construct HOG descriptor
        train_feature_HOG = []
        test_feature_HOG = []

        for fea in train_feature:
            img_hog = fea.reshape((32, 32))
            fd, hog_image = hog(img_hog, orientations=9, pixels_per_cell=(2, 2),
                                cells_per_block=(2, 2), visualize=True, multichannel=False, channel_axis=None)

            train_feature_HOG.append(fd.flatten())
        for fea in test_feature:
            img_hog = fea.reshape((32, 32))
            fd, hog_image = hog(img_hog, orientations=9, pixels_per_cell=(2, 2),
                                cells_per_block=(2, 2), visualize=True, multichannel=False, channel_axis=None)

            test_feature_HOG.append(fd.flatten())


        knn = KNeighborsClassifier(n_neighbors=k, p=p)
        knn.fit(train_feature_HOG, train_label)
        y_predict = knn.predict(test_feature_HOG)

        acc = accuracy_score(test_label, y_predict)
        return acc

def get_dataset_random(data, m=10):
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
k = 3
m = 30

p_list = [1, 2]

# LBP
err_list_LBP = []
for p in p_list:
    err_list_LBP.append(1-get_acc(m, k, p, LBP=True))
print(err_list_LBP)
x = np.asarray(p_list)
y_LBP = np.asarray(err_list_LBP)

# HOG
err_list_HOG = []
for p in p_list:
    err_list_HOG.append(1-get_acc(m, k, p, LBP=False))
print(err_list_HOG)
y_HOG = np.asarray(err_list_HOG)

x1 = np.arange(len(x))
# plot
fig, ax = plt.subplots(figsize=(10, 7))
ax.bar(x1, y_LBP, width=0.25, label="LBP", color="red")
ax.bar(x1 + 0.25, y_HOG, width=0.25, label="HOG", color="blue")

ax.set_title("Grouped Bar plot", fontsize=15)
ax.set_xlabel("p")
ax.set_ylabel("Error Rate")
ax.legend()

ax.set_xticks(x1+0.125)
ax.set_xticklabels(x)
plt.show()