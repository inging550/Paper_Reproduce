import os
import numpy as np
import cv2
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt

def search_info(img_info):
    x_info = []
    edge_info = []
    y_info = []
    y_i = 0
    for class_name in os.listdir(img_info):
        print(class_name)
        file_dir = os.path.join(img_info, class_name)
        for file_name in os.listdir(file_dir):
            img = cv2.imread(os.path.join(file_dir, file_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _, _, c = img.shape
            # 划分超像素（每个像素所属的超像素类别）
            segments = slic(img, n_segments=81, sigma=5) - 1
            segments_ids = np.unique(segments)
            centers = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])
            x_i = np.zeros((centers.shape[0], c))
            for i, coordinate in enumerate(centers):
                x_i[i, :] = img[int(coordinate[0]), int(coordinate[1])]

            vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
            vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
            edge = np.unique(np.hstack([vs_right, vs_below]), axis=1)
            x_info.append(x_i)
            edge_info.append(edge)
            y_info.append(y_i)
        y_i += 1
    return x_info, edge_info, y_info

def write_txt(img_info, x_dataset, edge_dataset, y_dataset):
    x, edge, y = search_info(img_info)
    for num in range(len(x)):
        for i in range(x[num].shape[0]):
            x_dataset.write(' '.join([str(j) for j in x[num][i, :]]) + ' ')
        x_dataset.write('\n')
    x_dataset.close()
    for num in range(len(edge)):
        for i in range(2):
            edge_dataset.write(' '.join([str(j) for j in edge[num][i, :]]))
            edge_dataset.write('\n')
    edge_dataset.close()
    for num in range(len(y)):
        y_dataset.write(str(y[num]))
        y_dataset.write('\n')
    y_dataset.close()



if __name__ == '__main__':
    train_path = "F:/CCCCCProject/DATASET/TRAIN/"
    test_path = "F:/CCCCCProject/DATASET/TEST/"

    train_x = open("x_train.txt", 'w')
    train_edge_info = open("edge_train.txt", 'w')
    train_y = open("y_train.txt", 'w')
    test_x = open("x_test.txt", 'w')
    test_edge_info = open("edge_test.txt", 'w')
    test_y = open("y_test.txt", 'w')

    write_txt(test_path, test_x, test_edge_info, test_y)
    write_txt(train_path, train_x, train_edge_info, train_y)
