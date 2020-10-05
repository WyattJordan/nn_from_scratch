import numpy as np
import gzip
import pickle

def load_data():
    f = gzip.open('../data/mnist.pkl.gz','rb')
    train, valid, test = pickle.load(f,encoding="bytes")
    # x is (784, n) and y is (10, n) with n=50,000 for training and
    # 10,000 for both validation and test data
    train_x, train_y = format_dataset(train)
    valid_x, valid_y = format_dataset(valid)
    test_x,  test_y  = format_dataset(test )
    
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def format_dataset(data):
    data_x = np.array(data[0],dtype=np.double).T
    data_y = np.zeros([10,len(data[1])])
    for i,y in enumerate(data[1]):
        data_y[y][i] = 1.0
    return data_x, data_y
