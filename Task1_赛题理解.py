import pandas as pd
import os
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


## ************** data *************************************
def read_data(file_path):
    data = pd.read_csv(file_path, sep='\t')
    return data

def check_data():
    data_train_path = os.path.join('./data', 'train_set.csv')
    data_train = read_data(data_train_path)
    print(data_train.info())
    """
    RangeIndex: 200000 entries, 0 to 199999
    Data columns (total 2 columns):
    label    200000 non-null int64
    text     200000 non-null object
    dtypes: int64(1), object(1)
    memory usage: 3.1+ MB
    """

    data_test_path = os.path.join('./data', 'test_a.csv')
    data_test = read_data(data_test_path)
    print(data_test.info())
    """
    RangeIndex: 50000 entries, 0 to 49999
    Data columns (total 1 columns):
    text    50000 non-null object
    dtypes: object(1)
    memory usage: 390.8+ KB
    """
## ************ 统计标签的分布 ******************************
def label_distribution():
    data_train_path = os.path.join('./data', 'train_set.csv')
    data_train = read_data(data_train_path)
    print(data_train['label'].value_counts())
    """
    0     38918
    1     36945
    2     31425
    3     22133
    4     15016
    5     12232
    6      9985
    7      8841
    8      7847
    9      5878
    10     4920
    11     3131
    12     1821
    13      908
    """
    data_train['label'].hist()
    plt.show()

def generate_submission_file():
    data_test_path = os.path.join('./data', 'test_a.csv')
    submission_path = os.path.join('./data', 'submission_20200721.csv')
    data_test = read_data(data_test_path)
    data_test['label'] = 0
    print(data_test.shape)  # (50000, 2)
    data_test[['label']].to_csv(submission_path, index=False)


## ************* metric ***********************************
# y_true = [0, 1, 2, 0, 1, 2]
# y_pred = [0, 2, 1, 0, 0, 1]
# f1 = f1_score(y_true, y_pred, average='macro')



if __name__ == '__main__':
    check_data()
    label_distribution()
    generate_submission_file()

