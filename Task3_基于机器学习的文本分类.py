import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score


def read_data(file_path):
    data = pd.read_csv(file_path, sep='\t')
    # data = pd.read_csv(file_path, sep='\t', nrows=100)
    return data

def text_classification_tradition():
    train_df = read_data('./data/train_set.csv')
    test_df = read_data('./data/test_a.csv')
    data = pd.concat([train_df,test_df],axis=0)
    print(data.shape)

    """
    传统的文本表示方法:
    1. One-hot：很少应用，因为词太多了
    2. BOW（Bag of Words，词袋表示）
    3. N-gram
    4. TF-IDF
    
    使用sklearn feature_extraction.text里的文本表示接口时，输入格式为：
    corpus = [
    ...     'This is the first document.',
    ...     'This document is the second document.',
    ...     'And this is the third one.',
    ...     'Is this the first document?',
    ... ]
    详见：https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text
    """
    # 1. one-hot: https://zhuanlan.zhihu.com/p/67856266
    # 2. BOW: CountVectorizer
    corpus = data['text'].values
    vectorizer = CountVectorizer(max_features=3000)
    vectorizer.fit(corpus) #用训练集和测试集的所有语料训练特征提取器
    X_train_all = vectorizer.transform(train_df['text'].values)
    y_train_all = train_df['label'].values
    X_test = vectorizer.transform(test_df['text'].values)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_all, y_train_all, test_size = 0.1, random_state = 2020)
    clf = RidgeClassifier()
    clf.fit(X_train, y_train)
    y_valid_pred = clf.predict(X_valid)
    print("f1 score: %.6f" % f1_score(y_valid, y_valid_pred, average='macro'))
    """
    f1 score: 0.820636
    """

    y_test_pred = clf.predict(X_test)
    test_df['label'] = y_test_pred
    print(test_df.shape)  # (50000, 2)
    test_df[['label']].to_csv('./data/submission_bow_20200725.csv', index=False)
    print(test_df['label'].value_counts())
    """
    1     11305
    0     10942
    2      8012
    3      5798
    4      3311
    5      2740
    6      1975
    7      1563
    9      1134
    8      1128
    10     1085
    11      548
    12      322
    13      137
    Name: label, dtype: int64
    """

    # 3. N-gram: CountVectorizer(ngram_range=(1,N))

    # 4. TF-IDF: TfidfVectorizer
    corpus = data['text'].values
    vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
    vectorizer.fit(corpus)  # 用训练集和测试集的所有语料训练特征提取器
    X_train_all = vectorizer.transform(train_df['text'].values)
    y_train_all = train_df['label'].values
    X_test = vectorizer.transform(test_df['text'].values)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_all, y_train_all, test_size=0.1, random_state=2020)
    clf = RidgeClassifier()
    clf.fit(X_train, y_train)
    y_valid_pred = clf.predict(X_valid)
    print("f1 score: %.6f" % f1_score(y_valid, y_valid_pred, average='macro'))
    """
    f1 score: 0.897664
    """

    y_test_pred = clf.predict(X_test)
    test_df['label'] = y_test_pred
    print(test_df.shape)  #
    test_df[['label']].to_csv('./data/submission_tfidf_20200725.csv', index=False)
    print(test_df['label'].value_counts())
    """
    0     9805
    1     9558
    2     7990
    3     5753
    4     3787
    5     3064
    6     2355
    7     2016
    8     1816
    9     1378
    10    1158
    11     712
    12     405
    13     203
    Name: label, dtype: int64
    """


if __name__ == '__main__':
    text_classification_tradition()



