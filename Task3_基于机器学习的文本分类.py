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
    1. One-hot
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

    # from sklearn.preprocessing import OneHotEncoder
    #
    # # 语料库里所有单词的集合
    # words_set = set(' '.join(list(data['text'])).split(' '))
    #
    # # 对每个单词编号，得到其索引（这一步也可以用sklearn的LabelEncoder来实现）
    # word2idx = {}
    # idx2word = {}
    # for i, word in enumerate(words_set):
    #     word2idx[word] = i + 1
    #     idx2word[i + 1] = word
    # print(word2idx)
    # """
    # {'6981': 1, '6307': 2, '5367': 3, '1066': 4,...}
    # """
    #
    # # OneHotEncoder输入为shape=(N_words,1)的索引值，输出为各索引值下的one-hot向量word_onehot
    # idx = list(word2idx.values())
    # idx = np.array(idx).reshape(len(idx), -1)
    # print(idx.shape) #(2958, 1)
    # print(idx)
    # """
    # [[   1]
    #  [   2]
    #  [   3]
    #  ...
    #  [2956]
    #  [2957]
    #  [2958]]
    # """
    # onehotenc = OneHotEncoder()
    # onehotenc.fit(idx)
    # word_onehot = onehotenc.transform(idx).toarray()
    # for i, word_onehot_i in enumerate(word_onehot):
    #     print("{0}\t-->\t{1}".format(idx2word[i + 1], word_onehot_i))
    # """
    # 6981	-->	[1. 0. 0. ... 0. 0. 0.]
    # 6307	-->	[0. 1. 0. ... 0. 0. 0.]
    # """
    #
    # # 用法：给定word，找到它的idx，然后从word_onehot里取出对应的one-hot向量
    # x = word_onehot[word2idx['6981']]
    # print(x) #word 6981 的idx 对应的one-hot向量


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
    vectorizer.fit(corpus)
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



