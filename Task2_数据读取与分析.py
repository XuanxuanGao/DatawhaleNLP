import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

## ************** data *************************************
def read_data(file_path):
    data = pd.read_csv(file_path, sep='\t')
    # data = pd.read_csv(file_path, sep='\t', nrows=100) # 只读取前nrows行，适合小数据量测试的时候用
    return data

## ************** count ************************************
def count_feature():
    train_df = read_data('./data/train_set.csv')
    print(train_df.head())

    ## 1.句子长度
    train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(" ")))
    print(train_df['text_len'].describe())
    """
    count    200000.000000
    mean        907.207110
    std         996.029036
    min           2.000000
    25%         374.000000
    50%         676.000000
    75%        1131.000000
    max       57921.000000
    Name: text_len, dtype: float64
    """
    train_df['text_len'].hist(bins=200) # hist图，也是频数，与bar图不同的是，hist图相邻两个变量所在的桶得挨着，适合个数较多的离散变量的频数统计
    plt.xlabel("Text char count")
    plt.title("Histogram of char count")
    plt.show()

    ## 2.字符出现次数
    corpus = ' '.join(list(train_df['text'])) #一整个string
    word_cnt = Counter(corpus.split(' ')) # Counter接受四种输入，详见https://blog.csdn.net/u014755493/article/details/69812244
    # 采用Counter对象自带的方法，找出现次数最多、最少的字符
    print(word_cnt.most_common(5)) # 最多
    print(word_cnt.most_common()[:-5-1:-1]) # 最少
    # 也可以把Counter对象当作dict来用，找出现次数最多、最少的字符
    word_cnt_large2small = sorted(word_cnt.items(), key=lambda d: d[1], reverse=True)
    print(word_cnt_large2small[0],word_cnt_large2small[1], word_cnt_large2small[2])#最多
    word_cnt_small2large = sorted(word_cnt.items(), key=lambda d: d[1])
    print(word_cnt_small2large[0], word_cnt_small2large[1], word_cnt_small2large[2])#最少
    """
    [('3750', 7482224), ('648', 4924890), ('900', 3262544), ('3370', 2020958), ('6122', 1602363)]
    [('3133', 1), ('4468', 1), ('1015', 1), ('1415', 1), ('155', 1)]
    ('3750', 7482224) ('648', 4924890) ('900', 3262544)
    ('3165', 1) ('3282', 1) ('1079', 1)

    """

    ## 3.字符在文中出现的比例（比例高的很可能是标点符号）
    train_df['text_unique'] = train_df['text'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
    corpus = ' '.join(list(train_df['text_unique']))
    word_cnt = Counter(corpus.split(' ')) #因为每篇文章自身去重了，所以这里统计的是每个词出现在多少篇文章中
    word_cnt = sorted(word_cnt.items(), key=lambda d: d[1], reverse=True)
    word_rate = [(k, v/len(train_df)) for (k, v) in word_cnt]
    print(word_rate[0], word_rate[1], word_rate[2])
    """
    ('3750', 0.989985) ('900', 0.988265) ('648', 0.959875)
    """

    ## 4.每篇文章的句子数（假设比例最高的字符3750、900、648是句子的标签符号）
    # 训练集平均每篇文章有78个句子
    train_df['sentence_cnt'] = train_df['text'].apply(lambda x: len(re.split(' 3750 | 900 | 648 ', x))) #?
    print(train_df['sentence_cnt'].describe())
    """
    count    200000.000000
    mean         78.094350
    std          84.052108
    min           1.000000
    25%          27.000000
    50%          55.000000
    75%         100.000000
    max        3351.000000
    Name: sentence_cnt, dtype: float64
    """

    ## 5.按标签分别统计出现次数最多的字符：都是3750
    print(train_df.groupby('label')['text'].apply(lambda x: Counter(' '.join(list(x)).split(' ')).most_common(1)))
    """
    label
    0     [(3750, 1267331)]
    1     [(3750, 1200686)]
    2     [(3750, 1458331)]
    3      [(3750, 774668)]
    4      [(3750, 360839)]
    5      [(3750, 715740)]
    6      [(3750, 469540)]
    7      [(3750, 428638)]
    8      [(3750, 242367)]
    9      [(3750, 178783)]
    10     [(3750, 180259)]
    11      [(3750, 83834)]
    12      [(3750, 87412)]
    13      [(3750, 33796)]
    Name: text, dtype: object
    """

if __name__ == '__main__':
    count_feature()