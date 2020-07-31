import pandas as pd
from gensim.models.word2vec import Word2Vec
from keras.layers import *
from keras.models import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.regularizers import L1L2, Regularizer
from keras.engine.topology import Layer
from keras import backend as K

def word2vec():
    """
    ## 初始化并训练一个Word2Vec：

        class Word2Vec(utils.SaveLoad):
            def __init__(
                    self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
                    max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                    sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                    trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH) # 默认参数

        · sentences：可以是一个list，对于大语料集，建议使用BrownCorpus,Text8Corpus或lineSentence构建。

        · size：是指特征向量的维度，默认为100。

        · alpha: 是初始的学习速率，在训练过程中会线性地递减到min_alpha。

        · window：窗口大小，表示当前词与预测词在一个句子中的最大距离是多少。

        · min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5。

        · max_vocab_size: 设置词向量构建期间的RAM限制，设置成None则没有限制。

        · sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)。

        · seed：用于随机数发生器。与初始化词向量有关。

        · workers：用于控制训练的并行数。

        · min_alpha：学习率的最小值。

        · sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。

        · hs: 如果为1则会采用hierarchica·softmax技巧。如果设置为0（默认），则使用negative sampling。

        · negative: 如果>0,则会采用negativesampling，用于设置多少个noise words（一般是5-20）。

        · cbow_mean: 如果为0，则采用上下文词向量的和，如果为1（default）则采用均值，只有使用CBOW的时候才起作用。

        · hashfxn： hash函数来初始化权重，默认使用python的hash函数。

        · iter： 迭代次数，默认为5。

        · trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）。

        · sorted_vocab： 如果为1（默认），则在分配word index 的时候会先对单词基于频率降序排序。

        · batch_words：每一批的传递给线程的单词的数量，默认为10000。


        一些参数的选择与对比：

        1.skip-gram （训练速度慢，对罕见字有效），CBOW（训练速度快）。一般选择Skip-gram模型；

        2.训练方法：Hierarchical Softmax（对罕见字有利），Negative Sampling（对常见字和低维向量有利）；

        3.欠采样频繁词可以提高结果的准确性和速度（1e-3~1e-5）

        4.Window大小：Skip-gram通常选择10左右，CBOW通常选择5左右。

    ## Word2Vec的保存、加载、继续训练
        https://blog.csdn.net/qq_19707521/article/details/79169826

    ## Word2Vec的属性
        https://juejin.im/post/6844903912256831502
        词数、词典、用法

    :return:
    """
    train_path = './data/train_set.csv'
    test_path = './data/test_a.csv'

    # train_df = pd.read_csv(train_path, sep='\t', nrows=10000)
    # test_df = pd.read_csv(test_path, sep='\t', nrows=5000)
    # print(train_df.shape)
    # print(test_df.shape)
    # data_df = pd.concat([train_df, test_df], axis=0)
    # print(data_df.shape)
    #
    # sentences = data_df['text'].apply(lambda x: x.split(' ')).values.tolist() #sentences=[['i', 'am', 'wei'], ['hello','everyone']]
    # # print(sentences[:5])
    #
    # # Word2Vec
    # num_features = 50  # Word vector dimensionality
    # min_word_count = 1  # Minimum word count
    # num_workers = 4  # Number of threads to run in parallel
    # window_size = 5  # Context window size
    # model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = window_size)
    # model.save('./model/word2vec_20200731')
    model = Word2Vec.load('./model/word2vec_20200731')
    print(model.wv['4149'])


def textcnn():
    def build_model(max_len=1000, embed_size=100, word_size=1000):
        input = Input(shape=(max_len,), dtype='int32')
        embedding = Embedding(word_size + 1, embed_size, input_length=max_len, trainable=True, name="Embedding")(input)
        embedding_reshape = Reshape((1000, 300, 1), name="Embedding_reshape")(embedding)
        conv_pools = []
        filters = [2, 3, 4]
        for filter in filters:
            conv = Conv2D(filters=100,
                          kernel_size=(filter, 300),
                          padding='valid',
                          kernel_initializer='normal',
                          activation='relu',
                          kernel_regularizer=l2(0.001), name="Conv%s" % (filter))(embedding_reshape)
            pooled = MaxPool2D(pool_size=(max_len - filter + 1, 1),
                               strides=(1, 1),
                               padding='valid',
                               name="Pooling%s" % (filter))(conv)
            conv_pools.append(pooled)
        x = Concatenate(axis=-1)(conv_pools)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        logits = Dense(units=1, kernel_regularizer=l2(0.001))(x)
        output = Activation('sigmoid')(logits)
        model =  Model(inputs=input, outputs=output, name="TextCNN")
        return model

    model = build_model()
    print(model.summary())

    pass

def textrnn():
    def build_model(max_len=1000, embed_size = 100, word_size=1000):
        input = Input(shape=(max_len,), dtype='int32')
        embedding = Embedding(word_size + 1, embed_size, input_length=max_len, trainable=True, name="Embedding")(input)
        x = embedding
        for layer_i in range(2):
            x = Bidirectional(LSTM(units=100,
                                     return_sequences=True,
                                     activation='relu',
                                     kernel_regularizer=regularizers.l2(0.32 * 0.1),
                                     recurrent_regularizer=regularizers.l2(0.32)
                                         ))(x)
            x = Dropout(0.2)(x)
        x = Flatten()(x)
        logits = Dense(units=1, kernel_regularizer=l2(0.001))(x)
        output = Activation('sigmoid')(logits)
        model = Model(inputs=input, outputs=output, name="TextRNN")
        return model

    model = build_model()
    print(model.summary())

    pass

def han():
    def build_model(max_len=1000, embed_size=100, word_size=1000):
        class AttentionSelf(Layer):
            """
                self attention,
                codes from:  https://mp.weixin.qq.com/s/qmJnyFMkXVjYBwoR_AQLVA
            """

            def __init__(self, output_dim, **kwargs):
                self.output_dim = output_dim
                super().__init__(**kwargs)

            def build(self, input_shape):
                # W、K and V
                self.kernel = self.add_weight(name='WKV',
                                              shape=(3, input_shape[2], self.output_dim),
                                              initializer='uniform',
                                              regularizer=L1L2(0.0000032),
                                              trainable=True)
                super().build(input_shape)

            def call(self, x):
                WQ = K.dot(x, self.kernel[0])
                WK = K.dot(x, self.kernel[1])
                WV = K.dot(x, self.kernel[2])
                # print("WQ.shape",WQ.shape)
                # print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)
                QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
                QK = QK / (64 ** 0.5)
                QK = K.softmax(QK)
                # print("QK.shape",QK.shape)
                V = K.batch_dot(QK, WV)
                return V

            def compute_output_shape(self, input_shape):
                return (input_shape[0], input_shape[1], self.output_dim)

        def word_level(rnn_units=100):
            x_input_word = Input(shape=(max_len, embed_size))
            # x = SpatialDropout1D(self.dropout_spatial)(x_input_word)
            x = Bidirectional(GRU(units=rnn_units,
                                  return_sequences=True,
                                  activation='relu',
                                  kernel_regularizer=regularizers.l2(0.001),
                                  recurrent_regularizer=regularizers.l2(0.001)))(x_input_word)
            out_sent = AttentionSelf(rnn_units * 2)(x)
            model = Model(x_input_word, out_sent)
            return model

        def sentence_level(rnn_units=100*2):
            x_input_sen = Input(shape=(max_len, rnn_units))
            # x = SpatialDropout1D(self.dropout_spatial)(x_input_sen)
            output_doc = Bidirectional(GRU(units=rnn_units * 2,
                                           return_sequences=True,
                                           activation='relu',
                                           kernel_regularizer=regularizers.l2(0.001),
                                           recurrent_regularizer=regularizers.l2(0.001)))(x_input_sen)
            output_doc_att = AttentionSelf(embed_size)(output_doc)
            model = Model(x_input_sen, output_doc_att)
            return model

        input = Input(shape=(max_len,), dtype='int32')
        embedding = Embedding(word_size + 1, embed_size, input_length=max_len, trainable=True, name="Embedding")(input)
        x_word = word_level()(embedding)
        x_word_to_sen = Dropout(0.2)(x_word)
        x_sen = sentence_level()(x_word_to_sen)
        x_sen = Dropout(0.2)(x_sen)
        x_sen = Flatten()(x_sen)
        logits = Dense(units=1, kernel_regularizer=l2(0.001))(x_sen)
        output = Activation('sigmoid')(logits)
        model = Model(inputs=input, outputs=output, name="HAN")
        return model

    model = build_model()
    print(model.summary())

    pass


if __name__=='__main__':
    # word2vec()
    # textcnn()
    # textrnn()
    han()