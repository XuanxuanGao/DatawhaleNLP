import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import fasttext
import os

def text_classification_fastText():
    """
    使用fastText包进行文本分类
    - 官方代码：https://github.com/facebookresearch/fastText/tree/master/python
    - 论文：https://arxiv.org/abs/1607.01759
    - 官网：https://fasttext.cc/
    - 用法：
        1. 安装fastText包
        2. 处理数据格式
        3. 训练
        4. 十折交叉验证调参
    - 可调参数：
        input             # training file path (required)
        lr                # learning rate [0.1]
        dim               # size of word vectors [100]
        ws                # size of the context window [5]
        epoch             # number of epochs [5]
        minCount          # minimal number of word occurences [1]
        minCountLabel     # minimal number of label occurences [1]
        minn              # min length of char ngram [0]
        maxn              # max length of char ngram [0]
        neg               # number of negatives sampled [5]
        wordNgrams        # max length of word ngram [1]
        loss              # loss function {ns, hs, softmax, ova} [softmax]
        bucket            # number of buckets [2000000]
        thread            # number of threads [number of cpus]
        lrUpdateRate      # change the rate of updates for the learning rate [100]
        t                 # sampling threshold [0.0001]
        label             # label prefix ['__label__']
        verbose           # verbose [2]
        pretrainedVectors # pretrained word vectors (.vec file) for supervised learning []
    """

    from sklearn.model_selection import StratifiedKFold

    # path setting
    train_path = './data/train_set.csv'
    test_path = './data/test_a.csv'
    train_foldk_path = 'train_fold_{k}.csv'
    valid_foldk_path = 'valid_fold_{k}.csv'
    model_foldk_path = './model/model_fasttext_fold_{k}.bin'

    train_df = pd.read_csv(train_path, sep='\t')
    test_df = pd.read_csv(test_path, sep='\t')
    print(train_df.shape)
    print(test_df.shape)

    skf = StratifiedKFold(n_splits=10, random_state=2020, shuffle=True)
    oof_pred = np.zeros(train_df.shape[0])
    sub_pred = np.zeros(test_df.shape[0])
    score = []
    for i, (train_index, valid_index) in enumerate(skf.split(train_df, train_df['label'])):
        print("FOLD | ", i + 1)
        print("###" * 35)
        data_train_kfold_path = train_foldk_path.format(k=str(i+1))
        data_valid_kfold_path = valid_foldk_path.format(k=str(i+1))
        model_foldk_path = model_foldk_path.format(k=str(i+1))
        print(model_foldk_path)

        ####### prepare data
        train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
        train_df[['text', 'label_ft']].iloc[train_index].to_csv(data_train_kfold_path, index=None, header=None, sep='\t')  # 训练集
        train_df[['text', 'label']].iloc[valid_index].to_csv(data_valid_kfold_path, index=None, sep='\t')  # 验证集

        # 模型训练及保存
        model = fasttext.train_supervised(data_train_kfold_path, lr=1.0, wordNgrams=2, verbose=2, minCount=1, epoch=25, loss="hs")
        # print(model.words[:5])  # ['3750', '648', '900', '3370', '6122']
        # print(model.labels)  # ['__label__0', '__label__1', '__label__2', '__label__3', '__label__4', '__label__5', '__label__6', '__label__7', '__label__8', '__label__9', '__label__10', '__label__11', '__label__12', '__label__13']
        model.save_model(model_foldk_path)

        # # 查看性能
        # def print_results(N, p, r):
        #     print("N\t" + str(N))  # N	5000
        #     print("P@{}\t{:.3f}".format(1, p))  # P@1	0.875
        #     print("R@{}\t{:.3f}".format(1, r))  # R@1	0.875
        #
        # model = fasttext.load_model("./model/model_fasttext.bin")
        # print_results(*model.test('valid.csv'))
        # """
        # N	5000
        # P@1	0.876
        # R@1	0.876
        # """

        # # 模型压缩
        # model.quantize(input='train.csv', retrain=True)
        # model.save_model("./model/model_fasttext_quantized.ftz")
        # model = fasttext.load_model("./model/model_fasttext_quantized.ftz")
        # print_results(*model.test('valid.csv'))
        # """
        # N	5000
        # P@1	0.865
        # R@1	0.865
        # """

        # # 模型验证
        model = fasttext.load_model(model_foldk_path)
        # print(model.predict(["3750 648 900","3370 6122"]))  # ([['__label__8'], ['__label__8']], [array([1.00005], dtype=float32), array([0.9955089], dtype=float32)])
        valid_df = pd.read_csv(data_valid_kfold_path, sep='\t')
        X_valid = list(valid_df['text'])
        y_valid_pred = model.predict(X_valid)
        y_valid_pred = np.array([int(x[0].split('__')[-1]) for x in y_valid_pred[0]])
        oof_pred[valid_index] = y_valid_pred
        y_valid = valid_df['label'].values
        f1score = f1_score(y_valid, y_valid_pred, average='macro')
        print(f1score)  # 0.8239
        score.append(f1score)


        ## 模型预测
        X_test = list(test_df['text'])
        y_test_pred = model.predict(X_test)
        y_test_pred = np.array([int(x[0].split('__')[-1]) for x in y_test_pred[0]])
        sub_pred += y_test_pred

    print("10 fold f1_score:\n", score)
    print("mean f1_score:\n", np.mean(score))

    # 处理test集的结果
    sub_pred_avg = sub_pred / skf.n_splits
    test_df['label'] = sub_pred_avg
    test_df['label'] = test_df['label'].apply(lambda x: int(round(x, 0)))
    print(test_df.shape)  #
    test_df[['label']].to_csv('./data/submission_fasttext_20200727.csv', index=False)
    print(test_df['label'].value_counts())


if __name__ == '__main__':
    text_classification_fastText()



