from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import re
import nltk
import numpy as np
import logging
from senticnet.senticnet import SenticNet

'''

'''


def reviewToWordlist(review, removeStopwords=False):
    reviewText = BeautifulSoup(review, features="html.parser").get_text()
    reviewText = re.sub("[^a-zA-Z]", " ", reviewText)  # Оставляем только английские буквы
    words = reviewText.lower().split()
    if removeStopwords:
        stops = set(nltk.corpus.stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences += reviewToWordlist(raw_sentence, remove_stopwords)
    return sentences



if __name__ == '__main__':
    start = time.time()
    # Настройка параметров
    # numFeatures = 300  # Количество простанств вектора слова
    # minWordCount = 50  # Minimum word count
    # num_workers = 16  # Number of threads to run in parallel
    # context = 10  # Context window size
    # downsampling = 1e-3  #  Downsample setting for frequent words

    train = pd.read_csv('Data//labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    # unlabeled_train = pd.read_csv('Data//unlabeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    # test = pd.read_csv('Data//testData.tsv', header=0, delimiter="\t", quoting=3)
    # labelTest = pd.read_csv('Data//LabeledTestData.csv', header=0, delimiter=",", quoting=0)

    sn = SenticNet()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    trueAccuracy = 0
    for review in train.values:
        wordList = review_to_sentences(review[2], tokenizer, True)
        sumMark = 0.0
        for word in wordList:
            wordProperty = sn.data.get(word)
            if wordProperty is not None:
                if abs(float(wordProperty[7]))>=0.5:
                    sumMark += float(wordProperty[7])
        if sumMark >= 0.5 and review[1] == 1:
            trueAccuracy += 1
        elif sumMark < 0 and review[1] == 0:
            trueAccuracy += 1

    print()

    # Verify the number of reviews that were read (100,000 in total)
    # print("Read %d labeled train reviews, %d labeled test reviews, "
    #       "and %d unlabeled reviews\n" % (train["review"].size,
    #                                       test["review"].size, unlabeled_train["review"].size))



    # modelName = "big300features_40minwords_10context"
    #
    # # Создание модели Word2Vec
    # createModel(modelName)
    #
    # model = Word2Vec.load('Model//' + modelName)
    # word_vectors = model.wv.vectors
    # num_clusters = int(word_vectors.shape[0] / 10) # Рекомендуется 5 слов на кластер
    # print("Running K means, where K = ", num_clusters)
    # kmeans_clustering = KMeans(n_clusters=num_clusters)
    # idx = kmeans_clustering.fit_predict(word_vectors)
    # word_centroid_map = dict(zip(model.wv.index2word, idx))
    #
    # train_centroids = createCentroids(train)
    # test_centroids = createCentroids(test)
    # labelTest_centroids = createCentroids(labelTest)
    #
    # forest = RandomForestClassifier(n_estimators=100)
    # print("Fitting a random forest to labeled training data...")
    # forest = forest.fit(train_centroids, train["sentiment"])
    #
    # result = forest.predict(test_centroids)
    # output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    # output.to_csv("Data//Result.csv", index=False, quoting=3)
    #
    # deviation = forest.predict(labelTest_centroids) # deviation - отклонение(ошибка)
    # deviationDF = pd.DataFrame(data={"id": labelTest["id"], "sentiment" : labelTest["sentiment"],
    #                                  "deviationSentiment" : deviation})
    # deviationDF.to_csv("Data//ResultDeviation.csv", index=False, quoting=3)
    #
    # deviationCount = 0.0
    # for index, row in deviationDF.iterrows():
    #     if row["sentiment"] == row["deviationSentiment"]:
    #         deviationCount += 1
    # print("Количество слов: ", word_vectors.shape[0] )
    # print("Количество простанств вектора слова: ", numFeatures)
    # print("Минимальное кол-во слова: ", minWordCount)
    # print("Точность: ", (deviationCount / len(deviationDF)))
    print("Done: ", time.time() - start, "seconds.")
