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


listNegative = ['no', 'not', 'don\'t', 'didn', 'won\'t', 'haven\'t', 'isn\'t', 'doesn\'t', 'nor', 'weren\'t',
                'hadn\'t', 'isn', 'hasn\'t', 't', 'wasn\'t', 'didn\'t', 'couldn\'t']


def reviewToWordlist(sentence):
    sentenceText = BeautifulSoup(sentence, features="html.parser").get_text()
    sentenceText = re.sub("[^a-zA-Z]", " ", sentenceText)  # Оставляем только английские буквы
    words = sentenceText.lower().split()
    stops = set(nltk.corpus.stopwords.words("english"))
    stops.add("still")
    words = [w for w in words if not w in stops or w in listNegative]
    return words


def reviewToSentences(reviewText):
    raw_sentences = tokenizer.tokenize(reviewText.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(reviewToWordlist(raw_sentence))
    return sentences


def listWordToStr(listWord):
    string = ''
    for wordIndex in range(len(listWord)):
        string += listWord[wordIndex]
        if len(listWord)-wordIndex != 1:
            string += '_'
    return string


def getMarkSentenseList(sentenseList):
    markSentenceList = list()
    for sentence in sentenseList:
        markWordList = list()
        wordIndex = 0
        while wordIndex < len(sentence):
            for lenCollocation in range(min(3, len(sentence) - wordIndex), 0, -1):
                stringWord = listWordToStr(sentence[wordIndex:wordIndex + lenCollocation])
                collocationProperty = sn.data.get(stringWord)
                if collocationProperty is not None:
                    if wordIndex-1 >= 0 and sentence[wordIndex-1] in listNegative:
                        markWordList.append(
                            {"plsn": float(collocationProperty[0])*-1,
                            "attn": float(collocationProperty[1]),
                            "snst": float(collocationProperty[2]),
                            "aptt": float(collocationProperty[3]),
                            "plrt": float(collocationProperty[7])*-1,
                            "text": sentence[wordIndex-1]+'_'+stringWord})
                    else:
                        markWordList.append(
                            {"plsn": float(collocationProperty[0]),
                            "attn": float(collocationProperty[1]),
                            "snst": float(collocationProperty[2]),
                            "aptt": float(collocationProperty[3]),
                            "plrt": float(collocationProperty[7]),
                            "text": stringWord})
                    break
            wordIndex += lenCollocation
        markSentenceList.append(markWordList)
    return markSentenceList


def getReviewMark(reviewText):
    if reviewText[0].isalpha():
        reviewText = reviewText[0].lower() + reviewText[1:]  # Перевод первого слова в нижний регистр
    reviewText = re.sub(r'".*"', '', reviewText)  # Удаление двойных кавычек с содержимым
    reviewText = re.sub(r'\b[A-Z]+[a-z]+\b', '', reviewText)  # Удаление слов с заглавных букв, имен собственных
    sentenseList = reviewToSentences(reviewText)
    markList = getMarkSentenseList(sentenseList)
    return markList


def getReviewResult(sentenseList):
    mark = 0
    sentenseList = list(filter(None, sentenseList))
    for sentence in sentenseList:
        sentenseMark = 0
        for ngram in sentence:
            sentenseMark += ngram["plrt"]
            # sentenseMark += (ngram["plsn"] + abs(ngram["attn"]) - abs(ngram["snst"]) + ngram["aptt"]) / 9
        mark += sentenseMark / len(sentence)
    return mark / len(sentenseList)


def getLabel(result):
    totalMark = 0.15
    if result >= totalMark:
        return 1
    else:
        return 0


if __name__ == '__main__':
    start = time.time()
    train = pd.read_csv('Data//labeledSentensData.tsv', header=0, delimiter="\t", quoting=3)
    sn = SenticNet()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    countTrue = 0
    reviews = [ [review[0], getReviewMark(review[1]) ] for review in train.values]
    for review in reviews:
        result = getReviewResult(review[1])
        if review[0] == getLabel(result):
            countTrue += 1
    reviews = [ [review[0], getReviewResult(review[1]), review[1]] for review in reviews ]
    print("acuracy = ", countTrue/len(reviews))
    # np.array([ [review[0], review[1]] for review in reviews if review[0] == 1 ]).transpose()
    print("Done: ", time.time() - start, "seconds.")
    print()
