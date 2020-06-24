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


listNegative = ['no', 'not']


def reviewToWordlist(sentence):
    sentenceText = BeautifulSoup(sentence, features="html.parser").get_text()
    sentenceText = re.sub("[^a-zA-Z]", " ", sentenceText)  # Оставляем только английские буквы
    words = sentenceText.lower().split()
    return (words)


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
                        markWordList.append([float(collocationProperty[7])*-1, sentence[wordIndex-1]+'_'+stringWord])
                    else:
                        markWordList.append([float(collocationProperty[7]), stringWord])
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


if __name__ == '__main__':
    start = time.time()
    train = pd.read_csv('Data//labeledSentensData.tsv', header=0, delimiter="\t", quoting=3)
    sn = SenticNet()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    resultReviews = [ [review[0], getReviewMark(review[1]) ] for review in train.values]


    print("Done: ", time.time() - start, "seconds.")
    print()
