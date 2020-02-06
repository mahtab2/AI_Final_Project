from nltk.classify import SklearnClassifier
from random import shuffle
from sklearn.naive_bayes import MultinomialNB
import re
import pandas as pd
from sklearn.metrics import accuracy_score


rawData = []
rawtest = []
testid=[]

###################################################################

def flatten(lst):
    for el in lst:
        if isinstance(el, list):
            yield from el
        else:
            yield el

def concat(comment , title):
    if(pd.isnull(title)==False and pd.isnull(comment)==False):
        comment=comment+" "+title
    if(pd.isnull(title)==False and pd.isnull(comment)==True):
        comment=title
    return comment

def parseReview(commentLine):
    return commentLine['id'],commentLine['text'], commentLine['verification_status'] , commentLine['rate']

def parsetestReview(commentLine):
    return commentLine['id'],commentLine["text"],commentLine['rate']

def preProcess(text):
    if(pd.isnull(text)==False):
        text=re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        #tokens=[t for t in text.split() if t not in stopwords]
        return tokens
    tokens=[]
    return tokens


def toVector(tokens):
    v = {}
    for t in tokens:
        try:
            v[t] +=1
        except KeyError:
            v[t] = 1
    return v

def loadData():
    for index, row in train.iterrows():
            (Id, Text, Label ,Rate) = parseReview(row)
            rawData.append((Id, Text, Label,Rate))

def loadtestData():
    for index, row in testcsv.iterrows():
            (Id, Text ,Rate) = parsetestReview(row)
            rawtest.append((Id, Text,Rate))
##################################################################################
def predictLabels(reviewTest, classifier):
    return classifier.classify_many(map(lambda t: t[1], reviewTest))

def trainClassifier(trainData):
    print("Training Classifier...")
    return SklearnClassifier(MultinomialNB()).train(trainData)

def crossValidate(dataset, folds ,testdata):
    shuffle(dataset)
    predictions = []
    train_real_res = []
    testPredicte=[]

    foldSize = int(len(dataset) / folds)
    dataset = [(t[0], toVector(preProcess(t[1])), t[2]) for t in dataset]
    testdata = [(t[0], toVector(preProcess(t[1]))) for t in testdata]

    for i in range(0, len(dataset), foldSize):
        trainFolds = dataset[:i] + dataset[i + foldSize:] # train part of training
        validationFold = dataset[i: i + foldSize]  #test part of training

        training_set = [(t[1], t[2]) for t in trainFolds]
        classifier = trainClassifier(training_set)

        if(i+foldSize<=len(testdata)):
            print("testing ...")
            testfold = testdata[i:i + foldSize]
            testPredicte.append(predictLabels(testfold,classifier))
        predictions.append(predictLabels(validationFold, classifier))
        train_real_res.append([l[2] for l in validationFold])

    return train_real_res, predictions ,testPredicte

######################################################################
#main part
stop_words= []
with open("stopwords-fa.txt" ,encoding = 'utf-8' ) as f:
    stop_words = f.read().splitlines()
stopwords = stop_words [1:]

train = pd.read_csv('train.csv',encoding='utf-8')
train["text"]=train["comment"]+" "+train["title"]
testcsv = pd.read_csv('test.csv', encoding='utf-8')
testcsv["text"]=testcsv["comment"]+" "+testcsv["title"]
testid = testcsv['id']

loadtestData()
loadData()

train_real_res, predictions ,testPredicte= crossValidate(rawData, 8 ,rawtest)
train_real_res = list(flatten(train_real_res))
predictions = list(flatten(predictions))
testPredicte=list(flatten(testPredicte))


print(testPredicte)
print(len(testPredicte))
#print(len(id))
verification=pd.DataFrame({"id":testid,
    "verification_status":testPredicte ,})
verification.to_csv("ans.csv",index=False)
