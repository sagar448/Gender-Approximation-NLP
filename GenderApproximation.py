#Gender guessing algorithm
import sys
sys.path.append("/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages")
import random
import nltk
from nltk.corpus import names
import re
prondict = nltk.corpus.cmudict.dict()

def gender_features(x):
    return {"suffix1": x[-1:],
            "suffix2": x[-2:],
            "length": len(x),
            "bigrams": tuple(re.findall(r"(?=([a-zA-Z]{2}))", x)),
            "suffix3": x[:2],
            "lastLetter2": x[-1] == "y"}


names = ([(name, "male") for name in names.words("male.txt")] + [(name, "female") for name in names.words("female.txt")])
random.shuffle(names)

train_names = names
#devtest_names = names[:2000]

train_set = [(gender_features(n), g) for (n,g) in train_names]
#devtest_set = [(gender_features(n), g) for (n,g) in devtest_names] #used to perform error analysis

classifier = nltk.NaiveBayesClassifier.train(train_set)
#print(nltk.classify.accuracy(classifier, devtest_set))

#errors = []
#for (name, tag) in devtest_names:
#    guess = classifier.classify(gender_features(name))
#    if guess != tag:
#        errors.append((tag, guess, name))

#for (tag, guess, name) in sorted(errors): # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
#    print("correct=%-8s guess=%-8s name=%-30s" %(tag, guess, name))


        
