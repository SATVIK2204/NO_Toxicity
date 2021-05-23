import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import wordpunct_tokenize

from collections import Counter


class Dfcleaner:
    def __init__(self):
        pass

    def __remove(self, text, frequent, rare, FREQWORDS, RAREWORDS):

        if frequent:
            text = " ".join(
                [word for word in str(text).split() if word not in FREQWORDS]
            )
        if rare:
            text = " ".join(
                [word for word in str(text).split() if word not in RAREWORDS]
            )

        return text

    # Remove the most frequent words
    def remove_frequent_rare(
        self, raw_comments, frequent=False, n_freq=10, rare=False, n_rare=10
    ):
        cnt = Counter()
        for comment in raw_comments:
            for word in comment.split():
                cnt[word] += 1

        FREQWORDS = set([w for (w, wc) in cnt.most_common(n_freq)])
        RAREWORDS = set([w for (w, wc) in cnt.most_common()[: -n_rare - 1 : -1]])
        length=len(raw_comments)
        i=0
        cleaned_comments = []
        for comment in raw_comments:
            cleaned_comments.append(self.__remove(comment, frequent, rare, FREQWORDS, RAREWORDS))
            i = i + 1
            if i % 10000 == 0:
                print(f"{i} examples cleaned out of {length}")
            if i==length:
                print('Cleaning Done')

        cleaned_comments=[x for x in cleaned_comments if str(x)!='nan']

        return cleaned_comments

    def __clean_text(self, text, remove_stopwords, stem, lemitize):
        # Lower the text
        text = text.lower()

        # Remove unwanted punctuations
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"  ", " ", text)
        text = re.sub(r'[_"”\-;%()’“|‘+&=*%.!,?:#$@\[\]/]', " ", text)
        text = re.sub(r"\'", " ", text)
        text = re.sub("\s+", " ", text).strip()

        # Remove stopwords
        if remove_stopwords:
            text = text.split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)

        if stem:
            stemmer = PorterStemmer()
            text = [stemmer.stem(word) for word in text.split()]
            text = " ".join(text)

        if lemitize:
            lemmatizer = WordNetLemmatizer()
            # Here we will use POS-tagging to pass on to lemmatizer using wornet to get the correct lemmitization
            wordnet_map = {
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "J": wordnet.ADJ,
                "R": wordnet.ADV,
            }
            pos_tagged_text = nltk.pos_tag(text.split())
            text = [
                lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN))
                for word, pos in pos_tagged_text
            ]
            text = " ".join(text)

        return text

    def clean(self, raw_comments, remove_stopwords=True, stem=True, lemitize=True, is_prediction=False):
        cleaned_comments = []
        i = 0
        length = len(raw_comments)
        for comment in raw_comments:
            cleaned_comments.append(
                self.__clean_text(comment, remove_stopwords, stem, lemitize)
            )
            if is_prediction==False:
                i = i + 1
                if i % 10000 == 0:
                    print(f"{i} examples cleaned out of {length}")
                if i == length:
                    print("Cleaning Done")
        cleaned_comments=[x for x in cleaned_comments if str(x)!='nan']

        return cleaned_comments
