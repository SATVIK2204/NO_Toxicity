import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


class Dfcleaner:
    def __init__(self):
        pass

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

    def clean(self, df, remove_stopwords=True, stem=True, lemitize=True):
        raw_comments = df["comment_text"].to_list()
        cleaned_comments = []

        for comment in raw_comments:
            cleaned_comments.append(
                self.__clean_text(comment, remove_stopwords, stem, lemitize)
            )

        df.iloc[:, "comment_text"] = raw_comments
        return df
