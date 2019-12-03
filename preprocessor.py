import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


class Preprocessor:
    """ Class for preprocess text data.
        The Idea - bring data to one format.
        Includes several methods for working with text:
            remove_punctuation - replace punctuation symbols, like ",.- etc.",
            remove_stopwords - remove stopwords,
            tokenize - replace words with their stem (base),
            preprocess - make all together
    """
    translator_p = str.maketrans('', '', string.punctuation)
    stopwords = stopwords.words('english')
    stemmer = PorterStemmer()

    def remove_punctuation(self, string:str):
        return string.translate(self.translator_p)

    def remove_stopwords(self, string):
        return ' '.join([x for x in string.split() if x not in self.stopwords])

    def tokenize(self, string):
        return ' '.join([self.stemmer.stem(x) for x in string.split()])

    def preprocess(self, string):
        return list(
            self.tokenize(
                self.remove_stopwords(
                    self.remove_punctuation(
                        string.lower())))
            .split()
        )
