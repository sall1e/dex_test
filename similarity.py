import os
from collections import Counter
import nltk
import gensim
from flask_restful import Resource, reqparse
from configs import Configs
from preprocessor import Preprocessor


class SimilarityMeasurer(Resource):
    """ Class for handle measuring distance between two documents.
        It is implemented: Jaccard index, Overlap coefficient, Levenshtein distance
            and Word Mover’s Distance using gensim model.
        Implemented postmethod with 3 parameters:
            string1 - first string
            string2 - second string
            measure - what measure to use
    """
    argparser = reqparse.RequestParser()
    argparser.add_argument(
        'measure', help='measure to count distance with', type=str, default='wmd',
        choices=Configs.MEASURES)
    argparser.add_argument('string1', type=str, required=True, help='first string must be passed')
    argparser.add_argument('string2', type=str, required=True, help='second string must be passed')
    print('Loading model')
    model = gensim.models.KeyedVectors.load_word2vec_format(Configs.MODEL_PATH, binary=True)
    print('Model have been loaded')

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.fun_mapper = {
            'wmd': self.wmd_distance,
            'jaccard': self.jaccard_index,
            'overlap': self.overlap_coef,
            'levenshtein': self.levinstain
        }

    def post(self):
        data = self.argparser.parse_args()
        distance = self.fun_mapper[data['measure']](data['string1'], data['string2'])
        return {'distance': distance}

    def _preprocess(self, v1, v2):
        return [self.preprocessor.preprocess(x) for x in (v1, v2)]

    def wmd_distance(self, v1, v2):
        v1, v2 = self._preprocess(v1, v2)
        return self.model.wmdistance(v1, v2)

    def jaccard_index(self, v1, v2):
        v1, v2 = self._preprocess(v1, v2)
        c1, c2 = Counter(v1), Counter(v2)
        return sum((c1 & c2).values()) / (sum((c1 + c2).values()) - sum((c1 & c2).values()))

    def overlap_coef(self, v1, v2):
        v1, v2 = self._preprocess(v1, v2)
        s1, s2 = set(v1), set(v2)
        return len(s1.intersection(s2)) / min(len(s1), len(s2))

    def levinstain(self, v1, v2):
        return nltk.edit_distance(v1, v2)
