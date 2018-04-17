from sklearn.base import BaseEstimator, TransformerMixin

# punctuations = string.punctuation
punctuations = ['!', '?', '.']


class PunctuationExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def getPuncCount(self, sen):
        count = 0
        # print(sen)
        for char in sen:
            if char in punctuations:
                count += 1
        return count

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.getPuncCount)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        print(self)
        return self
