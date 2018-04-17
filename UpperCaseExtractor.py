from sklearn.base import BaseEstimator, TransformerMixin


class CapitalExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def getCapCount(self, sen):
        count = 0
        # print('Caps', sen)
        for char in sen:
            if char.isupper():
                count += 1
        # print(count)
        return count

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.getCapCount)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        # print(self)
        return self