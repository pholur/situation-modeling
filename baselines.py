from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import make_pipeline
import numpy as np


class MyOwnTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X

##################################### Model 1 ##########################################

class RandomEntityClassifier(BaseEstimator, ClassifierMixin):
    # the naive classifier
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self
    def predict(self, X):
        return np.random.randint(0, self.classes_.size,
                                 size=X.shape[0])

##################################### Model 2 ##########################################

class SemiRandomWordMatching(BaseEstimator, ClassifierMixin):
    # not a scalable classifier, but it's a good example
    def fit(self, X, y):
        self.classes_ = {}
        for x,y in zip(X,y):
            if x not in self.classes_:
                self.classes_[x] = (0,0,0)
            self.classes_[x][y] += 1
        return self
    def predict(self, X):
        return_set = []
        for i,x in enumerate(X):
            if x not in self.classes_:
                return_set.append(np.random.randint(0, 3))
            else:
                return_set.append(np.argmax(self.classes_[x]))
        return np.array(return_set)

####################################### Model 3 ########################################
def clean_text(text):
    text = text.replace("*", "") # many more of these to come but may remove matches so be careful!
    text = text.replace("(", "") # many more of these to come but may remove matches so be careful!
    text = text.replace(")", "") # many more of these to come but may remove matches so be careful!
    return text

def clean2(text):
    import nltk
    from nltk.corpus import stopwords

    def purify_words(k):
        words = nltk.word_tokenize(k)
        words = [w for w in words if not w in stopwords.words('english')]
        words = [nltk.WordNetLemmatizer().lemmatize(w, 'n') for w in words]
        words = " ".join(words)
        return words
    
    return purify_words(text)


class SemiRandom_Stopword_Lemmatized_Lower_WordMatching(BaseEstimator, ClassifierMixin):
    # not a scalable classifier, but it's a good example
    def fit(self, X, y):
        self.classes_ = {}
        for x,y in zip(X,y):

            x = clean_text(x)
            x = x.lower()
            x = clean2(x)

            if x not in self.classes_:
                self.classes_[x] = (0,0,0)
            self.classes_[x][y] += 1
        return self

    def predict(self, X):
        return_set = []
        for i,x in enumerate(X):

            x = clean_text(x)
            x = x.lower()
            x = clean2(x)

            if x not in self.classes_:
                return_set.append(np.random.randint(0, 3))
            else:
                return_set.append(np.argmax(self.classes_[x]))
        return np.array(return_set)

######################################## Model 4 #######################################

#def special_results(test_X, test_y, model):







models = {"RANDOM":RandomEntityClassifier(), \
    "SEMI1":SemiRandomWordMatching(), \
    "SEMI2":SemiRandom_Stopword_Lemmatized_Lower_WordMatching(), \
    "OUR MODEL": None}

def get_score(train_data, test_data, model):
    """
    Random entity classifier
    """
    # Get the training and testing data
    train_X, train_y = train_data
    test_X, test_y = test_data
    # Create a random model
    if model != "OUR MODEL":
        pipe = make_pipeline(MyOwnTransformer(), models[model])
        # Train the model
        pipe.fit(train_X, train_y)
        # Test the model
        predictions = pipe.predict(test_X)
        # Pipe Score
        pipe_score = pipe.score(test_X, test_y)
        # Return the predictions
        return ("hard accuracy", pipe_score)
    # else:
    #     return ("hard accuracy", special_results(test_X, test_y, model))

