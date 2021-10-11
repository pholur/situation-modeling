from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import make_pipeline
import numpy as np


class MyOwnTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X

##################################### Model 0 ##########################################
class DeterministicClassifier0(BaseEstimator, ClassifierMixin):
    # the naive classifier
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.array([0]*len(X))

class DeterministicClassifier1(BaseEstimator, ClassifierMixin):
    # the naive classifier
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.array([1]*len(X))

class DeterministicClassifier2(BaseEstimator, ClassifierMixin):
    # the naive classifier
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.array([2]*len(X))


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
                self.classes_[x] = [0,0,0]
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
# clustype
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
                self.classes_[x] = [0,0,0]
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

from imports import *
from data import *
from arch import *

model = return_token_model(CORE_MODEL, len(labels), preferred_cuda_test);
#model.load_state_dict(torch.load(CHECKPOINT_PATH + ROOT_NAME + '3_17000.pt')["model_state_dict"]);
#model.load_state_dict(torch.load("/mnt/SSD2/pholur/CTs/checkpoints/Day_0928_Insider_Outsider_79_54480.pt")["model_state_dict"]);
#model.load_state_dict(torch.load("/mnt/SSD2/pholur/CTs/checkpoints/finalized_checkpoints/Day_1007_Insider_Outsider_49_34050.pt")["model_state_dict"]);
model.load_state_dict(torch.load(MODEL_IN_TESTING)["model_state_dict"]);
#model.load_state_dict(torch.load("/mnt/SSD2/pholur/CTs/checkpoints/Day_1006_Insider_Outsider_21_14982.pt")["model_state_dict"]);
model.eval();

from shared_train_and_test_functions import tokenization
import spacy
from spacy import displacy
import nltk
nlp = spacy.load('en_core_web_sm')

from data import *
def return_insiders_and_outsiders(text, option_display=False, sample_text=None):
    text = text.lower()
    text = clean_text(text)
    doc = nlp(text)
    add_np_text = []
    spans = []

    FLAG = False
    if sample_text == None:
        FLAG = True

    for npy in doc.noun_chunks:
        add_np_text.append(npy.text)
        if npy.text == sample_text:
            FLAG = True
        spans.append((npy.start_char, npy.end_char))

    test_encodings, tokens = tokenization([text], True) # you can print the tokens here for debugging
    offset_mappings = test_encodings.offset_mapping[0]

    input_ids = torch.tensor(test_encodings['input_ids']).to(preferred_cuda_test)
    attention_mask = torch.tensor(test_encodings['attention_mask']).to(preferred_cuda_test)
    outputs = model(input_ids, attention_mask=attention_mask) # try this: labels = torch.tensor([0]*len(input_ids)).to(preferred_cuda_test))
    predictions = outputs[0].detach().cpu().numpy()
    single_prediction_post = predictions[0]
    predictions_argmax = np.argmax(single_prediction_post, axis=1)

    ents = []
    
    if FLAG:
        for j in range(len(spans)):
            
            choice = "no"
            count_votes = []
            for i,index in enumerate(predictions_argmax):
                current_token_mapping = offset_mappings[i]

                if current_token_mapping[1] >= spans[j][0] and current_token_mapping[0] <= spans[j][1]: # if the token is in the span of a chunk
                    if index == 2:
                        count_votes.extend([index])
                    elif index == 1:
                        count_votes.extend([index])
                    else:
                        count_votes.extend([index])
            try:
                ents.append({"start": spans[j][0], "end": spans[j][1], "label": reverse_labels[np.bincount(count_votes).argmax()]})
            except:
                print("Something's wrong with the span intersection.")
    else:
        # in case our model cannot find the chunk
        start_index = text.find(sample_text)
        end_index = start_index + len(sample_text)
        count_votes = []

        for i,index in enumerate(predictions_argmax):

            current_token_mapping = offset_mappings[i]
            if current_token_mapping[1] >= start_index and current_token_mapping[0] <= end_index: # if the token is in the span of a chunk
                if index == 2:
                    count_votes.extend([index])
                elif index == 1:
                    count_votes.extend([index])
                else:
                    count_votes.extend([index])
        try:
            ents.append({"start": start_index, "end": end_index, "label": reverse_labels[np.bincount(count_votes).argmax()]})
        except:
            print("Something's wrong with the span intersection.")

    # Requirements for using Displacy correctly
    # 1. Deduplicate list of dictionaries
    ents = [dict(t) for t in {tuple(d.items()) for d in ents}]

    # 2. Sort list of dictionaries by start index
    ents = sorted(ents, key=lambda x: x['start'])

    options = {"colors": {"INSIDER":"#88C6F1", "OUTSIDER":"#FF82AB"}} #blue, #red
    ex = [{"text": text, "ents": ents}]

    if option_display == True: # return for show
        displacy.render(ex, style="ent", manual=True, options=options)
        
    else: # return the ents for compute
        dictionary_map = {}
        for ent in ents:
            np_all = text[ent['start']:ent['end']].lower()
            np_all = np_all.lower()
            np_all = clean_text(np_all)
            dictionary_map[np_all] = labels[ent['label']]
        return dictionary_map

class OurModel(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X_rep):
        missed = 0
        X, X_ = X_rep[0], X_rep[1]
        output = []
        for x,x_ in zip(X,X_):

            x = x.lower()
            x = clean_text(x)

            x_ = x_.lower()
            x_ = clean_text(x_)

            returned_dict = return_insiders_and_outsiders(x_, False)
            if x in returned_dict:
                output.append(returned_dict[x])
            else:
                returned_dict = return_insiders_and_outsiders(x_, False, x)
                output.append(returned_dict[x])
                missed += 1 # missed are important
        return np.array(output)


models = {"RANDOM":RandomEntityClassifier(), \
    "DET0": DeterministicClassifier0(), \
    "DET1": DeterministicClassifier1(), \
    "DET2": DeterministicClassifier2(), \
    "SEMI1":SemiRandomWordMatching(), \
    "SEMI2":SemiRandom_Stopword_Lemmatized_Lower_WordMatching(), \
    "OUR MODEL": None}

def get_score(train_data, test_data, model, context):
    train_X, train_y = train_data
    test_X, test_y = test_data
    if model != "OUR MODEL":
        pipe = make_pipeline(MyOwnTransformer(), models[model])
        pipe.fit(train_X, train_y)
        pipe_score = pipe.score(test_X, test_y)
        return pipe_score
    else:
        pipe = make_pipeline(MyOwnTransformer(), OurModel())
        pipe.fit(train_X, train_y)
        pipe_score = pipe.score([test_X, context], test_y)
        return pipe_score

