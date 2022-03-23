import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm

def load_keys(keys_dir):
    keys = {}
    for name in os.listdir(keys_dir):
        path = os.path.join(keys_dir, name)
        key = []
        with open(path, 'r') as file:
            texts = file.readlines()
            for text in texts:
                text = text.replace('\t', '')
                text = text.replace('\n', '')
                key.extend(text.lower().split())
        keys[int(name.split('.')[0])] = key
    return keys

def load_docs(docs_dir):
    docs = {}
    for name in os.listdir(docs_dir):
        path = os.path.join(docs_dir, name)
        doc = []
        with open(path, 'r') as file:
            texts = file.readlines()
            for text in texts:
                doc.append(text.lower())
        docs[int(name.split('.')[0])] = " ".join(doc)
    return docs

def eval(y, y_hat):
    no = 0
    for element in y_hat:
        if element in y:
            no += 1
    return (float(no)/ len(y_hat), float(no)/len(y))     

class EKW_Model:
    
    def __init__(self, corpus, n_keywords=5) -> None:
        self.n_keywords = n_keywords
        countvector = CountVectorizer()
        countvector.fit(corpus)
        vocab = countvector.vocabulary_
        self.vocab = [token for token in vocab if token.isalpha()]
        self.model = TfidfVectorizer(vocabulary=self.vocab)
        self.model.fit(corpus)

    def predict(self, texts):
        tfidf_texts = self.model.transform(texts).toarray()
        index_predict_keys = tfidf_texts.argsort(axis=1)[:,-self.n_keywords:]
        predict_keys = []
        for index_predcit_key in tqdm(index_predict_keys):
            predict_key = []
            for index in index_predcit_key:
                predict_key.append(self.vocab[index])
            predict_keys.append(predict_key)
        return predict_keys

def init_model():
    docs_dir = '../inspec/docsutf8'
    docs = load_docs(docs_dir)
    _, docs = list(docs.keys()), list(docs.values())
    model = EKW_Model(docs)
    return model