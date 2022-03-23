from extracting_keywords import *
    
if __name__ == '__main__':
    
    keys_dir = '../inspec/keys'
    docs_dir = '../inspec/docsutf8'
    
    docs = load_docs(docs_dir)
    no_docs = len(docs)
    ids, docs = list(docs.keys()), list(docs.values())
    keys = load_keys(keys_dir)

    precision, recall = 0, 0
    model = EKW_Model(docs)
    predict_keys = model.predict(docs)
    
    precision, recall = 0, 0
    for i, predict_key in enumerate(predict_keys):
        t1, t2 = eval(keys[ids[i]], predict_key)
        precision += t1
        recall += t2
    
    print(f"precision = {round(precision/2000, 3)}, recall = {round(recall/2000, 3)}")