from gensim.models.doc2vec import Doc2Vec


save_path = r'D:\PycharmProjects\parliamentary_emotions\special_embeddings\data'
model = Doc2Vec.load(save_path + 'doc2vec_0.model')

corpus = pd.read_csv(r'D:\PycharmProjects\parliamentary_emotions\data/abortion_with_metrics') #### INNY KORPUS


person_labels = ['PERSON_TAG' + '_' + str(row[1]['kto']) for row in corpus.iterrows()]
person_dict = {tag: tag.replace('PERSON_TAG_', '') for tag in person_labels}
person_reverse_dict = {tag.replace('PERSON_TAG_', ''): tag for tag in person_labels}

for term in corpus['term'].unique():
    temp_corpus = corpus[corpus['term'] == term]

    # ZBIERAMY WEKTORY
    person_vectors = {}
    for person in temp_corpus['kto'].unique():
        person_vectors[person + '_' + term] = model.dv[person_reverse_dict[person]]

