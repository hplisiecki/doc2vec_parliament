from gensim.models.doc2vec import Doc2Vec
import pandas as pd
import os

save_path = r'D:\GitHub\doc2vec_parliament\doc2vec_parliament\doc2vec\data\sk'

corpus = pd.read_csv(r'D:\PycharmProjects\data\newest_debates\corpus_sk.csv') #### INNY KORPUS

corpus['kto'] = corpus['speaker_name'] + ' ' + corpus['speaker_surname']
person_labels = ['PERSON_TAG' + '_' + str(row[1]['kto']) for row in corpus.iterrows()]
person_dict = {tag: tag.replace('PERSON_TAG_', '') for tag in person_labels}
person_reverse_dict = {tag.replace('PERSON_TAG_', ''): tag for tag in person_labels}

person_tags = [tag for tag in model.dv.index_to_key if 'PERSON_TAG' in tag]

person_vectors = {}
for term in corpus['term'].unique():
    model = Doc2Vec.load(os.path.join(save_path, str(term), 'doc2vec_0.model'))

    temp_corpus = corpus[corpus['term'] == term]

    # ZBIERAMY WEKTORY
    for person in temp_corpus['kto'].unique():
        # if person is string
        if type(person) == str:
            person_vectors[person + '_' + str(term)] = model.dv[person_reverse_dict[person]]

