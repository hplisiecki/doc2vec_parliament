import pandas as pd
from gensim.utils import simple_preprocess
import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim import corpora
from collections import namedtuple
import logging
from tqdm import tqdm
import numpy as np
import os
# Create document embeddings

# word2vec = KeyedVectors.load('our_vectors.kv')

# load stopwords from txt file
with open('data/stopwords_pl.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().splitlines()

class corpusIterator(object):

    def __init__(self, corpus, bigram=None, trigram=None):
        if bigram:
            self.bigram = bigram
        else:
            self.bigram = None
        if trigram:
            self.trigram = trigram
        else:
            self.trigram = None
        self.corpus = corpus

    def __iter__(self):
        self.speeches = namedtuple('speeches', 'words tags')
        for row in self.corpus.iterrows():
            # if party is not none
            text = row[1].text
            gender = row[1].sex
            club = row[1].klub
            id = row[0]
            marriage = row[1]['marital-status']
            education = row[1]['education']
            person = row[1]['kto']

            doc_tag = 'DOC_TAG' + '_' + str(id)
            person_tag = 'PERSON_TAG' + '_' + str(person)
            gender_tag = 'GENDER_TAG' + '_' + str(gender)
            club_tag = 'CLUB_TAG' + '_' + str(club)
            marriage_tag = 'MARRIAGE_TAG' + '_' + str(marriage)
            education_tag = 'EDUCATION_TAG' + '_' + str(education)
            club_gender_tag = 'CLUB_GENDER_TAG' + '_' + str(club) + '_' + str(gender)

            tokens = simple_preprocess(text)
            if self.bigram and self.trigram:
                self.words = self.trigram[self.bigram[tokens]]
            elif self.bigram and not self.trigram:
                self.words = self.bigram[tokens]
            else:
                self.words = tokens
            self.tags = [doc_tag, gender_tag, club_tag, marriage_tag, education_tag, club_gender_tag, person_tag]
            yield self.speeches(self.words, self.tags)

class phraseIterator(object):

    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for row in tqdm(self.corpus.iterrows(), total=len(self.corpus)):
            # if party is not none
            text = row[1].text
            yield simple_preprocess(text)


if __name__=='__main__':

    ######### DO ZMIANY #########
    # LINIE 37 - 43 -> zmienić nazwy kolumn tak żeby odpowiadały tym w korpusie (w różnych krajach są różne nazwy klumn
    # LINIA 84 - 85 -> zmienić pod dany kraj (różny korpus, różny kraj)
    # LINIA 17 -> zmienić ścieżkę do stoppwords
    #############################

    save_path = 'data'
    country = 'pl' # ZMIENIĆ NA KRAJ KTÓRY JEST ANALIZOWANY
    corpus = pd.read_csv(r'D:\PycharmProjects\parliamentary_emotions\data/abortion_with_metrics') #### TUTAJ ŁADUJEMY KORPUS
    corpus = corpus.dropna(subset=['text'])
    corpus['text'] = corpus['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

    for term in corpus['term'].unique():
        temp_corpus = corpus[corpus['term'] == term]

        print('corpus loaded')
        ############################################################## PO PIERWSZYM URUCHOMIENIU NA KAŻDYM Z KRAJÓW ZAKOMENTOWAĆ PONIŻEJ
        phrases = Phrases(phraseIterator(temp_corpus))
        bigram = Phraser(phrases)
        print('bigram done')
        tphrases = Phrases(bigram[phraseIterator(temp_corpus)])
        trigram = Phraser(tphrases)
        print('trigram done')

        bigram.save(save_path + f'phraser_bigrams_{country}_{term}')
        trigram.save(save_path + f'phraser_trigrams_{country}_{term}')
        ############################################################## DOTĄD ZAKOMENTOWAĆ
        bigram = Phraser.load(save_path + f'phraser_bigrams_{country}_{term}')
        trigram = Phraser.load(save_path + f'phraser_trigrams_{country}_{term}')

        print('phraser loaded')

        model0 = Doc2Vec(vector_size=300, window=5, min_count=50, workers=8, epochs=5)

        # , bigram=bigram, trigram=trigram
        model0.build_vocab(corpusIterator(temp_corpus, bigram=bigram, trigram=trigram))
        print('vocab done')

        model0.train(corpusIterator(temp_corpus, bigram=bigram, trigram=trigram), total_examples=model0.corpus_count, epochs=model0.epochs)
        print('training done')

        # create savepath if not exists
        temp_save_path = os.path.join(save_path, country, str(term), 'doc2vec_0.model')
        if not os.path.exists(temp_save_path):
            os.makedirs(temp_save_path)

        model0.save(temp_save_path)
        print('model saved')