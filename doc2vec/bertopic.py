import pandas as pd
from bertopic import BERTopic

country = 'pl'

corpus = pd.read_csv(rf'D:\PycharmProjects\data\{country}\corpus_{country}.csv')

topic_model = BERTopic(language= 'multilingual', calculate_probabilities=True, verbose=True, nr_topics=15, min_topic_size = 50)

topics, probs = topic_model.fit_transform(corpus['text'])

# get topics
topics = topic_model.get_topics()

# save model
topic_model.save(rf'D:\PycharmProjects\coherence\multilingual\models\{country}_corpus_topic_model')

# save topics and probabilities
topics_and_probs = pd.DataFrame({'topics': topics})
for i in range(14):
    topics_and_probs[f'prob_{i}'] = [prob[i] for prob in probs]

topics_and_probs['prob_minus_1'] = [prob[-1] for prob in probs]

topics_and_probs.to_csv(rf'D:\PycharmProjects\coherence\multilingual\data\{country}_corpus_topics_and_probs.csv')