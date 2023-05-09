import pandas as pd
import ast
import numpy as np


# HUNGARIAN
hungary = pd.read_csv(r'D:\GitHub\doc2vec_parliament\doc2vec_parliament\doc2vec\data\hungary.csv')
hungary['vector'] = hungary['vector'].apply(lambda x: x.replace('\n', ' '))
hungary['vector'] = hungary['vector'].apply(lambda x: x.replace('  ', ' '))
hungary['vector'] = hungary['vector'].apply(lambda x: x.replace('  ', ' '))
hungary['vector'] = hungary['vector'].apply(lambda x: x.replace('  ', ' '))
hungary['vector'] = hungary['vector'].apply(lambda x: x.replace('  ', ' '))
hungary['vector'] = hungary['vector'].apply(lambda x: x.replace('[ ', '['))
hungary['vector'] = hungary['vector'].apply(lambda x: x.replace(' ]', ']'))
hungary['vector'] = hungary['vector'].apply(lambda x: x.replace(' ', ', '))

hungary['embeddings'] = hungary['vector'].apply(lambda x: ast.literal_eval(x))
names = hungary['person'].unique()
vectors = []
for name in names:
    temp = hungary[hungary['person'] == name]
    array = np.array(temp['embeddings'].tolist())
    mean = np.mean(array, axis=0)
    vectors.append(mean)

hungary_vectors = pd.DataFrame({'person': names, 'vector': vectors})
# split the list into columns
hungary_vectors = pd.concat([hungary_vectors['person'], hungary_vectors['vector'].apply(pd.Series)], axis=1)

poland = pd.read_csv(r'D:\GitHub\doc2vec_parliament\doc2vec_parliament\doc2vec\data\polska.csv')

poland['vector'] = poland['vector'].apply(lambda x: x.replace('\n', ' '))
poland['vector'] = poland['vector'].apply(lambda x: x.replace('  ', ' '))
poland['vector'] = poland['vector'].apply(lambda x: x.replace('  ', ' '))
poland['vector'] = poland['vector'].apply(lambda x: x.replace('  ', ' '))
poland['vector'] = poland['vector'].apply(lambda x: x.replace('  ', ' '))
poland['vector'] = poland['vector'].apply(lambda x: x.replace('[ ', '['))
poland['vector'] = poland['vector'].apply(lambda x: x.replace(' ]', ']'))
poland['vector'] = poland['vector'].apply(lambda x: x.replace(' ', ', '))

poland['embeddings'] = poland['vector'].apply(lambda x: ast.literal_eval(x))
names = poland['person'].unique()
vectors = []
for name in names:
    temp = poland[poland['person'] == name]
    array = np.array(temp['embeddings'].tolist())
    mean = np.mean(array, axis=0)
    vectors.append(mean)

poland_vectors = pd.DataFrame({'person': names, 'vector': vectors})

# split the list into columns
poland_vectors = pd.concat([poland_vectors['person'], poland_vectors['vector'].apply(pd.Series)], axis=1)

# SLOVAKIA
slovakia = pd.read_csv(r'D:\GitHub\doc2vec_parliament\doc2vec_parliament\doc2vec\data\slovakia.csv')

slovakia['vector'] = slovakia['vector'].apply(lambda x: x.replace('\n', ' '))
slovakia['vector'] = slovakia['vector'].apply(lambda x: x.replace('  ', ' '))
slovakia['vector'] = slovakia['vector'].apply(lambda x: x.replace('  ', ' '))
slovakia['vector'] = slovakia['vector'].apply(lambda x: x.replace('  ', ' '))
slovakia['vector'] = slovakia['vector'].apply(lambda x: x.replace('  ', ' '))
slovakia['vector'] = slovakia['vector'].apply(lambda x: x.replace('[ ', '['))
slovakia['vector'] = slovakia['vector'].apply(lambda x: x.replace(' ]', ']'))
slovakia['vector'] = slovakia['vector'].apply(lambda x: x.replace(' ', ', '))

slovakia['embeddings'] = slovakia['vector'].apply(lambda x: ast.literal_eval(x))
names = slovakia['person'].unique()
vectors = []
for name in names:
    temp = slovakia[slovakia['person'] == name]
    array = np.array(temp['embeddings'].tolist())
    mean = np.mean(array, axis=0)
    vectors.append(mean)

slovakia_vectors = pd.DataFrame({'person': names, 'vector': vectors})
# split the list into columns
slovakia_vectors = pd.concat([slovakia_vectors['person'], slovakia_vectors['vector'].apply(pd.Series)], axis=1)

czech = pd.read_csv(r'D:\GitHub\doc2vec_parliament\doc2vec_parliament\doc2vec\data\czech.csv')

czech['vector'] = czech['vector'].apply(lambda x: x.replace('\n', ' '))
czech['vector'] = czech['vector'].apply(lambda x: x.replace('  ', ' '))
czech['vector'] = czech['vector'].apply(lambda x: x.replace('  ', ' '))
czech['vector'] = czech['vector'].apply(lambda x: x.replace('  ', ' '))
czech['vector'] = czech['vector'].apply(lambda x: x.replace('  ', ' '))
czech['vector'] = czech['vector'].apply(lambda x: x.replace('[ ', '['))
czech['vector'] = czech['vector'].apply(lambda x: x.replace(' ]', ']'))
czech['vector'] = czech['vector'].apply(lambda x: x.replace(' ', ', '))

czech['embeddings'] = czech['vector'].apply(lambda x: ast.literal_eval(x))
names = czech['person'].unique()
vectors = []
for name in names:
    temp = czech[czech['person'] == name]
    array = np.array(temp['embeddings'].tolist())
    mean = np.mean(array, axis=0)
    vectors.append(mean)

czech_vectors = pd.DataFrame({'person': names, 'vector': vectors})

# split the list into columns
czech_vectors = pd.concat([czech_vectors['person'], czech_vectors['vector'].apply(pd.Series)], axis=1)

# save all
hungary_vectors.to_csv(r'D:\GitHub\doc2vec_parliament\doc2vec_parliament\doc2vec\data\hungary_vectors.csv', index=False)
poland_vectors.to_csv(r'D:\GitHub\doc2vec_parliament\doc2vec_parliament\doc2vec\data\poland_vectors.csv', index=False)
slovakia_vectors.to_csv(r'D:\GitHub\doc2vec_parliament\doc2vec_parliament\doc2vec\data\slovakia_vectors.csv', index=False)
czech_vectors.to_csv(r'D:\GitHub\doc2vec_parliament\doc2vec_parliament\doc2vec\data\czech_vectors.csv', index=False)