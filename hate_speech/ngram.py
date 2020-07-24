from nsml import DATASET_PATH
from data import HateSpeech
from torchtext.data import Iterator
import operator

TRAIN_DATA_PATH = '{}/train/train_data'.format(DATASET_PATH[0])
UNLABELED_DATA_PATH = '{}/train/raw.json'.format(DATASET_PATH[1])
N_GRAM_SIZE = 3

task = HateSpeech(TRAIN_DATA_PATH)

vocab_size = task.max_vocab_indexes['syllable_contents']

ds_iter = Iterator(task.datasets[0], batch_size=1, repeat=False, shuffle=False, train=False,
                   sort_key=lambda x: -len(x.syllable_contents))
ds_iter.init_epoch()

recent_syllables = []
n_gram_cnt = {}

for i, batch in enumerate(ds_iter):
    syllable_list = batch.syllable_contents.squeeze().tolist()

    for syllable_idx in range(len(syllable_list)-N_GRAM_SIZE+1):
        n_gram = tuple(syllable_list[syllable_idx:syllable_idx+N_GRAM_SIZE])
        n_gram_cnt[n_gram] = n_gram_cnt[n_gram] + 1 if n_gram in n_gram_cnt else 1

print(f'# of {N_GRAM_SIZE}-grams: {len(n_gram_cnt)}')

k = 20

# most frequent k n-grams
print('Most frequent n-grams')
frequent_n_grams = sorted(n_gram_cnt, key=n_gram_cnt.get, reverse=True)[:k]
for n_gram in frequent_n_grams:
    print(n_gram, n_gram_cnt[n_gram])

# most rare k n-grams
print('Most rare n-grams')
rare_n_grams = sorted(n_gram_cnt, key=n_gram_cnt.get, reverse=False)[:k]
for n_gram in rare_n_grams:
    print(n_gram, n_gram_cnt[n_gram])
