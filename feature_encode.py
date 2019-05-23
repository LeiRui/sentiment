#coding:utf-8
import numpy as np
import pandas as pd
import os
import re
import string

# emoji
# https://gist.github.com/brendano/25521552453909400e2310b04f1b2ac9
JUNK_RE = (
    u'[' +
    # Supplemental Multilingual Plane
    u'\U00010000-\U0001ffff' +

    # The weird extra planes
    u'\U00030000-\U0010ffff' +

    # E000–EFFF private use area,since I noticed \ue056 (doesnt display for me)
    u'\U0000e000-\U0000efff' +

    # There's a bunch of symbol blocks in the BMP here:
    # https://en.wikibooks.org/wiki/Unicode/Character_reference/2000-2FFF
    # Box Drawing
    # Box Elements
    # Miscellaneous Symbols
    # Dingbats
    # Miscellaneous Mathematical Symbols-A
    # Supplemental Arrows-A
    # Braille Patterns
    # Supplemental Arrows-B
    # Miscellaneous Mathematical Symbols-B
    # Supplemental Mathematical Operators
    # Miscellaneous Symbols and Arrows
    # e.g. \ue056  ✌

    u'\U00002500-\U00002bff' +

    # zero-width space, joiner, nonjoiner .. ZW Joiner is mentioned on Emoji wikipedia page
    # omg the ZWJ examples are downright non-heteronormative http://www.unicode.org/emoji/charts/emoji-zwj-sequences.html
    u'\U0000200B-\U0000200D' +

    # http://unicode.org/reports/tr51/
    # Certain emoji have defined variation sequences, where an emoji character can be followed by one of two invisible emoji variation selectors:
    # U+FE0E for a text presentation
    # U+FE0F for an emoji presentation
    u'\U0000fe0e-\U0000fe0f' +


    # https://www.charbase.com/2026-unicode-horizontal-ellipsis 水平省略号
    u'\u2026' +

    # " 这个双引号可分可不分 "good"和good有时未必同一个意思 这里我还是分 因为我发现也有很多句子开头"blabla
    # 而那种真正一个单词双引号起来的词，不加双引号直接出现的概率也很小，所以分了
    u'\u201c' +

    # "
    u'\u201d' +

    u']+')

punc = string.punctuation
punc = punc.replace("-", "") # don't remove hyphens
print(punc)

# Add optional whitespace. Because we want
# 1. A symbol surrounded by nonwhitespace => change to whitespace
# 2. A symbol surrounded by whitespace => no extra whitespace
# the current rule is too aggressive: also collapses pre-existing whitespace.
# this is ok for certain applications including ours.
# SUB_RE = re.compile( r'\s*' + JUNK_RE + r'\s*', re.UNICODE)
SUB_RE = re.compile(r'('+JUNK_RE+r')', re.UNICODE)
# SUB_RE = re.compile(re.compile(u'[\U00010000-\U0010ffff]'))

# 在标点符号、表情和正文文本之间加空格隔开，并且聚集在一起的标点符号不会隔开，比如:)
def add_space_to_punc_and_emoji(text):
  text = re.sub( r'([a-zA-Z])(['+punc+'])', r'\1 \2',text)
  text = re.sub(r'(?='+JUNK_RE+r')', r" ", text)
  text = text[::-1]
  text = re.sub(r'([a-zA-Z])(['+punc+'])', r'\1 \2',text)
  text = re.sub(r'(?='+JUNK_RE+r')', r" ", text)
  text = text[::-1]
  return text

# # a="😂my version of dieting😁😂"
# # a="5974,Tuu bhii khaaabeeess hai 🔥,Negative"
# # a="In ki shahkar tasneef �EEE€�EEE Shahnama-e-Islam �EEE€�EEE ne inhe maqbooliya"
a="5289,Tu aa to sae dekh kia kia sa??? pilati😂😎😋sss:)ss.… “aa”,Negative'"
b=add_space_to_punc_and_emoji(a)
print(b)
#from string import punctuation
#all_text = ''.join([c for c in b if c not in punctuation])
#print(all_text)


train_df = pd.read_csv('data/train.csv',encoding='utf-8')
test_df = pd.read_csv('data/20190520_test.csv',encoding='utf-8')
# print(train_df.head())

print(train_df[train_df.isnull().values==True]) # find anomaly rows and fix by hand
# print(test_df.isnull().any())

## create a list of words
all_text = ''
for index, row in train_df.iterrows():
  review = row['review'].lower()

  # this is important
  ## method 1: 输出没有标点符号的句子
  # review_processed = ''.join([c for c in review if c not in punctuation])
  # method 2: 输出标点符号、标点符号表情、emoji和文本用空格隔开的句子，这样之后的words中就有
  review_processed = add_space_to_punc_and_emoji(review)

  all_text = all_text + ' ' + review_processed # add ' ' for split
  train_df.iloc[index,train_df.columns.get_loc('review')] = review_processed
words = all_text.split()

## encoding the words
from collections import Counter
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
unknown = '<UNK>'
vocab.append(unknown)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
print(vocab_to_int)

import json
json = json.dumps(vocab_to_int)
f = open("wordDict.json","w")
f.write(json)
f.close()

## use the dict to tokenize each review in reviews_split
train_reviews_ints = []
for review in train_df['review']:
  train_reviews_ints.append([vocab_to_int[word] for word in review.split()])
print(train_reviews_ints[:1])

## encode the labels
train_labels = np.array([1 if label.lower() == 'positive' else 0 for label in train_df['label']])
print(train_labels)

## prepare test set in the same way
test_id = list(test_df['ID'])
test_reviews_ints = []
for index, row in test_df.iterrows():
  review = row['review'].lower()

  # review_processed = ''.join([c for c in review if c not in punctuation])
  review_processed = add_space_to_punc_and_emoji(review)

  test_df.iloc[index,test_df.columns.get_loc('review')] = review_processed

  review_ints = []
  for word in review_processed.split():
    if word not in vocab_to_int:
      word = unknown
    review_ints.append(vocab_to_int[word])

  test_reviews_ints.append(review_ints)

#print(test_id)
print(test_reviews_ints[:1])

## Now we have train_reviews_ints, train_labels, test_id, test_reviews_ints
train_reviews_ints_array = np.array(train_reviews_ints)
np.save('train_reviews_ints.npy',train_reviews_ints_array)
np.save('train_labels.npy',train_labels)
test_id_array = np.array(test_id)
np.save('test_id.npy',test_id_array)
test_reviews_ints_array = np.array(test_reviews_ints)
np.save('test_reviews_ints.npy',test_reviews_ints_array)

review_lens = Counter([len(x) for x in train_reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))

