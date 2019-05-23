import numpy as np
from collections import Counter

train_reviews_ints = np.load("train_reviews_ints.npy",allow_pickle=True).tolist()
train_labels = np.load("train_labels.npy")
test_reviews_ints = np.load("test_reviews_ints.npy",allow_pickle=True).tolist()
test_id = np.load("test_id.npy").tolist()
print("load success")

for i in range(len(test_id)):
  if i==0:
    continue
  if test_id[i]-test_id[i-1]!=1:
    print(test_id[i])

## removing outliers
# outlier review stats
review_lens = Counter([len(x) for x in train_reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0])) # 0个长度为0的
print("Maximum review length: {}".format(max(review_lens))) # 最大长度359
print("Average review length: {}".format(sum(review_lens)/len(review_lens)))

## padding sequences
def pad_features(reviews_ints, seq_length):
  ''' Return features of review_ints, where each review is padded with 0's
      or truncated to the input seq_length.
  '''

  # getting the correct rows x cols shape
  features = np.zeros((len(reviews_ints), seq_length), dtype=int)

  # for each review, I grab that review and
  for i, row in enumerate(reviews_ints):
    features[i, -len(row):] = np.array(row)[:seq_length] #这一句话就涵盖了所有情况
    # len(features[i]) = seq_length
    # 如果len(row) <= seq_length, (已经排除了row长度为0的情况），就是左边补0处理，np.array(row)[:seq_length]还是row自身长度
    # 如果len(row) > seq_length, 就是截取前seq_length个处理，features[i, -len(row):]就是features[i]全部，因features[i]最大长度也就是seq_length

  return features

# Test your implementation!
seq_length = 50
train_features = pad_features(train_reviews_ints, seq_length=seq_length)
test_features = pad_features(test_reviews_ints, seq_length=seq_length)
## test statements - do not change - ##
assert len(train_features)==len(train_reviews_ints), "Your features should have as many rows as reviews."
assert len(train_features[0])==seq_length, "Each feature row should contain seq_length values."
# print first 10 values of the first 30 batches
print(train_features[:30,-10:])

np.save("train_features.npy", train_features)
np.save("test_features.npy", test_features)