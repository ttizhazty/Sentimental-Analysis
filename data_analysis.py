import pickle
import os
import scipy
from wordcloud import WordCloud
from collections import Counter
from matplotlib import pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize

'''
train_path = './yelp_dataset/data_split/train/emb_label/'
test_path = './yelp_dataset/data_split/test/emb_label/'

train_files = os.listdir(train_path)
train_label_list = []
for file in train_files:
    with open(train_path + file, 'rb') as f:
        train_label_list += pickle.load(f)
temp = [item.index(1) for item in train_label_list]
train_distrb = Counter(temp)


test_files = os.listdir(test_path)
test_label_list = []
for file in test_files:
    with open(test_path + file, 'rb') as f:
        test_label_list += pickle.load(f)
temp = [item.index(1) for item in test_label_list]
test_distrb = Counter(temp)

print('train_label: ', train_distrb)
print('test_label：', test_distrb)

'''
positive_text_path = './yelp_dataset/positive_text.pkl'
neutral_text_path = './yelp_dataset/neutral_text.pkl'
negative_text_path = 'yelp_dataset/negative_text.pkl'

with open(positive_text_path, 'rb') as f:
    positive_text = pickle.load(f)
text = ''
for item in positive_text:
    text += item[1]
wordcloud = WordCloud(background_color='white', width=1000, height=860, margin=2, collocations=False).generate(text)
plt.imshow(wordcloud)
plt.axis('off')
wordcloud.to_file('./plots/positive_text.png')



with open(neutral_text_path, 'rb') as f:
    positive_text = pickle.load(f)
text = ''
for item in positive_text:
    text += item[1]
wordcloud = WordCloud(background_color='white', width=1000, height=860, margin=2, collocations=False).generate(text)
plt.imshow(wordcloud)
plt.axis('off')
wordcloud.to_file('./plots/neutral_text.png')

with open(negative_text_path, 'rb') as f:
    positive_text = pickle.load(f)
text = ''
for item in positive_text:
    text += item[1]
wordcloud = WordCloud(background_color='white', width=1000, height=860, margin=2, collocations=False).generate(text)
plt.imshow(wordcloud)
plt.axis('off')
wordcloud.to_file('./plots/negative_text.png')

## plot one star......
with open('./yelp_dataset/label2text.pkl', 'rb') as f:
    text = pickle.load(f)[1]
tmp = ''
for item in text:
    tmp += item
wordcloud = WordCloud(background_color='white', width=1000, height=860, margin=2, collocations=False).generate(tmp)
plt.imshow(wordcloud)
plt.axis('off')
wordcloud.to_file('./plots/onestar_text.png')

with open('./yelp_dataset/label2text.pkl', 'rb') as f:
    text = pickle.load(f)[2]
tmp = ''
for item in text:
    tmp += item
wordcloud = WordCloud(background_color='white', width=1000, height=860, margin=2, collocations=False).generate(tmp)
plt.imshow(wordcloud)
plt.axis('off')
wordcloud.to_file('./plots/twostar_text.png')

with open('./yelp_dataset/label2text.pkl', 'rb') as f:
    text = pickle.load(f)[3]
tmp = ''
for item in text:
    tmp += item
wordcloud = WordCloud(background_color='white', width=1000, height=860, margin=2, collocations=False).generate(tmp)
plt.imshow(wordcloud)
plt.axis('off')
wordcloud.to_file('./plots/threestar_text.png')

with open('./yelp_dataset/label2text.pkl', 'rb') as f:
    text = pickle.load(f)[4]
tmp = ''
for item in text:
    tmp += item
wordcloud = WordCloud(background_color='white', width=1000, height=860, margin=2, collocations=False).generate(tmp)
plt.imshow(wordcloud)
plt.axis('off')
wordcloud.to_file('./plots/fourstar_text.png')
