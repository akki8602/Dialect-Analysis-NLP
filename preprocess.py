import pandas as pd
import re
import nltk
import ssl
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models import Word2Vec

# Noise removal
def  clean_messages_regex(message):
     message = re.sub(r'[^A-Za-z1-9\w\s]', '', message, flags=re.UNICODE)
     return message

cleaned_messages = []
with open('messages.txt', 'r') as file:
     messages = file.readlines()
     cleaned_messages = [clean_messages_regex(message) for message in messages]
# print(cleaned_messages)

# with open('cleaned_messages.txt', 'w', encoding='utf-8') as file:
#      for message in cleaned_messages:
#           file.write(message)
#           # print(message)

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
tokenized_messages = []
with open('cleaned_messages.txt', 'r', encoding='utf-8') as file:
    messages = file.readlines()
    for message in messages:
        tokenized_messages.append(word_tokenize(message))

# print(tokenized_messages)

stop_words = set(stopwords.words('english'))
cleaned_messages = [
    [word.lower() for word in message if word.lower() not in stop_words]
    for message in tokenized_messages
]
# print(cleaned_messages)
cleaned_messages = [' '.join(message) for message in cleaned_messages]
tagged_messages = [nltk.pos_tag(word_tokenize(message)) for message in cleaned_messages]
# for i, tagged_message in enumerate(tagged_messages):
#     print(f"Message {i+1}: {tagged_message}")

def get_pos_tag(word, tagged_messages):
    word = word.lower()
    pos_tags = []
    for tagged_message in tagged_messages:
        for token, pos in tagged_message:
            if token.lower() == word:
                pos_tags.append(pos)
    return pos_tags
pos_tag_for_word = get_pos_tag('lah', tagged_messages)
print(f"POS tag for lah': {pos_tag_for_word}")
pos_tag_for_word = get_pos_tag('leh', tagged_messages)
print(f"POS tag for leh': {pos_tag_for_word}")
pos_tag_for_word = get_pos_tag('lor', tagged_messages)
print(f"POS tag for lor': {pos_tag_for_word}")

def generate_ngrams(data, n):
    ngrams_list = []
    for tokens in data:
        ngrams_list.extend(list(ngrams(tokens, n)))
    return ngrams_list

n = 3
ngrams_list = generate_ngrams(cleaned_messages, n)

def filter_ngrams(ngrams_list, keyword):
    filtered_ngrams = [ngram for ngram in ngrams_list if keyword in ngram]
    ngram_freq = Counter(filtered_ngrams)
    print(f"Most common n-grams with {keyword}: ", ngram_freq.most_common(20), "\n")
    # return filtered_ngrams

filter_ngrams(ngrams_list, 'lah')
filter_ngrams(ngrams_list, 'la')
filter_ngrams(ngrams_list, 'leh')
filter_ngrams(ngrams_list, 'le')
filter_ngrams(ngrams_list, 'lor')
filter_ngrams(ngrams_list, 'liao')  

#Word2Vec analysis
word2vec_model = Word2Vec(sentences=cleaned_messages, vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.save("word2vec_model.model")
word2vec_model = Word2Vec.load("word2vec_model.model")

# similar_words = word2vec_model.wv.most_similar("lah", topn=10)
# print("Words similar to 'lah':", similar_words)
# similar_words = word2vec_model.wv.most_similar("lor", topn=10)
# print("Words similar to 'lor':", similar_words)
# similar_words = word2vec_model.wv.most_similar("leh", topn=10)
# print("Words similar to 'leh':", similar_words)
# similar_words = word2vec_model.wv.most_similar("la", topn=10)
# print("Words similar to 'la':", similar_words)
# similar_words = word2vec_model.wv.most_similar("le", topn=10)
# print("Words similar to 'le':", similar_words)
# similar_words = word2vec_model.wv.most_similar("liao", topn=10)
# print("Words similar to 'liao':", similar_words)


word_pairs = [('lah', 'la'), ('lah', 'lor'), ('lah', 'le'), ('lah', 'leh'), ('lah', 'liao'), ('lah', 'must'), ('le', 'leh')]
for word1, word2 in word_pairs:
    similarity = word2vec_model.wv.similarity(word1, word2)
    print(f"Similarity between {word1} and {word2}: {similarity:.4f}")



#LDA analysis
# cleaned_message_words = [' '.join(message) for message in cleaned_messages]
# # clean_message_words = [word for message in cleaned_messages for word in message]
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(cleaned_message_words)

# lda = LatentDirichletAllocation(n_components=3, random_state=42)
# lda.fit(X)

# def display_topics(model, feature_names, no_top_words):
#     for topic_idx, topic in enumerate(model.components_):
#         print(f"Topic {topic_idx}:")
#         print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

# no_top_words = 10
# display_topics(lda, vectorizer.get_feature_names_out(), no_top_words)

# def topics_containing_word(lda, vectorizer, word):
#     word_index = vectorizer.vocabulary_.get(word)
#     topics = []
#     for idx, topic in enumerate(lda.components_):
#         if word_index in topic.argsort()[:-11:-1]:
#             topics.append((idx, [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]]))
#     return topics

# lah_topics = topics_containing_word(lda, vectorizer, 'lah')
# for topic in lah_topics:
#     print(f"Topic {topic[0]}: {topic[1]}")








# tokenized_words = [word for message in tokenized_messages for word in message]

# tokenized_words = [word.lower() for word in tokenized_words]
# # print(tokenized_words)
# # print(len(tokenized_words))

# stop_words = set(stopwords.words('english'))
# custom_stopwords = set(['im', 'u', 'go', 'ok', 'haha', 'lol', 'hahaha', 'love', 'wat', 'help', 'as', 'guy', 'right', 'maybe', 'well', 'dear', 'help', 'n', 'dun', 'k', 'tmr', 'take', 'sure', '4', 'cant', 'say', 'first', 'eat', 'let', 'send', 'still', 'today', 'really', 'free', 'tell', 'ur', 'hey', 'think', 'come', 'know', 'call', 'day', 'back', 'wan', 'sorry', 'thanks', 'already', 'got', 'like', 'okay', 'dont', 'time', 'get', 'need', 'want', 'home', 'good', 'meet', 'also', 'ill', 'later', 'going', 'see', 'yeah', 'oh', 'hi', 'ya', '2', 'one' ])
# stop_words = stop_words.union(custom_stopwords)
# clean_words = [word for word in tokenized_words if word not in stop_words]
# # print(clean_words)

# stemmer = PorterStemmer()
# lemmatizer = WordNetLemmatizer()

# # stemmed = [stemmer.stem(word) for word in clean_words]
# lemmatized = [lemmatizer.lemmatize(token) for token in clean_words]
# # print(stemmed)
# # print(lemmatized)
# # print(len(lemmatized))

# # def dummy_tokenizer(tokens):
# #     return tokens

# # bow_vectorizer = CountVectorizer(stop_words='english')
# # vectorized_data = bow_vectorizer.fit_transform(cleaned_messages)

# # print(bow_vectorizer.get_feature_names_out())
# # print(vectorized_data.toarray())

# bag_of_words = Counter(lemmatized)
# most_common_ten = bag_of_words.most_common(100)

# # print(most_common_ten)

# # token = 'la'
# # token_frequency = bag_of_words[token]
# # print(token_frequency)

# word_freq = dict(bag_of_words)

# wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()