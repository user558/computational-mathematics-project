import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
from sklearn.datasets import fetch_20newsgroups
from sklearn import decomposition
from scipy import linalg
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import umap.umap_ as umap


#Загрузим данные из Usenet(Новости по 20-ти темам)
#//scikit-learn.org/0.19/datasets/twenty_newsgroups.html
categories = ['sci.electronics' , 'alt.atheism','comp.graphics', 'sci.space']
remove=('headers', 'footers', 'quotes')
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=remove)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=remove)
#newsgroups_train.filenames.shape, newsgroups_train.target.shape

#Стоп слова(общие термины)
from sklearn.feature_extraction import stop_words
sorted(list(stop_words.ENGLISH_STOP_WORDS))[:20]

#Предварительная обработка данных
#Разделим текст на предложения, а предложения на слова. Сделаем все буквы маленькими и удалим знаки препинания(Лексемезация)
#Слова, содержащие менее 3 символов, удаляются
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        SYMBOLS_TO_KEEP = re.compile('[^A-Za-z0-9]+')
        doc = re.sub(SYMBOLS_TO_KEEP," ",doc)
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if (len(t) >3) & (t not in stop_words.ENGLISH_STOP_WORDS)]
        
#Векторизуем
vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(),lowercase=True,max_df = 0.9,min_df =5)
vectors = vectorizer.fit_transform(newsgroups_train.data).todense()
#vectors.shape
vocab = np.array(vectorizer.get_feature_names())
print(vocab.shape)
vocab[3000:3050]


#Запустим full SVD(разложение написать самим)
U, s, Vh = linalg.svd(vectors, full_matrices=True)
print(U.shape, s.shape, Vh.shape, vocab.shape)
plt.plot(s)

#Визуализация для full svd
embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(U[:,:10])
plt.figure(figsize=(7,5))
plt.scatter(embedding[:, 0], embedding[:, 1],c = newsgroups_train.target,s = 10)
print(newsgroups_train.target.shape)
plt.show()

#Запустим reduced SVD(разложение написать самим)
U1, s1, Vh1 = linalg.svd(vectors, full_matrices=False)
print(U.shape, s.shape, Vh.shape, vocab.shape)
plt.plot(s)

#Теперь мы можем получить 5 самых популярных тем (ограничение:они содержат только топ-7 популярных слов)
#num_top_words=7
#def show_topics(a):
#    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
#    topic_words = ([top_words(t) for t in a])
#   return [' '.join(t) for t in topic_words]
#show_topics(Vh[:5])


#Визуализация для reduced SVD
embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(U1[:,:10])
plt.figure(figsize=(7,5))
plt.scatter(embedding[:, 0], embedding[:, 1],c = newsgroups_train.target,s = 10)
print(newsgroups_train.target.shape)
plt.show()

#Запустим randomized SVD
from sklearn import decomposition
%time u_rand, s_rand, v_rand = decomposition.randomized_svd(vectors, 10)
print(u_rand.shape,s_rand.shape,v_rand.shape)
print()

#Визуализация для randomized SVD
import umap.umap_ as umap
embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(u_rand)
plt.figure(figsize=(7,5))
plt.scatter(embedding[:, 0], embedding[:, 1], c = newsgroups_train.target,s = 10 )
print(newsgroups_train.target.shape)
plt.show()
