'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年02月25日 12:15
@Description: 
@URL: 
@version: V1.0
'''
import os
import numpy as np

imdb_dir = 'F:\\workspace\\Tensorflow\\src\\deep-learning-with-keras-notebooks\\8-0-using-word-embeddings\\data\\aclImdb'
train_dir = os.path.join( imdb_dir, 'train' )

labels = []
texts = []

# 迭代檔案目錄來產生訓練資料
for label_type in ['neg', 'pos']:
    dir_name = os.path.join( train_dir, label_type )
    for fname in os.listdir( dir_name ):
        if fname[-4:] == '.txt':
            f = open( os.path.join( dir_name, fname ), encoding='utf8' )
            texts.append( f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

print( len( texts ) )
print( len( labels ) )

#對文本數據進行標記(tokenize)
#讓我們將我們收集的文本進行向量化，並準備一個訓練和驗證的分割。因為預先訓練的詞嵌入對
# 於只有很少的訓練數據可用的問題特別有用（否則，針對特定任務所訓練的詞嵌入表現可能會優於它們），
# 我們將添加以下內容：我們將訓練數據限制在前200樣本。因此，我們將在學習僅僅200個例子後，對電影評論進行分類。
from keras.preprocessing.text import  Tokenizer
from keras.preprocessing.sequence import pad_sequences
maxlen = 300    # 每一筆評論我們將裁減成100個字
training_samples = 20000  # 我們將使用200個樣本進行訓練
validation_samples = 1000 # 我們用10000個樣本來進行驗證
max_words = 88582         # 我們只考慮數據集中前10,000個單詞

tokenizer = Tokenizer( num_words=max_words )
tokenizer.fit_on_texts( texts )
sequences = tokenizer.texts_to_sequences( texts )

word_index = tokenizer.word_index
print( 'Found %s unique tokens.' % len( word_index ))

data = pad_sequences( sequences, maxlen=maxlen )

labels = np.asarray( labels )
print( 'Shape of data tensor:', data.shape )
print( 'Shape of label tensor:', labels.shape )

# 將數據分成訓練集和驗證集
# 但首先，進行數據順序洗牌
indices = np.arange( data.shape[0] )
np.random.shuffle( indices )
data = data[ indices ]
labels = labels[ indices ]

x_train = data[ :training_samples ]
y_train = labels[ :training_samples ]
x_val = data[ training_samples:training_samples+validation_samples ]
y_val = labels[ training_samples:training_samples+validation_samples ]

#單詞嵌入(word embeddings)預處理
glove_dir = 'F:\\workspace\\Tensorflow\\src\\deep-learning-with-keras-notebooks\\8-0-using-word-embeddings\\model\\glove'
embeddings_index = {}
f = open( os.path.join( glove_dir, 'glove.6B.100d.txt'), encoding='utf8' )

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asanyarray( values[1:], dtype='float32' )
    embeddings_index[ word ] = coefs
f.close()

print( 'Found %s word vectors' % len( embeddings_index ))

#建立一個嵌入矩陣，我們將能夠加載進到一個Keras的Embedding層。
#numpy的2D矩陣(max_words, embedding_dim)，其中每個條目i包含我
# 們引用詞索引（在標記化期間構建）中索引'i'的單詞的嵌入_dim維向量
embeddings_dim = 100 # 代表每一個單詞會用一個100維的向量來表示

embeddings_matrix = np.zeros( ( max_words, embeddings_dim ) )
for word, i in word_index.items():
    embeddings_vector = embeddings_index.get( word )
    if i < max_words:
        if embeddings_vector is not None:
            embeddings_matrix[i] = embeddings_vector

#定義一個模型
from keras.models import Sequential
from keras.layers import  Embedding, Flatten, Dense

model = Sequential()
model.add( Embedding( max_words, embeddings_dim, input_length=maxlen ))
model.add( Flatten() )
model.add( Dense(32, activation='relu' ) )
model.add( Dense( 1, activation='sigmoid' ) )
model.summary()

#嵌入層具有單一權重矩陣：2D浮點數矩陣，其中每個條目“i”是與索引“i”
# 相關聯的單詞向量。讓我們將我們準備好的GloVe矩陣加載到我們的模型中的
# Embedding層
model.layers[0].set_weights( [embeddings_matrix] )
model.layers[0].trainable = False

model.compile( optimizer='rmsprop', loss='binary_crossentropy',
               metrics=['acc'] )
model.fit( x_train, y_train,
           epochs=100,
           batch_size=32,
           validation_data=(x_val, y_val))
