'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年02月22日 14:39
@Description: 
@URL: https://nbviewer.jupyter.org/github/erhwenkuo/deep-learning-with-keras-notebooks/blob/master/1.c-lstm-learn-alphabetic-seq.ipynb
@version: V1.0
'''
import  numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

# 給定隨機的種子, 以便讓大家跑起來的結果是相同的
numpy.random.seed(7)

# 定義序列數據集
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# 創建字符映射到整數（0 - 25)和反相的查詢字典物件
char_to_int = dict( (c, i) for i, c in enumerate( alphabet ) )
int_to_char = dict( (i, c) for i, c in enumerate( alphabet ) )

#準備訓練用資料
num_inputs = 1000
max_len = 5
dataX = []
dataY = []
for i in range( num_inputs ):
    start = numpy.random.randint( len( alphabet ) - 2 )
    end = numpy.random.randint( start, min( start + max_len, len( alphabet ) - 1))
    sequence_in = alphabet[ start : end+1 ]
    sequence_out = alphabet[ end+1 ]
    dataX.append( [ char_to_int[char] for char in sequence_in ] )
    dataY.append( [ char_to_int[ sequence_out ]] )
    print( sequence_in, '->', sequence_out )

# 資料預處理
# 將訓練資料轉換為陣列和並進行序列填充（如果需要）
X = pad_sequences( dataX, maxlen=max_len, dtype='float32')
# 重塑 X 資料的維度成為 (samples, time_steps, features)
X = numpy.reshape( X, ( X.shape[0], max_len, 1 ))  # <-- 特別注意這裡
# 歸一化
X = X / float( len( alphabet ) )
# 使用one hot encode 對Y值進行編碼
y = np_utils.to_categorical( dataY )

#建立模型
batch_size = 1
model = Sequential()
model.add( LSTM( 32, input_shape=( X.shape[1], 1 )))  # <-- 注意這裡
model.add( Dense( y.shape[1], activation='softmax'))
model.summary()

#定義訓練並進行訓練
model.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )
model.fit( X, y, epochs=10, batch_size=batch_size, verbose=2 )

# 評估模型的性能
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))

#預測結果
# 讓我們擷取1~5個字符轉成張量結構 shape:(1,5,1)來進行infer
for i in range(200):
    pattern_index = numpy.random.randint( len(dataX) )
    pattern = dataX[ pattern_index ]
    x = pad_sequences( [pattern], maxlen=max_len, dtype='float32' )
    x = numpy.reshape( x, (1, max_len, 1) )
    x = x / float( len(alphabet) )
    prediction = model.predict( x, verbose=0 )
    index = numpy.argmax( prediction )
    result = int_to_char[ index ]
    seq_in = [ int_to_char[value] for value in pattern ]
    print( seq_in, '->', result )
