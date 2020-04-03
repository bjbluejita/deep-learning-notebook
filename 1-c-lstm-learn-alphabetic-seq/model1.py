'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年02月22日 10:54
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

#準備資料
# 定義序列數據集
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# 創建字符映射到整數（0 - 25)和反相的查詢字典物件
char_to_int = dict( (c, i) for i, c in enumerate( alphabet ) )
int_to_char = dict( (i, c) for i, c in enumerate( alphabet ) )
print( "字母對應到數字編號: \n", char_to_int)
print( "數字編號對應到字母: \n", int_to_char)

#準備訓練用資料
#創建我們的輸入(X)和輸出(y)來訓練我們的神經網絡
#通過定義一個輸入序列長度，然後從輸入字母序列中讀取序列
#輸入長度1.從原始輸入數據的開頭開始，我們可以讀取第一個字母“A”，下一個字母作為預測“B”
seq_length = 3
dataX = []
dataY = []
for i in range( 0, len(alphabet) - seq_length, 1 ):
    seq_in = alphabet[ i : i + seq_length ]
    seq_out = alphabet[ i + seq_length ]
    dataX.append( [ char_to_int[char] for char in seq_in ] )
    dataY.append( [ char_to_int[seq_out] ] )
    print( seq_in, '->', seq_out )

#NumPy數組重塑為LSTM網絡所期望的格式，也就是: (samples, time_steps, features)。
# 同時我們將進行資料的歸一化(normalize)來讓資料的值落於0到1之間。並對標籤值進行one-hot的編碼
# 重塑 X 資料的維度成為 (samples, time_steps, features)
X = numpy.reshape( dataX, ( len(dataX), 1,seq_length ))

# 歸一化
X = X / float( len(alphabet) )

# one-hot 編碼輸出變量
y = np_utils.to_categorical( dataY )

print( 'X shape:', X.shape )
print( 'y shape:', y.shape )

#建立模型
model = Sequential()
model.add( LSTM( 32, input_shape=( X.shape[1], X.shape[2] ))) # <-- 特別注意這裡
model.add( Dense( y.shape[1], activation='softmax' ) )
model.summary()

#定義訓練並進行訓練
model.compile( loss='categorical_crossentropy', optimizer='adam', metrics=[ 'accuracy' ] )
model.fit( X, y, epochs=500, batch_size=1, verbose=2 )

#評估模型準確率
scores = model.evaluate( X, y, verbose=0 )
print( 'Model Accuracy: %.2f%%' % ( scores[1] * 100 ) )

#預測結果
# 展示模型預測能力
for pattern in dataX:
    # 把26個字母一個個拿進模型來預測會出現的字母
    x = numpy.reshape( pattern, (1, 1, len( pattern ) ))
    x = x / float( len(alphabet) )

    predict = model.predict( x, verbose=0 )
    index = numpy.argmax( predict )# 機率最大的idx
    result = int_to_char[ index ]  # 看看預測出來的是那一個字母
    seq_in = [ int_to_char[value] for value in pattern ]
    print(  seq_in, '->', result )