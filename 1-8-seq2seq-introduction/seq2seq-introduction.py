'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年02月02日 16:51
@Description: 
@URL: https://colab.research.google.com/drive/1ePAUYuDnE_M0UXq6e9Ug0Qu87t3QJjCi#scrollTo=yKhqbmtXXbDA
@version: V1.0
'''
from keras.models import Sequential
from keras import layers
from keras.utils import plot_model
import numpy as np
from six.moves import range
from IPython.display import Image

class CharacterTable(object):
    """
    給予一組的字符:
    + 將這些字符使用one-hot編碼成數字表示
    + 解碼one-hot編碼數字表示成為原本的字符
    + 解碼字符機率的向量以回覆最有可能的字符
    """
    def __init__(self, chars):
        """初始化字符表

        # 參數:
            chars: 會出現在輸入的可能字符集
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """對輸入的字串進行one-hot編碼

        # 參數:
            C: 要被編碼的字符
            num_rows: one-hot編碼後要回傳的最大行數。這是用來確保每一個輸入都會得到
            相同行數的輸出
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """對輸入的編碼(向量)進行解碼

        # 參數:
            x: 要被解碼的字符向量或字符編碼
            calc_argmax: 是否要用argmax算符找出機率最大的字符編碼
        """
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

# 模型與資料集的參數
TRAINING_SIZE = 50000 # 訓練資料集的samples數
DIGITS = 3            # 加數或被加數的字符數
INVERT = True

# 輸入的最大長度 'int + int' (比如, '345+678')
MAXLEN = DIGITS + 1 + DIGITS

# 所有要用到的字符(包括數字、加號及空格)
chars = '0123456789+ '
ctable = CharacterTable(chars) # 創建CharacterTable的instance

questions = [] # 訓練用的句子 "xxx+yyy"
expected = []  # 訓練用的標籤
seen = set()

print('Generating data...') # 產生訓練資料

while len(questions) < TRAINING_SIZE:
    # 數字產生器 (3個字符)
    f = lambda: int(''.join(np.random.choice(list('0123456789'))
                            for i in range(np.random.randint(1, DIGITS+1))))
    a, b = f(), f()
    # 跳過己經看過的題目以及x+Y = Y+x這樣的題目
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)

    # 當數字不足MAXLEN則填補空白
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)

    # 答案的最大的字符長度為DIGITS + 1
    ans += ' ' * (DIGITS + 1 - len(ans))
    if INVERT:
        # 調轉問題字符的方向, 比如. '12+345'變成'543+21'
        query = query[::-1]
    questions.append(query)
    expected.append(ans)

print('Total addition questions:', len(questions))

# 把資料做適當的轉換, LSTM預期的資料結構 -> [samples, timesteps, features]
print('Vectorization...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool) # 初始一個3維的numpy ndarray (特徵資料)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool) # 初始一個3維的numpy ndarray (標籤資料)

# 將"特徵資料"轉換成LSTM預期的資料結構 -> [samples, timesteps, features]
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)      # <--- 要了解為什麼要這樣整理資料

print("Feature data: ", x.shape)

# 將"標籤資料"轉換成LSTM預期的資料結構 -> [samples, timesteps, features]
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)  # <--- 要了解為什麼要這樣整理資料

print("Label data: ", y.shape)

# 打散 Shuffle(x, y)
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# 保留10%的資料來做為驗證
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)
# 可以試著替代其它種的rnn units, 比如,GRU或SimpleRNN
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

print('Build model...')
model = Sequential()

# ===== 編碼 (encoder) ====

# 使用RNN“編碼”輸入序列，產生HIDDEN_SIZE的輸出。
# 注意：在輸入序列長度可變的情況下，使用input_shape =（None，num_features）
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars)))) # MAXLEN代表是timesteps, 而len(chars)是one-hot編碼的features

# 作為解碼器RNN的輸入，重複提供每個時間步的RNN的最後一個隱藏狀態。
# 重複“DIGITS + 1”次，因為這是最大輸出長度，例如當DIGITS = 3時，最大輸出是999 + 999 = 1998（長度為4)。
model.add(layers.RepeatVector(DIGITS+1))

# ==== 解碼 (decoder) ====
# 解碼器RNN可以是多層堆疊或單層。
for _ in range(LAYERS):
    # 通過將return_sequences設置為True，不僅返回最後一個輸出，而且還以（num_samples，timesteps，output_dim）
    # 的形式返回所有輸出。這是必要的，因為下面的TimeDistributed需要第一個維度是時間步長。
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# 對輸入的每個時間片推送到密集層來對於輸出序列的每一時間步，決定選擇哪個字符。
model.add(layers.TimeDistributed(layers.Dense(len(chars))))

model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
for iteration in range(1, 30):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))

    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)

        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if INVERT else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)