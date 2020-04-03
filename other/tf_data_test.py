'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年07月22日 14:40
@Description: 
@URL: https://www.cnblogs.com/hellcat/p/8569651.html
@version: V1.0
'''
import  numpy as np
import tensorflow as tf
from tensorflow.python.data import  Dataset

'''
tf.data.Dataset.from_tensor_slices真正作用是切分传入Tensor的第一个维度，
生成相应的dataset，即第一维表明数据集中数据的数量，之后切分batch等操作
都以第一维为基础。
'''
#dataset = Dataset.from_tensor_slices( np.array( [ 1.0, 2.0, 3.0, 4.0, 5.0 ] ) )
#dataset = Dataset.from_tensor_slices( np.random.uniform( size=(5, 2, 2)))
'''
字典使用:在实际使用中，我们可能还希望Dataset中的每个元素具有更复杂的形式，
如每个元素是一个Python中的元组，或是Python中的词典.
注意，image_tensor、label_tensor和上面的高维向量一致，第一维表示数据集中
数据的数量。相较之下，字典中每一个key值可以看做数据的一个属性，value则存
储了所有数据的该属性值。

dataset = Dataset.from_tensor_slices(
    {
        "a": np.array( [ 1.0, 2.0, 3.0, 4.0, 5.0 ] ),
        "b": np.random.uniform( size=( 5, 2 ) )
    })
'''
'''
复杂的tuple组合数据
'''
dataset = Dataset.from_tensor_slices(
    ( np.array( [ 1.0, 2.0, 3.0, 4.0, 5.0 ] ), np.random.uniform( size=( 5, 2 ) ))
)

'''
map 操作
和python中的map类似，map接收一个函数，Dataset中的每个元素都会被当作这个函数的输入，
并将函数返回值作为新的Dataset
'''
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.map( lambda  x: x + 1 )

iterator = dataset.make_one_shot_iterator()
one_elemnet = iterator.get_next()
with tf.Session() as sess:
    for i in range( 5 ):
        print( sess.run( one_elemnet ) )


'''
batch 操作
batch就是将多个元素组合成batch
'''
dataset = Dataset.from_tensor_slices(
    {
        'a': np.array( [1.0, 2.0, 3.0, 4.0, 5.0 ] ),
        'b': np.random.uniform( size=( 5, 2))
    }
)
dataset = dataset.batch(2)
iterator = dataset.make_one_shot_iterator()
batch_one_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(3):
        print( sess.run( batch_one_element ) )

'''
shuffle
shuffle的功能为打乱dataset中的元素，它有一个参数buffersize，
表示打乱时使用的buffer的大小，建议舍的不要太小，一般是1000：
'''
dataset = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "b": np.random.uniform(size=(5, 2))
    })

dataset = dataset.shuffle(buffer_size=5)

iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session( ) as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")