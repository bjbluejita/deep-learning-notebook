'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年02月26日 15:49
@Description: 
@URL: https://nbviewer.jupyter.org/github/erhwenkuo/deep-learning-with-keras-notebooks/blob/master/8.1-jieba-word-tokenizer.ipynb
@version: V1.0
'''
import  jieba

seg_list = jieba.cut( '我来自重庆大学', cut_all=True )
print( '全模式:', '/'.join( seg_list ))

seg_list = jieba.cut( '我来自重庆大学', cut_all=False )
print( '准确模式:', '/'.join( seg_list ))

seg_list = jieba.cut("他來到了台北101大樓")  # 預設為精確模式
print("精確模式(預設): " +"/ ".join(seg_list))

seg_list = jieba.cut_for_search( 'Apache Spark是一個開源叢集運算框架，最初是由加州大學柏克萊分校AMPLab所開發。')
print("搜索引擎模式模式: " +"/ ".join(seg_list))

