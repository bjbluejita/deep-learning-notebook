import thulac	

thu1 = thulac.thulac( seg_only=True )  #默认模式
text = thu1.cut("大概一年前，我在AINLP的公众号对话接口里基于腾讯800万大的词向量配置了一个相似词查询的接口", text=True)  #进行一句话分词
print( [ word + '@' for word in text.split( ' ' ) ] )