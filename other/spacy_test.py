from torchtext import data, datasets
import spacy

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
                    eos_token = EOS_WORD, pad_token=BLANK_WORD)

MAX_LEN = 100
train, val, test = datasets.IWSLT.splits( exts=( '.de', '.en' ), fields=(SRC, TGT), \
                                           filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN )

MIN_FREQ = 2
SRC.build_vocab( train.src, min_freq=MIN_FREQ )
TGT.build_vocab( train.trg, min_freq=MIN_FREQ ) 

en_text = "I think it won't succeed."
print( [ x.text + "*" for x in spacy_en.tokenizer( en_text ) ] )

import thulac	

thu1 = thulac.thulac( seg_only=True )  #默认模式
text = thu1.cut("大概一年前，我在AINLP的公众号对话接口里基于腾讯800万大的词向量配置了一个相似词查询的接口", text=True)  #进行一句话分词
print( [ word + '@' for word in text.split( ' ' ) ] )