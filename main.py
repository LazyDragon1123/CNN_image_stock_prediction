import importlib
import pandas as pd
import lib.Image_generator 
import lib.image_numerator
import model.simple_cnn
filepath_names = '**your tick names list path'
IMAGEN = importlib.reload(lib.Image_generator).IMAGEN
IMAGN = importlib.reload(lib.image_numerator).IMAGN
CNN_simple = importlib.reload(model.simple_cnn).CNN_simple
ticker_names = pd.read_csv(filepath_names,index_col=0).iloc[:,0].to_list()

for ticker in ticker_names:
    try:
        df = pd.read_csv('your ticker list path' + ticker + '.csv',index_col='Date')
        b = IMAGEN(df,ticker,'2021-05-21',100)
        b.generator_w_target()
    except:
        # print(ticker)
        continue

n = IMAGN(ticker_names)
n.numerize()
n.save()

c = CNN_simple('dataset.npy')
c.train()
c.eval()