import numpy as np 
import pandas as pd 
import pickle
from sklearn.linear_model import LogisticRegression


def split_fun(data,ratio):
  np.random.seed(42)
  shuffled_index=np.random.permutation(len(data))
  train_size=(int)(len(data)*ratio)
  train_data_idx=shuffled_index[:train_size]
  test_data_idx=shuffled_index[train_size:]
  return data.iloc[train_data_idx],data.iloc[test_data_idx]
if __name__=='__main__':

    df = pd.read_csv('covid_data.csv')
    train_data,test_data=split_fun(df,0.6)
    X_train=train_data[['Fever','BodyPain','Age','RunnyNose','DiffBreath']].to_numpy()
    X_test=test_data[['Fever','BodyPain','Age','RunnyNose','DiffBreath']].to_numpy()
    Y_train=train_data['Infec_Prob'].to_numpy()
    Y_test=test_data['Infec_Prob'].to_numpy()

    Y_train=Y_train.reshape(649,)
    Y_test=Y_test.reshape(433,)

    clf=LogisticRegression()
    clf.fit(X_train,Y_train)

    file = open("model.pkl","wb")
    pickle.dump(clf,file)
    file.close()

    

