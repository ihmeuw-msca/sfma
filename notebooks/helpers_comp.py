import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
class compare():
    def plot_all(self,df_cnls, df_qle, df_mom, df_sfma, df_dea):
        plt.scatter(df_cnls['x'], df_cnls['y'], color= "gray", edgecolor="none", alpha=0.2) 
        plt.plot(df_cnls['x'], df_cnls['y_true'])
        plt.plot(df_cnls['x'], df_cnls['value'])
        plt.plot(df_sfma['x'], df_sfma['value'])
        plt.plot(df_qle['x'], df_qle['value'])
        plt.plot(df_mom['x'], df_mom['value'])
        plt.plot(df_dea['x'], df_dea['y'])
        plt.legend(['data','true_fun','cnls','sfma','QLE','MOM','DEA'])
        plt.grid()

    def mse_all(self,df_cnls, df_qle, df_mom, df_sfma):
        mse_QLE = mean_squared_error(df_qle['value'], df_qle['y_true'])
        mse_MOM = mean_squared_error(df_mom['value'], df_mom['y_true'])
        mse_cnls = mean_squared_error(df_cnls['value'], df_cnls['y_true'])
        mse_sfma = mean_squared_error(df_sfma['value'], df_sfma['y_true'])
        print('mse_QLE:',mse_QLE)
        print('mse_MOM:',mse_MOM)
        print('mse_cnls:',mse_cnls)
        print('mse_sfma:',mse_sfma)