
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
from pylab import *
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler,RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,explained_variance_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from pickle import dump
import warnings
warnings.filterwarnings("ignore")

font1 = {'family': 'Times New Roman','weight': 'normal','size': 13,}
font2 = {'family': 'STSong','weight': 'normal','size': 13,}
fontsize1=13

# 设置字体，以作图显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
# 设置显示属性
pd.set_option('display.max_columns',1000)
pd.set_option('display.max_rows',100)
pd.set_option('display.width',1000)           #宽度
np.set_printoptions(suppress=True)
pd.set_option('precision',4)
np.set_printoptions(precision=4)

def Calculate_Regression_metrics(true_value, predict, label='训练集'):
    mse = mean_squared_error(true_value, predict)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_value, predict)
    r2 = r2_score(true_value, predict)
    ex_var = explained_variance_score(true_value, predict)
    mape = mean_absolute_percentage_error(true_value, predict)
    train_result = pd.DataFrame([mse, rmse, mae, r2,ex_var,mape], columns=[label],
                                index=['mse', 'rmse', 'mae', 'r2','ex_var','mape']).T
    return train_result

def figure_plot(predict, true_value, figure_property,key_label=None):
    # 折线图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(predict, label='Predict Value')
    ax.plot(true_value, label='True Value')
    # ax.plot(true_value,predict,  '*')
    # ax.plot(true_value, true_value, '-')
    # x_ticks = ax.set_xticks([i for i in range(len(key_label))])
    # x_labels = ax.set_xticklabels(key_label,rotation=45,fontdict=font1)
    ax.set_title(figure_property['title'], fontdict=font2)
    ax.set_xlabel(figure_property['X_label'], fontdict=font2)
    ax.set_ylabel(figure_property['Y_label'], fontdict=font2)
    plt.tick_params(labelsize=12)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # #     y_ticks=ax.set_yticks([])
    # #     y_labels=ax.set_yticklabels([-20+i for i in range(20)],rotation=0,fontsize=14)
    # plt.grid()
    plt.legend(prop=font2)
    plt.tight_layout()
    # plt.savefig('../fig/{}.jpg'.format(figure_property['title']), dpi=500, bbox_inches='tight')  # 保存图片
    plt.show()



'''=============1. 读取数据======================='''
def load_data(type=1):
    print(">> 开始运行--加载数据")
    if type == 0:
        data = pd.read_excel('../data/water_tank_noise_5.xlsx', header=0,sheet_name='Sheet1',index_col=0)
        data = data[['u1-L/min','u2-L/min','u3-L/min','h-m']].iloc[:2000]
        # print(data.index)
    else:
        data = pd.read_excel('../data/7月份主要数据.xlsx', header=0,index_col='时间')
        data = data[['下料量','转速','进风管压力','窑尾']].iloc[:2000]
    print(">> 加载数据完成!")
    return data

# -----------------------数据归一化-----------------------------------------
def data_process(data,normal_type='minmax'):
    x = data.values
    y = data.iloc[:, -1].values.reshape(-1, 1)
    # x
    if normal_type == 'minmax':
        scaler_x = MinMaxScaler(feature_range=(-1, 1))
        scaler_x = scaler_x.fit(x)
        # y
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = scaler_y.fit(y)
    else:
        scaler_x = StandardScaler()
        scaler_x = scaler_x.fit(x)
        # y
        scaler_y = StandardScaler()
        scaler_y = scaler_y.fit(y)
    return scaler_x,scaler_y

# -------------------------滑动窗数据处理---------------------------------------
def sliding_windows(data,window_size=1):
    # 定义滑动窗口数据大小
    norm_data = {}                    # 保存数据
    x = []                          #  数据存储
    y = []                          # 原始数据
    num_windows = len(data) - window_size
    time_name = []                   # 数据下标
    time_index = data.index
    # scaler_x,scaler_y = data_process(data, normal_type='minmax')  # 返回归一化函数
    for i in range(num_windows):
        temp_x = data.iloc[i:i+window_size,:].values
        temp_y = data.iloc[i+window_size,-1]
        x.append(temp_x)   # 输入变量为除最后一列的数值
        y.append(temp_y)   # 输出变量为最后一列的数值
        time_name.append(time_index[i+window_size])
        if i % int((num_windows+1)/10) == 0:
             print(f'>> 处理进度--{int(i/num_windows *100)}%')
    norm_data['index_names'] = time_name
    norm_data['x'] = np.array(x)
    norm_data['y'] = np.array(y).reshape(-1,1)
    print(f' >> 处理进度--completed!')
    return norm_data

# ---------------------数据集划分-------------------------------------------
def data_split(data, split_rate=[0.7, 0.5]):
    norm_data = {}
    x = data['x']
    y = data['y']
    index_names = data['index_names']
    train_num = int(x.shape[0] * split_rate[0])
    valid_num = int((len(x) - train_num) * split_rate[1])

    # train data
    train_index = index_names[:train_num]
    x_train = x[:train_num, :].reshape(-1, x.shape[1] * x.shape[2])
    y_train = y[:train_num]

    # valid data
    valid_index = index_names[train_num:train_num + valid_num]
    x_valid = x[train_num:train_num + valid_num, :].reshape(-1, x.shape[1] * x.shape[2])
    y_valid = y[train_num:train_num + valid_num, :]

    # test data
    test_index = index_names[train_num + valid_num:]
    x_test = x[train_num + valid_num:, :].reshape(-1, x.shape[1] * x.shape[2])
    y_test = y[train_num + valid_num:, :]

    print(
        f'\n >> 数据形状：x_train:{x_train.shape}，\t y_train:{y_train.shape},\t x_valid:{x_valid.shape},\t y_valid:{y_valid.shape}'
        f'\t x_test:{x_test.shape},\t y_test:{y_test.shape}')

    norm_data['train_index'] = train_index
    norm_data['valid_index'] = valid_index
    norm_data['test_index'] = test_index
    norm_data['x_train'] = x_train
    norm_data['y_train'] = y_train
    norm_data['x_valid'] = x_valid
    norm_data['y_valid'] = y_valid
    norm_data['x_test'] = x_test
    norm_data['y_test'] = y_test
    return norm_data

# -------------------------定义模型---------------------------------------
def get_model():
    pipelines = {}
    pipelines['LR'] = Pipeline([('scar', StandardScaler()), ('LR', LinearRegression())])
    pipelines['Lasso'] = Pipeline([('scar', StandardScaler()), ('Lasso', Lasso(alpha=0.010))])
    pipelines['RidgeR'] = Pipeline([('scar', StandardScaler()), ('RidgeR', Ridge(alpha=28.91))])
    pipelines['EN'] = Pipeline([('scar', StandardScaler()), ('EN', ElasticNet())])
    pipelines['BP'] = Pipeline([('scar', StandardScaler()), ('BP', MLPRegressor(hidden_layer_sizes=(150,),  activation='relu', learning_rate_init=0.001))])
    pipelines['SVR'] = Pipeline([('scar', StandardScaler()), ('SVR', SVR(C=1,gamma=0.01))])
    pipelines['CART'] = Pipeline([('CART',DecisionTreeRegressor())])
    pipelines['RF'] = Pipeline([('RF',RandomForestRegressor(max_depth=5, max_features=10, n_estimators=1000))])
    pipelines['Adaboost'] = Pipeline([('Ada',AdaBoostRegressor())])
    pipelines['Bag'] = Pipeline([('Bag',BaggingRegressor())])
    # pipelines['AdaRF'] = Pipeline([('AdaRF',AdaBoostRegressor(RandomForestRegressor()))])
    # pipelines['BagRF'] = Pipeline([('BagRF',BaggingRegressor(RandomForestRegressor()))])
    pipelines['StackRF'] = Pipeline([('AdaRF',StackingRegressor(estimators=[('SVR', SVR(C=1,gamma=0.01)),('knn',KNeighborsRegressor(n_neighbors=37))]))])
    # pipelines['VotingRF'] = Pipeline([('BagRF',VotingRegressor(RandomForestRegressor()))])
    pipelines['KNN'] = Pipeline([('Minmax', MinMaxScaler()),('knn',KNeighborsRegressor(n_neighbors=37))])
    # pipelines['ETR'] = Pipeline([('ETR',ExtraTreesRegressor())])
    pipelines['GBR'] = Pipeline([('GBR',GradientBoostingRegressor())])
    return pipelines

if __name__ == '__main__':
    # TODO: 参数设置：
    window_size = 1            # 滑动窗大小
    save_model = False
    save_result = False

    # 1. 加载数据
    data = load_data(type=1)   # type=1 or 0
    # 2. 滑动窗数据处理
    norm_data = sliding_windows(data, window_size)
    # 3. 数据集划分
    train_data = data_split(norm_data, split_rate=[0.7, 0.5])  # 70%:15%:15%
    # 4. 获取数据
    # index_train_names = train_data['train_index']
    # index_valid_names = train_data['valid_index']
    index_test_names = train_data['test_index']

    x_train = train_data['x_train']
    y_train = train_data['y_train']

    x_valid = train_data['x_valid']
    y_valid = train_data['y_valid']

    x_test = train_data['x_test']
    y_test = train_data['y_test']


    # 5. 获取mx
    # 模型命名
    model_name = 'LR'
    pipelines = get_model()
    model = pipelines['LR']

    # 6. 模型训练
    import time
    a = time.time()
    model.fit(x_train,y_train)
    print(f'\n >> {model_name}算法运行时间:{time.time() - a}s')

    # 7. 计算结果
    train_predict =model.predict(x_train)
    valid_predict = model.predict(x_valid)
    test_predict = model.predict(x_test)

    # 8. 是否保存模型
    if save_model == True:
        model_file = f'../model/{model_name}.sav'
        with open(model_file, 'wb') as model_f:
            dump(clf, model_f)

    # 9.模型测试
    # 训练集
    train_result = Calculate_Regression_metrics(y_train, train_predict, label='训练集')
    title = '{}算法训练集结果对比'.format(model_name)
    figure_property = {'title': title, 'X_label': 'Prediction set samples', 'Y_label': 'Prediction Value'}
    figure_plot(train_predict, y_train, figure_property)
    # 验证集
    valid_result =  Calculate_Regression_metrics(y_valid, valid_predict, label='验证集')
    title = '{}算法测试集结果对比'.format(model_name)
    figure_property = {'title': title, 'X_label': 'Prediction set samples', 'Y_label': 'Prediction Value'}
    figure_plot(valid_predict, y_valid, figure_property)
    # 测试集
    test_result =  Calculate_Regression_metrics(y_test, test_predict, label='测试集')
    title = '{}算法测试集结果对比'.format(model_name)
    figure_property = {'title': title, 'X_label': 'Prediction set samples', 'Y_label': 'Prediction Value'}
    figure_plot(test_predict, y_test, figure_property)

    result = pd.concat([train_result, valid_result, test_result], axis=0)
    print('\n >> {}算法计算结果'.format(model_name))
    print(f'\n {result}')


    test_data = pd.DataFrame([y_test.flatten(), test_predict.flatten()]).T
    test_data.index = index_test_names
    test_data.columns = ['实际值','预测值']
    print(f'\n {test_data}')

    # 保存结果
    if save_result == True:
        result.to_excel('../result/{}算法计算结果.xlsx'.format(model_name))
        test_data.to_excel('../result/{}算法预测结果.xlsx'.format(model_name))



