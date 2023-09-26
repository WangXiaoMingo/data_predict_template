
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
from pylab import *
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler,RobustScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,explained_variance_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import tensorflow as tf
import warnings
import random
warnings.filterwarnings("ignore")

# 固定随机数种子
random_seed = 42
random.seed(random_seed)  # set random seed for python
np.random.seed(random_seed)  # set random seed for numpy
tf.random.set_seed(random_seed)  # set random seed for tensorflow-cpu
os.environ['TF_DETERMINISTIC_OPS'] = '1' # set random seed for tensorflow-gpu
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

def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 200 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)


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
        data = pd.read_excel('../data/water_tank_noise_0.xlsx', header=0,sheet_name='Sheet1',index_col=0)
        data = data[['u1-L/min','u2-L/min','u3-L/min','h-m']].iloc[:2000]
        # print(data.index)
    else:
        data = pd.read_excel('../data/7月份主要数据.xlsx', header=0,index_col='时间')
        data = data[['下料量','转速','进风管压力','窑尾']].iloc[:2000]
    print(">> 加载数据完成!")
    return data

# -----------------------数据归一化-----------------------------------------
def data_process(data,normal_type='minmax'):
    norm_data = {}
    x = data.values
    y = data.iloc[:, -1].values.reshape(-1, 1)
    # x
    if normal_type == 'minmax':
        scaler_x = MinMaxScaler(feature_range=(-1, 1))
        scaler_x = scaler_x.fit(x)
        x_norm = scaler_x.transform(x)
        # y
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = scaler_y.fit(y)
        y_norm = scaler_y.transform(y)

    else:
        scaler_x = StandardScaler()
        scaler_x = scaler_x.fit(x)
        x_norm = scaler_x.transform(x)
        # y
        scaler_y = StandardScaler()
        scaler_y = scaler_y.fit(y)
        y_norm = scaler_y.transform(y)
    norm_data['index_names'] = data.index.values
    norm_data['x_norm'] = np.array(x_norm)
    norm_data['y_norm'] = np.array(y_norm)
    norm_data['y'] = np.array(y)
    norm_data['scaler_x'] = scaler_x
    norm_data['scaler_y'] = scaler_y
    return norm_data

def data_split(data, split_rate=[0.7, 0.5]):
    norm_data = {}
    x_norm = data['x_norm']
    y_norm = data['y_norm']
    y = data['y']
    index_names = data['index_names']
    train_num = int(x_norm.shape[0] * split_rate[0])
    valid_num = int((len(x_norm) - train_num) * split_rate[1])

    # train data
    train_index = index_names[:train_num]
    x_train_norm = tf.expand_dims(x_norm[:train_num, :],1)
    y_train_norm = tf.expand_dims(y_norm[:train_num],1)
    y_train = y[:train_num]

    # valid data
    valid_index = index_names[train_num:train_num + valid_num]
    x_valid_norm = tf.expand_dims(x_norm[train_num:train_num + valid_num, :],1)
    y_valid_norm = tf.expand_dims(y_norm[train_num:train_num + valid_num, :],1)
    y_valid = y[train_num:train_num + valid_num, :]

    # test data
    test_index = index_names[train_num + valid_num:]
    x_test_norm = tf.expand_dims(x_norm[train_num + valid_num:, :],1)
    y_test_norm = tf.expand_dims(y_norm[train_num + valid_num:, :],1)
    y_test = y[train_num + valid_num:, :]

    print(
        f'\n >> 数据形状：x_train:{x_train_norm.shape}，\t y_train:{y_train.shape},\t x_valid:{x_valid_norm.shape},\t y_valid:{y_valid.shape}'
        f'\t x_test:{x_test_norm.shape},\t y_test:{y_test.shape}')

    norm_data['train_index'] = train_index
    norm_data['valid_index'] = valid_index
    norm_data['test_index'] = test_index
    norm_data['x_train_norm'] = x_train_norm
    norm_data['y_train_norm'] = y_train_norm
    norm_data['y_train'] = y_train
    norm_data['x_valid_norm'] = x_valid_norm
    norm_data['y_valid_norm'] = y_valid_norm
    norm_data['y_valid'] = y_valid
    norm_data['x_test_norm'] = x_test_norm
    norm_data['y_test_norm'] = y_test_norm
    norm_data['y_test'] = y_test
    return norm_data

def get_model():
    model = Sequential()
    model.add(LSTM(128, input_shape=(window, x_dim), return_sequences=False))
    model.add(Dense(units=64))
    model.add(Dense(units=1, activation='linear'))
    return model

def train_model(model,model_name,train_data,lr=0.0005,epoch=200,show_loss=True):
    import time
    a = time.time()
    # 保存每次训练过程中的最佳的训练模型,有一次提升, 则覆盖一次.
    checkpoint = ModelCheckpoint(f'../temp/{model_name}_model.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='min')
    reduce_lr = LearningRateScheduler(scheduler)
    callbacks_list = [checkpoint, reduce_lr]

    optimer = Adam(lr)  #
    # optimer = Adam(lr=0.0001, beta_1=0.8, beta_2=0.9, epsilon=1e-08, decay=0.0, amsgrad=True)
    model.compile(loss='mse', optimizer=optimer, metrics=['mse'])

    history = model.fit(train_data['x_train_norm'], train_data['y_train_norm'], epochs=epoch, batch_size=128, shuffle=False,
                        validation_data=(train_data['x_valid_norm'], train_data['y_valid_norm']), callbacks=callbacks_list)  # 训练模型epoch次
    model.summary()
    b = time.time() - a
    print(f'>> 程序运行时间{b} s')
    del model
    # 迭代图像
    if show_loss:
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(epoch)
        plt.plot(epochs_range, loss, label='Train Loss')
        plt.plot(epochs_range, val_loss, label='Test Loss')
        plt.legend(loc='upper right')
        plt.title(f'{model_name} Train and Val Loss ')
        plt.show()



if __name__ == '__main__':
    # TODO: 参数设置：
    save_model = False
    save_result = False
    normal_type = 'minmax'     # 归一化方法
    split_rate=[0.7, 0.5]      # 划分数据集
    # train = True              # 是否训练
    train = False             # 是否训练

    # 1. 加载数据
    data = load_data(type=0)   # type=1 or 0
    # 2. 滑动窗数据处理 + 归一化
    norm_data = data_process(data,normal_type)
    scaler_y = norm_data['scaler_y']
    # 3. 数据集划分
    train_data = data_split(norm_data, split_rate)  # 70%:15%:15%
    # 4. 获取数据
    # index_train_names = train_data['train_index']
    # index_valid_names = train_data['valid_index']
    # y_train_norm = train_data['y_train_norm']
    # y_valid_norm = train_data['y_valid_norm']
    # y_test_norm = train_data['y_test_norm']
    index_test_names = train_data['test_index']
    x_train_norm = train_data['x_train_norm']
    y_train = train_data['y_train']
    x_valid_norm = train_data['x_valid_norm']
    y_valid = train_data['y_valid']
    x_test_norm = train_data['x_test_norm']
    y_test = train_data['y_test']

    # 5. 获取模型
    # 模型命名
    model_name = 'LSTM'
    window, x_dim = x_train_norm.shape[1], x_train_norm.shape[2]
    model = get_model()

    # 6. 模型训练
    # 参数设置
    if train:
        train_model(model,model_name,train_data,lr=0.0005,epoch=200,show_loss=True)

    # 7. 加载最优模型
    model = load_model(f'../temp/{model_name}_model.h5')

    # 8. 模型预测
    train_predict = scaler_y.inverse_transform(model.predict(x_train_norm))
    valid_predict = scaler_y.inverse_transform(model.predict(x_valid_norm))
    test_predict = scaler_y.inverse_transform(model.predict(x_test_norm))

    # 9. 结果展示
    # 训练集
    train_result = Calculate_Regression_metrics(y_train.flatten(), train_predict.flatten(), label='训练集')
    title = '{}算法训练集结果对比'.format(model_name)
    figure_property = {'title': title, 'X_label': 'Prediction set samples', 'Y_label': 'Prediction Value'}
    figure_plot(train_predict.flatten(), y_train.flatten(), figure_property)
    # 验证集
    valid_result =  Calculate_Regression_metrics(y_valid.flatten(), valid_predict.flatten(), label='验证集')
    title = '{}算法测试集结果对比'.format(model_name)
    figure_property = {'title': title, 'X_label': 'Prediction set samples', 'Y_label': 'Prediction Value'}
    figure_plot(valid_predict.flatten(), y_valid.flatten(), figure_property)
    # 测试集
    test_result =  Calculate_Regression_metrics(y_test.flatten(), test_predict.flatten(), label='测试集')
    title = '{}算法测试集结果对比'.format(model_name)
    figure_property = {'title': title, 'X_label': 'Prediction set samples', 'Y_label': 'Prediction Value'}
    figure_plot(test_predict.flatten(), y_test.flatten(), figure_property)

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



