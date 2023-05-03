import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from discretize import FCCA

if __name__ == '__main__':
    # Reading the dataset
    data = pd.read_csv('datasets/boston.csv')
    label_column = data.columns[-1]
    feature_columns = data.columns[:-1]

    # Scaling the features between 0 and 1
    scaler = MinMaxScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])

    # Train - test split
    data_ts = data.sample(n=int(0.3*len(data)))
    data_tr = data.drop(index=data_ts.index)

    x_tr, y_tr = data_tr[feature_columns], data_tr[label_column]
    x_ts, y_ts = data_ts[feature_columns], data_ts[label_column]

    # Target model
    target = GradientBoostingClassifier(max_depth=1, n_estimators=100,learning_rate=0.1)

    # Hyperparameters for the discretization - default values
    discretizer = FCCA(target, p0=0.5, p1=1, lambda0=0.1, lambda1=1, lambda2=0)

    # Discretization
    x_tr_discr, y_tr_discr = discretizer.fit_transform(x_tr, y_tr)
    x_ts_discr, y_ts_discr = discretizer.transform(x_ts, y_ts)

    # Compression - inconsistency rate
    print(f'Compression rate: {discretizer.compression_rate(x_ts, y_ts)}')
    print(f'Inconsistency rate: {discretizer.inconsistency_rate(x_ts, y_ts)}')

    print('Setting Q to 0.7')
    # Increasing the value of Q
    tao_q = discretizer.selectThresholds(0.7)

    # Discretization
    x_tr_discr, y_tr_discr = discretizer.transform(x_tr, y_tr, tao_q)
    x_ts_discr, y_ts_discr = discretizer.transform(x_ts, y_ts, tao_q)

    # Compression - inconsistency rate
    print(f'Compression rate: {discretizer.compression_rate(x_ts, y_ts, tao_q)}')
    print(f'Inconsistency rate: {discretizer.inconsistency_rate(x_ts, y_ts, tao_q)}')






