import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

## Regression with one output variable
def regression_one_variable():
    '''Fit a neural net to a linear function'''
    data_size=100
    # Set the tensorflow random seed; sometimes we get a really bad fit for some reason
    tf.set_random_seed(2018)
    # create random x values and y values as a function of x
    X = 100*np.random.random(data_size)
    f = lambda x: 2*x + np.random.random(x.shape)
    y = f(X)
    X_test = 100*np.random.random(data_size)
    y_test = f(X_test)
    # create eval and test input_fn for tensorflow. test doesn't have the y values
    eval_fn = tf.estimator.inputs.numpy_input_fn({'x':X_test}, y_test, shuffle=False)
    test_fn = tf.estimator.inputs.numpy_input_fn({'x':X_test}, shuffle=False)
    x = tf.feature_column.numeric_column('x')
    # fit a DNN regressor to X, y
    reg = tf.estimator.DNNRegressor(hidden_units=[4], feature_columns=[x])
    reg.train(lambda: ({'x':X}, y), max_steps=1000)
    # evaluate its performance
    print(reg.evaluate(input_fn=eval_fn))
    # predict values for the test set
    y_predict = [y['predictions'][0] for y in reg.predict(test_fn)]
    # plot results
    plt.scatter(X,y)
    plt.scatter(X_test,y_predict)
    plt.legend(['y actual', 'y predict'])
    plt.show()

## Regression with two output variables
def regression_two_variables():
    '''Fit two linear functions functions of X at the same time'''
    # The regressor wasn't converging fairly often with a data size of 100
    # A larger data size seems to have helped
    data_size=10000
    X = 100*np.random.random(data_size)
    f = lambda x: np.array([2*x + np.random.random(x.shape), 3*x + np.random.random(x.shape)]).transpose()
    y = f(X)
    X_test = 100*np.random.random(data_size)
    y_test = f(X_test)
    train_fn = tf.estimator.inputs.numpy_input_fn({'x':X}, y, shuffle=True)
    eval_fn = tf.estimator.inputs.numpy_input_fn({'x':X_test}, y_test, shuffle=False)
    test_fn = tf.estimator.inputs.numpy_input_fn({'x':X_test}, shuffle=False)
    x = tf.feature_column.numeric_column('x')
    reg = tf.estimator.DNNRegressor(hidden_units=[10], feature_columns=[x], label_dimension=2)
    reg.train(input_fn = train_fn, steps=10000)
    print(reg.evaluate(input_fn=eval_fn))
    y_predict = np.array([y['predictions'][0] for y in reg.predict(test_fn)])
    y_predict1 = np.array([y['predictions'][1] for y in reg.predict(test_fn)])
    plt.scatter(X_test[0:100],y_test[0:100,0],marker='.')
    plt.scatter(X_test[0:100],y_predict[0:100],marker='.')
    plt.scatter(X_test[0:100], y_test[0:100,1],marker='.')
    plt.scatter(X_test[0:100],y_predict1[0:100],marker='.')
    plt.legend(['y actual [0]', 'y predict [0]', 'y_actual [1]', 'y predict [1]'])
    plt.show()


## x**2 regression example
def regression_x2():
    '''Fit a neural net regressor to the function x**2'''
    data_size = 10000
    X = 10*np.random.random(data_size)
    f = lambda x: np.power(x, 2)
    y = f(X)
    X_test = 10*np.random.random(data_size)
    y_test = f(X_test)
    test_fn = tf.estimator.inputs.numpy_input_fn({'x':X}, y, shuffle=True)
    eval_fn = tf.estimator.inputs.numpy_input_fn({'x':X_test}, y_test, shuffle=False)
    train_fn = tf.estimator.inputs.numpy_input_fn({'x':X_test}, shuffle=False)
    x = tf.feature_column.numeric_column('x')
    reg = tf.estimator.DNNRegressor(hidden_units=[10,10,10], feature_columns=[x], label_dimension=1)
    reg.train(lambda: ({'x':X}, y), max_steps=10000)
    print(reg.evaluate(input_fn=eval_fn))
    y_predict = np.array([y['predictions'][0] for y in reg.predict(train_fn)])
    plt.scatter(X[0:100],y[0:100], marker='.')
    plt.scatter(X_test[0:100],y_predict[0:100], marker='.')
    plt.legend(['y actual [0]', 'y predict [0]'])
    plt.title('Neural net regression fit of y=x**2')
    plt.show()
    
regression_two_variables()