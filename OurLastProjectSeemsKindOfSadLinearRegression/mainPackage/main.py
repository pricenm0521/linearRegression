# main.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# collapsing code into one module today
def demo():
    np.random.seed(42)
    X = 30 * np.random.random((20, 1)) # 20 random values from 0 to 30
    # capital X is used for independent variables
    # Just take the default mu and sigma (mean and std dev)
    y = 0.5 * X + 1.0 + np.random.normal(size=X.shape) # adds randomness to the x values
    # lower case y is used for dependent variables
    # change the y variable so the first value is 100 aka an outlier
    print(type(y))
    print(y.shape)
    # y[0]=100 # remove the outlier
    # for i in range(0, len(X)):
    #     print(X[i], ", ", y[i])
    model = LinearRegression()
    model.fit(X, y)
    
    X_new = np.linspace(0, 30, 100)
    y_new = model.predict(X_new[:, np.newaxis])
    #for i in range(0, len(X_new)):
    #    print(X_new[i], ", ", y_new[i])
    # Coefficient of Determination: best score is 1.
    print('Model Score: ', model.score(X, y))
    # find another metric for judging the model
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print('MSE: ', mse)
    fig, axs = plt.subplots(1, 1, figsize=(9, 9))
    fig.suptitle('Linear Regression of random data.\nTraining data in green.\nRegression line in blue.')

    axs.scatter(X, y, color = "g", s = 99)
    axs.plot(X_new, y_new)

    # Clean up the plot and display it
    axs.set_xlabel('x')
    axs.set_ylabel('y')
    
    axs.axis('tight')
    
    plt.show()


demo() # invoke the function above