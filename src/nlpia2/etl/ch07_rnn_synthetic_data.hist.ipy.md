>>> import pandas as pd
>>> import numpy as np
>>> t = np.arange(10000)
>>> x = 10*t**2 + v0*t + sin(t/20/np.pi)
>>> x = 10*t**2 + v0*t + sin(t/20/np.pi)
>>> v0 = -3
>>> x = 10*t**2 + v0*t + sin(t/20/np.pi)
>>> x = 10*t**2 + v0*t + np.sin(t/20/np.pi)
>>> x
array([0.00000000e+00, 7.01591482e+00, 3.40318256e+01, ...,
       9.99370100e+08, 9.99570047e+08, 9.99770014e+08])
>>> df = pd.DataFrame()
>>> df['t'] = t
>>> df['y'] = x
>>> df.plot(x='t', y='y')
<AxesSubplot:xlabel='t'>
>>> import seaborn
>>> from matplotlib import pyplot as plt
>>> plt.show()
>>> y = a*t**2 + v0*t + A*np.sin(t/20/np.pi)
>>> y = a*t**2 + v0*t + A*np.sin(t/20/np.pi)
>>> a = -9.8
>>> y = a*t**2 + v0*t + A*np.sin(t/20/np.pi)
>>> A = 1e3
>>> y = a*t**2 + v0*t + A*np.sin(t/20/np.pi)
>>> df['y'] = y
>>> df.plot(x='t', y='y')
<AxesSubplot:xlabel='t'>
>>> plt.show()
>>> y = a*t**2 + v0*t + A*np.sin(t/20/np.pi)*t**2
>>> a = 10
>>> y = a*t**2 + v0*t + A*np.sin(t/20/np.pi)*t**2
>>> df['y'] = y
>>> df.plot(x='t', y='y') ; plt.show()
>>> df.shift?
>>> df['y'].shift(1)
0                NaN
1       0.000000e+00
2       2.291482e+01
3       1.613025e+02
4       5.105551e+02
            ...     
9995    9.263976e+10
9996    9.201437e+10
9997    9.136567e+10
9998    9.069381e+10
9999    8.999895e+10
Name: y, Length: 10000, dtype: float64
>>> df['y_t1'] = df['y'].shift(1)
>>> X = df[['y']]
>>> y = df['y_t1']
>>> from sklearn.linear_model import Lasso
>>> model = Lasso()
>>> model.fit(X, y)
>>> model.fit(X[1:], y[1:])
Lasso()
>>> X = df[['y_t1']]
>>> y = df['y;]
>>> y = df['y']
>>> model.fit(X[1:], y[1:])
Lasso()
>>> model.score(X, y)
>>> X = df[['y_t1']][1:]
>>> y = df['y'][1:]
>>> model.fit(X, y, num_iter=1000)
>>> model.fit(X, y, n_iter=1000)
>>> model.fit(X, y, niter=1000)
>>> model = Lasso(n_iter=1000)
>>> model = Lasso(niter=1000)
>>> model = Lasso?
>>> model = Lasso(max_iter=1000)
>>> model.fit(X, y)
Lasso()
>>> model = Lasso(max_iter=100000)
>>> model.fit(X, y)
Lasso(max_iter=100000)
>>> from sklearn.linear_model import LinearRegression
>>> model = LinearRegression()
>>> model.fit(X, y)
LinearRegression()
>>> model.score(X, y)
0.9997534890196353
>>> hist -o -p -f ch07_rnn_synthetic_data.hist.ipy.md
