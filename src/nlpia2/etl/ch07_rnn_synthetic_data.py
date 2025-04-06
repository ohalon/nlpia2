import pandas as pd
import numpy as np
t = np.arange(10000)
x = 10*t**2 + v0*t + sin(t/20/np.pi)
x = 10*t**2 + v0*t + sin(t/20/np.pi)
v0 = -3
x = 10*t**2 + v0*t + sin(t/20/np.pi)
x = 10*t**2 + v0*t + np.sin(t/20/np.pi)
x
df = pd.DataFrame()
df['t'] = t
df['y'] = x
df.plot(x='t', y='y')
import seaborn
from matplotlib import pyplot as plt
plt.show()
y = a*t**2 + v0*t + A*np.sin(t/20/np.pi)
y = a*t**2 + v0*t + A*np.sin(t/20/np.pi)
a = -9.8
y = a*t**2 + v0*t + A*np.sin(t/20/np.pi)
A = 1e3
y = a*t**2 + v0*t + A*np.sin(t/20/np.pi)
df['y'] = y
df.plot(x='t', y='y')
plt.show()
y = a*t**2 + v0*t + A*np.sin(t/20/np.pi)*t**2
a = 10
y = a*t**2 + v0*t + A*np.sin(t/20/np.pi)*t**2
df['y'] = y
df.plot(x='t', y='y') ; plt.show()
df.shift?
df['y'].shift(1)
df['y_t1'] = df['y'].shift(1)
X = df[['y']]
y = df['y_t1']
from sklearn.linear_model import Lasso
model = Lasso()
model.fit(X, y)
model.fit(X[1:], y[1:])
X = df[['y_t1']]
y = df['y;]
y = df['y']
model.fit(X[1:], y[1:])
model.score(X, y)
X = df[['y_t1']][1:]
y = df['y'][1:]
model.fit(X, y, num_iter=1000)
model.fit(X, y, n_iter=1000)
model.fit(X, y, niter=1000)
model = Lasso(n_iter=1000)
model = Lasso(niter=1000)
model = Lasso?
model = Lasso(max_iter=1000)
model.fit(X, y)
model = Lasso(max_iter=100000)
model.fit(X, y)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
model.score(X, y)
hist -o -p -f ch07_rnn_synthetic_data.hist.ipy.md
hist -f ch07_rnn_synthetic_data.py
