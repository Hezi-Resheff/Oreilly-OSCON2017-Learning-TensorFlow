
import numpy as np
from sklearn import datasets, linear_model

x_data = np.meshgrid(neurons,epochs)
x = np.zeros([15,2])
for ii,(n,e) in enumerate(zip(x_data[0].flatten(),x_data[1].flatten())):
    x[ii,:] = [n,e]
    
x = x.astype(np.float32)
y = means.astype(np.float32)


regr = linear_model.LinearRegression()
regr.fit(x,y)
print('Coefficients: \n', regr.coef_)
plt.plot(y,regr.predict(x),'o',ms=10,alpha=0.7)
plt.plot(plt.xlim(), plt.ylim(), ls="--", c=".3")
plt.xlabel('Real means',fontsize=20)
plt.ylabel('Model prediction',fontsize=20)
plt.show()