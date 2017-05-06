
feature_columns = learn.infer_real_valued_columns_from_input(x_data)
optimizer = tf.train.GradientDescentOptimizer(0.1)

regressor = learn.LinearRegressor(feature_columns=feature_columns,
                                  optimizer=optimizer)

MSE = []
for i in range(20):
    regressor.fit(x_train, y_train, steps=i, batch_size=506)
    MSE.append(regressor.evaluate(x_test, y_test, steps=1)['loss'])

plt.figure()
plt.plot(np.arange(20),MSE,lw=3,alpha=0.5)
plt.plot(np.arange(20),MSE,'ko',alpha=0.5)
plt.title('Boston housing test data MSE',fontsize=20)
plt.xlabel('# steps',fontsize=20)
plt.ylabel('MSE',fontsize=20)
plt.show()