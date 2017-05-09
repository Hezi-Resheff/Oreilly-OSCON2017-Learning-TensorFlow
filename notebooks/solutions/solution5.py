


input_img = Input(shape=(28, 28, 1))  
im = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
im = MaxPooling2D((2, 2), padding='same')(im)
im = Conv2D(32, (3, 3), activation='relu', padding='same')(im)
encoded = MaxPooling2D((2, 2), padding='same')(im)


im = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
im = UpSampling2D((2, 2))(im)
im = Conv2D(32, (3, 3), activation='relu', padding='same')(im)
im = UpSampling2D((2, 2))(im)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(im)

model = Model(input_img, decoded)
model.compile(optimizer='adadelta', loss='binary_crossentropy')

model.fit(x_train_noisy, x_train,
                epochs=1,
                batch_size=32,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

decoded_imgs = model.predict(x_test_n)

n_imgs = 10
f,axarr = plt.subplots(2,n_imgs,figsize=[20,5])
for i in range(n_imgs):
    ax = axarr[0,i]
    ax.imshow(x_test_n[i,:,:,0])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = axarr[1,i]
    ax.imshow(decoded_imgs[i,:,:,0])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
            
plt.tight_layout()
plt.show()