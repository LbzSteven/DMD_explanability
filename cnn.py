#Change depend on the perfomrance
#N_OF_L=1
#F_NODE=16
#Ka=2
#Kb=2
tf.keras.backend.clear_session()
model = Sequential()
model.add(Conv2D(F_NODE, (Ka, Ka), activation = 'relu', input_shape = X_train[0].shape))#,kernel_initializer='random_normal',bias_initializer='zeros'))
########################
for i in range(2,N_OF_L+2):
  model.add(Dropout(0.1))
  model.add(Conv2D(F_NODE*i, (Kb, Kb), activation='relu'))#,kernel_initializer='random_normal',bias_initializer='zeros'))
########################
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense((i+2*F_NODE), activation = 'relu'))#,kernel_initializer='random_normal',bias_initializer='zeros'))
model.add(Dropout(0.5))
model.add(Dense(N_OF_CLASSES, activation='softmax'))#,kernel_initializer='random_normal',bias_initializer='zeros')) #sigmoid [paper]->softmax
model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']) #binary_crossentropy #sparse_categorical_crossentropy
##########################
epoch_max=epoch_dl #$$$
RATE=0.2
if MODE==1:# and 1!=1: #, batch_size=128
  print("<h1>X_val,y_val</h1>")
  #############
  history = model.fit(X_train, y_train,epochs = epoch_max, validation_data= (X_val, y_val) , verbose=0, callbacks=[checkpoint])#  validation_split=0.2   CSVLogger('history.csv')   '''validation_data= (X_test, y_test)'''
else:
  print("<h1>",RATE,"</h1>")
  history = model.fit(X_train, y_train, epochs = epoch_max, validation_split=RATE , verbose=0, callbacks=[checkpoint])#  validation_split=0.2   CSVLogger('history.csv')   '''validation_data= (X_test, y_test)'''