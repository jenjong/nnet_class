# callback
rm(list = ls())
library(keras)
mnist <- dataset_mnist()
str(mnist)
# small dataset
idx = 1:1000
x_train <- mnist$train$x[idx,,]
y_train <- mnist$train$y[idx]
x_test <- mnist$test$x[idx,,]
y_test <- mnist$test$y[idx]
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
x_train <- x_train / 255
x_test <- x_test / 255
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 16, activation = 'relu', input_shape = c(784)) %>% 
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# callback restore
# check point: last epoch
history <- model %>% fit(
  x_train, y_train, 
  epochs = 10, batch_size = 500, 
  validation_split = 0.2,
  callbacks = list(callback_model_checkpoint(filepath = "cp.h5"))
)

# : every epoch
history <- model %>% fit(
  x_train, y_train, 
  epochs = 10, batch_size = 500, 
  validation_split = 0.2,
  callbacks = list(callback_model_checkpoint(filepath = "cp-{epoch:04d}.h5"))
)

model_1 = load_model_hdf5("cp-0001.h5")
model_1 %>% evaluate(x_test, y_test, verbose = 0)
model_2 = load_model_hdf5("cp-0005.h5")
model_2 %>% evaluate(x_test, y_test, verbose = 0)

# weights
w_list = get_weights(model_1)
str(w_list)
# weight to the first node: w_list[[1]][,1]
# bias to the first node: w_list[[2]][1]
new_model = keras_model_sequential()
new_model %>% 
  layer_dense(units = 16, activation = 'relu', input_shape = c(784))
new_model %>% set_weights(w_list[1:2])
predict(new_model, x = x_test)
