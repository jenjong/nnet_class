# I. introduction 
# check the installation anaconda!
# install guide: "https://belitino.tistory.com/257"
install.packages('devtools')
install.packages('Rcpp')
devtools::install_github("rstudio/tensorflow")
devtools::install_github("rstudio/keras")
library(keras)
install_keras() 
#* Miniconda has been successfully installed at "C:/Users/Jeon/AppData/Local/r-miniconda".
#Error: Installation of TensorFlow not found.
# start!
library(keras)
mnist <- dataset_mnist()
str(mnist)
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# check the figure
idx = 1
im = x_train[idx,,]
# check data 
im
y_train[idx]
# just for proper visualization
  im = t(im)
  image(im[, nrow(im):1], 
        col = gray.colors(256,start=1, end=0),
        axes = FALSE)

# picture size?
str(im)
#  reshape
x_set = x_train[1:2,,]
x_set_r = array_reshape(x_set, c(2,28*28), order = "C") 
# what is order "C"? -> rowwise
x_set_r[1,]

# vectorization of y
y_set = y_train[1:2]
y_set_c <- to_categorical(y_set, 10)
y_set_c[1,]

# OK! make train and test data set
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 16, activation = 'sigmoid', input_shape = c(784)) %>% 
  layer_dense(units = 10, activation = 'softmax')
summary(model)
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)
history <- model %>% fit(
  x_train, y_train, 
  epochs = 1, batch_size = 128, 
  validation_split = 0.2
)





# figure
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')
summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)
plot(history)
model %>% evaluate(x_test, y_test)
pred_y = model %>% predict_classes(x_test)
idx_vec = which(pred_y != mnist$test$y)
# check the misclassified data

# check the figure
i = 3
idx = idx_vec[i]
im = mnist$test$x[idx,,]
im = t(im)
image(im[, nrow(im):1], 
      col = gray.colors(256,start=1, end=0),
      axes = FALSE)
pred_y[i]
mnist$test$y[idx]

# predict probabilities?
pred_y = model %>% predict_proba(x_test)
pred_y[i,]
hist(apply(pred_y,1,max))
# save the model
save_model_hdf5(model, "mnist.h5")
# loda the model and predict
new_model = load_model_hdf5("mnist.h5")
predict_classes(new_model, x_test)

# multiclass glm?
# what is different?
# regression problem: linear vs nonlinear