rm(list = ls())
library(keras)
# Data Preparation -----------------------------------------------------
batch_size <- 128
num_classes <- 10
epochs <- 10

# Input image dimensions
img_rows = img_cols = 28

# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y


#  reshape
x_set = x_train[1:2,,]
# 2 samples, row = 2, col = 28, channel = 1
x_set_r = array_reshape(x_set, c(2,28,28,1), 
                        order = "C") 
x_set_r[1,,,]
# Redefine  dimension of train/test inputs
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

# scaling: Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

# x_train_shape?   dim(x_train)
# n of training?   nrow(x_train)
# n of test?   nrow(x_test)
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)


# Define model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')

# Compile model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# Train model
model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2
)

scores = model %>% evaluate(x_test, y_test)
str(scores)

pred_y = predict_classes(model, x = x_test)
idx_vec = which(pred_y != mnist$test$y)
# check the misclassified data
i = 3
idx = idx_vec[i]
im = mnist$test$x[idx,,]
im = t(im)
image(im[, nrow(im):1], 
      col = gray.colors(256,start=1, end=0),
      axes = FALSE)
pred_y[i]
mnist$test$y[idx]
save_model_hdf5(model, "mnist_cnn.h5")
summary(model)




keras::optimizer