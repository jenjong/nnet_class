# installation guide
# 1. install R (>=version 3.6)
# 2. install Rstudio
# 3. install Rtools 3.4
    #  https://cran.r-project.org/bin/windows/Rtools/
# 4. install anaconda 3.7 
    # "https://www.anaconda.com/distribution/#download-section"
version
update.packages()
install.packages("devtools")
library(devtools)
# Very important notice:
#  Do not install keras with gpu version unless your CUDA library is adequate!
install.packages("keras")
library(keras) 
install_keras()


# Note: after intallation, please check the following code! 
# if it fails, please the installation of Keras R in google!
library(keras) 
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
x_train <- x_train / 255
x_test <- x_test / 255
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
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



