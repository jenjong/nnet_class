# custom loss (quatile function)
library(keras)
quantile <- 0.5
tilted_loss <- function(q, y, f) {
  e <- y - f
  # return type shoud be tensor
  k_mean(k_maximum(q * e, (q - 1) * e), axis = 2) 
}

tilted_loss2 <- function(q, y, f) {
  e <- y - f
  # return type shoud be tensor
  mean(e^2)
}

# check k_mean and k_maximum

ys <- k_constant(c(1,2,3,4), shape = c(4,1))
yhats <- k_constant(c(4,3,3,4), shape = c(4,1))
# evalutation
tilted_loss(0.5, ys, yhats)
# this function does not work!
tilted_loss2(0.5, ys, yhats)




ys <- 1:4
yhats <- 4:1
input = layer_input(shape = c(1))
pred = input %>% layer_dense(unit=1, activation='linear',
                   weights = initializer_constant(value = 1))
keras_model(inputs = input, outputs = pred)

x_train <-
  iris[1:120, c("Petal.Length", "Sepal.Length")] %>% as.matrix()
y_train <-
  iris[1:120, c("Petal.Width", "Sepal.Width")] %>% as.matrix()
x_test <-
  iris[121:150, c("Petal.Length", "Sepal.Length")] %>% as.matrix()
y_test <-
  iris[121:150, c("Petal.Width", "Sepal.Width")] %>% as.matrix()

model <- keras_model_sequential()
model %>%
  layer_dense(units = 32, input_shape = 2) %>%
  layer_activation_leaky_relu() %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 2)

model %>% compile(
  optimizer = "adam",
  loss = function(y_true, y_pred)
    tilted_loss(quantile, y_true, y_pred),
  metrics = "mae"
)

history <-
  model %>% fit(x_train,
                y_train,
                batch_size = 10,
                epochs = 120)

plot(history) 