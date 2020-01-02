library(keras)
rdata = read.csv("kyphosis.csv", stringsAsFactors = FALSE)
rdata$y = as.integer(rdata$Kyphosis=="present")
rdata$Kyphosis = NULL
str(rdata)
# train_set index
set.seed(1)
ir = sort(sample(1:nrow(rdata), 60))
x_train = as.matrix(rdata[ir,1:3])
y_train = rdata[ir,4]
x_test = as.matrix(rdata[-ir,1:3])
y_test = rdata[-ir,4]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = keras_model_sequential()
model %>% layer_dense(units = 256, activation = 'relu', 
                      input_shape = ncol(x_train)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 2, activation = 'softmax')


model %>% compile(loss = 'binary_crossentropy',
                  optimizer = 'sgd',
                  metrics = c('accuracy'))

history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 5, 
  validation_split = 0.2
)

# model evaluation
model %>% evaluate(x_test, y_test)
model %>% predict_classes(x = x_test)

# tuning
model_r = keras_model_sequential()
model_r %>% layer_dense(units = 1024, activation = 'relu', input_shape = ncol(x_train)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 2, activation = 'sigmoid')

# optimizer (delete)
opt = optimizer_rmsprop(lr = 0.001, rho = 0.9, epsilon = NULL, decay = 0,
                  clipnorm = NULL, clipvalue = NULL)
model_r %>% compile(loss = 'binary_crossentropy',
                  optimizer = opt,
                  metrics = c('accuracy'))

history <- model_r %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 5, 
  validation_split = 0.2
)
# model evaluation
model_r %>% evaluate(x_test, y_test)
model_r %>% predict_classes(x = x_test)


#### glmnet 
model_g = keras_model_sequential()
model_g %>% layer_dense(unit = 2, activation = 'sigmoid',
                        kernel_regularizer = 
                          regularizer_l1_l2(l1 = 0, l2 = 0.0001))
model_g %>% compile(loss = 'binary_crossentropy',
                    optimizer = "sgd",
                    metrics = c('accuracy'))
history <- model_g %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 5, 
  validation_split = 0.2
)
model_g %>% evaluate(x_test, y_test)
model_g %>% predict_classes(x = x_test)

library(MASS);library(glmnet)
fit_cv = cv.glmnet(x = x_train, y = y_train[,2], family = "binomial")
fit = glmnet(x = x_train, y = y_train[,2], family = "binomial", 
             lambda = fit_cv$lambda.min)
predict(fit, newx = x_test, type = 'class')
