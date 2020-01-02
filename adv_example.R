# use keras_model (advanced topic)
rm(list = ls()); gc()
library(keras)
# toy example 1 (regression: metric mse)
set.seed(1)
n = 100; p = 10
x = matrix(rnorm(n*p), n, p)
y = sin(x[,5]*10) + rnorm(n)

input = layer_input(shape = p, name='input')
pred = input %>%  layer_dense(units=2, activation='sigmoid') %>% 
  layer_dense(unit=1, activation='linear')
model = keras_model(inputs = input, outputs = pred)
model %>% compile(loss='mse', optimizer='sgd', metric='mse')
model %>% keras::fit(x=x, y=y, epochs=50,
                     validation_split=0.2)
model %>% evaluate(x=x, y=y, verbose = 0)
model %>% predict(x=x)


# toy example 3 (regression: metric mse)
library(rattle.data)
rdata = wine
y = as.numeric(rdata$Type)-1
# why '-1'? 
x = as.matrix(rdata[,-1])
str(x)
y = to_categorical(y, num_classes = 3)
x = scale(x)
input = layer_input(shape = ncol(x))
pred = input %>% layer_dense(units=10, activation='sigmoid') %>%
  layer_dense(units=5, activation='sigmoid') %>%
  layer_dense(units=3, activation='softmax')
model = keras_model(inputs=input, outputs=pred)
model %>% compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metric='acc')
model %>%  fit(x=x, y=y, epochs=50, validation_split=0.7)
model %>%  evaluate(x=x,y=y, verbose = 0)


# concatenating network (multiple inputs)
set.seed(1)
rdata = wine
y = as.numeric(rdata$Type)-1
# why '-1'? 
x1 = as.matrix(rdata[,-1]) 
y = to_categorical(y, num_classes = 3)
x1 = scale(x1)
n=nrow(x1); p=ncol(x1)
x2 = matrix(rnorm(n*p*2),n,p*2)
input1 = layer_input(shape=13, name='i1')
input2 = layer_input(shape=26, name='i2')
pred1 = input1 %>% layer_dense(units = 3, activation = 'sigmoid') %>% 
  layer_dense(units = 3, activation = 'sigmoid')
pred2 = input2 %>% layer_dense(units = 5, activation = 'sigmoid')
pred3  = layer_concatenate(list(pred2, pred1)) %>% 
  layer_dense(units=3, activation='softmax')
model = keras_model(inputs=list(input1, input2), outputs= pred3) 
model %>% compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metric='acc')
model %>% fit(x=list(x1,x2), y=y, epochs=10,
              validation_split = 0.2)

get_weights(model)
