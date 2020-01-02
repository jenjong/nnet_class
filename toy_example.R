rm(list = ls()); gc()
library(ggplot2)
library(keras)
# toy example 1 (regression: metric mse)
# hidden = 2 
set.seed(1)
n = 100; p = 10
x = matrix(rnorm(n*p), n, p)
y = sin(x[,5]*10) + rnorm(n)
ggplot()+geom_point(aes(x[,5],y))

model_reg1 = keras_model_sequential(name = 'reg1')
model_reg1 %>% 
  layer_dense(unit = 2, activation ='sigmoid', input_shape=p, name='reg1_d1')%>%
  layer_dense(unit = 1, activation = 'linear', name = 'reg1_d2')

model_reg1 %>% compile(optimizer = 'sgd',
                       loss = 'mse',
                       metric ='mse')
summary(model_reg1)

model_reg1 %>% fit(x = x, y = y, epoch = 5, validation_split = 0.2)
yhat = model_reg1 %>% predict_proba(x = x)
plot(y,yhat ); abline(a = 0, b= 1)

w_list = get_weights(model_reg1)
str(w_list)

# toy example 2
# response y R^3 : metric mse
set.seed(1)
n = 1000; p = 10
x = matrix(rnorm(n*p), n, p)
y1 = sin(x[,5]*10) + rnorm(n)
y2 = cos(x[,5]+x[,6]*5) + rnorm(n)
y3 = cos(x[,1]-x[,2]*5) + rnorm(n)
y = cbind(y1, y2, y3)

model_reg2 = keras_model_sequential(name = 'reg2')
model_reg2 %>% layer_dense(units = 10, activation = 'sigmoid',
                           input_shape = p, name='reg2_d1')%>%
                layer_dense(units = 3, activation = 'linear',
                            input_shape = p, name='reg2_d2')
summary(model_reg2)
model_reg2 %>% compile(loss = 'mse',
                       optimizer = 'sgd',
                       metric = 'mse')
model_reg2 %>% fit(x=x, y=y, epoch = 5, validation_split = 0.2)
yhat = model_reg2 %>% predict_proba(x = x)
plot(y,yhat ); abline(a = 0, b= 1)

w_list = get_weights(model_reg2)
str(w_list)
w_list[[3]]
w_list[[4]]

# top example 3 (classification: accurarcy)
if(!require('rattle.data')) install.packages('rattle.data')
library(rattle.data)
rdata = wine
y = as.numeric(rdata$Type)-1
# why '-1'? 
x = as.matrix(rdata[,-1])
str(x)
y = to_categorical(y, num_classes = 3)
model_cs1 = keras_model_sequential()
model_cs1 %>% layer_dense(units = 10, activation = 'relu',
                          input_shape = ncol(x)) %>%
              layer_dense(units = 3, activation = 'softmax')
model_cs1
model_cs1 %>% 
  compile(loss='categorical_crossentropy', optimizer = 'sgd',
          metric = 'acc')
model_cs1 %>% fit(x = x, y = y, echo = 50, validation_split = 0.2)  
model_cs1 %>% predict(x=x)
model_cs1 %>% predict_proba(x=x)
model_cs1 %>% evaluate(x=x, y=y, verbose = 0)
get_weights(model_cs1)

# data normalization!!
x = as.matrix(rdata[,-1])
x = scale(x)
model_cs1 = keras_model_sequential()
model_cs1 %>% layer_dense(units = 100, activation = 'relu',
                          input_shape = ncol(x)) %>%
  layer_dense(units = 3, activation = 'softmax')
model_cs1
model_cs1 %>% 
  compile(loss='categorical_crossentropy', optimizer = 'sgd',
          metric = 'acc')

model_cs1 %>% fit(x = x, y = y, echo = 50, validation_split = 0.2)  
model_cs1 %>%  predict(x=x)
model_cs1 %>% evaluate(x=x, y=y, verbose = 0)
