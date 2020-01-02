rm(list = ls()) ; gc()
library(keras)
model = load_model_hdf5("mnist_cnn.h5")
# Input image dimensions
img_rows <- 28
img_cols <- 28

# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_test <- mnist$test$x
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)
x_test <- x_test / 255
pred_y = predict_classes(model, x = x_test)

# check weights
w_list = get_weights(model)
str(w_list)
w_list[[1]][,,,1]
w_list[[2]][1]




## model  second fileter
new_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu')
new_model %>% set_weights(w_list[1:4])

# original image
par(mfrow = c(1,2))
k = 1
im = mnist$test$x[k,,]
im = t(im)
image(im[, nrow(im):1], 
      col = gray.colors(256,start=1, end=0),
      axes = FALSE, 
      main = paste(mnist$test$y[k]))

pred_h = new_model %>% predict(x = x_test[k,,,, drop = F])
h = array_reshape(pred_h, c(24,24,64))
i = 3
im = h[,,i]
im = t(im)
image(im[, nrow(im):1], 
      col = gray.colors(256,start=1, end=0),
      axes = FALSE)


jpeg("test.jpg", width = 4800, height = 4800)
par(mfrow = c(8,8))
h = array_reshape(pred_h, c(24,24,64))
for (i in 1:64)
{
  im = h[,,i]
  im = t(im)
  image(im[, nrow(im):1], 
        col = gray.colors(256,start=1, end=0),
        axes = FALSE, 
        main = paste0('fileter-',i))
}
dev.off()




