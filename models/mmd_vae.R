library(tensorflow)
library(keras)
library(reticulate)
library(tfdatasets)
library(tidyverse)
library(glue)

tf$keras$backend$set_floatx('float32')
source('utils.R')


# Obtenció de dades i creacio del split train / test ----------------------------------

GetData <- function(){
  
  mnist <- keras::dataset_mnist()
  c(train_images, train_labels) %<-% mnist$train
  c(test_images, test_labels) %<-% mnist$test
  
  train_x <-
    train_images %>% `/`(255) %>% 
    k_reshape(c(60000, 28, 28, 1)) %>% 
    k_cast(dtype='float32')
  
  test_x <-
    test_images %>% `/`(255) %>% 
    k_reshape(c(10000, 28, 28, 1)) %>% 
    k_cast(dtype='float32')
  
  buffer_size <- 60000
  batch_size <- 100
  batches_per_epoch <- buffer_size / batch_size
  
  train_dataset <- tensor_slices_dataset(train_x) %>%
    dataset_shuffle(buffer_size) %>%
    dataset_batch(batch_size)
  
  test_dataset <- tensor_slices_dataset(test_x) %>%
    dataset_batch(10000)
  
  return(list(train_dataset = train_dataset, 
              test_dataset = test_dataset))
  
}

c(train_dataset,test_dataset) %<-% GetData()


# Model -------------------------------------------------------------------

latent_dim <- 2

encoder_model <- function(name = NULL) {
  
  keras_model_custom(name = name, function(self) {
    self$conv1 <-
      layer_conv_2d(
        filters = 32,
        kernel_size = 3,
        strides = 2,
        activation = "relu"
      )
    self$conv2 <-
      layer_conv_2d(
        filters = 64,
        kernel_size = 3,
        strides = 2,
        activation = "relu"
      )
    self$flatten <- layer_flatten()
    self$dense <- layer_dense(units = latent_dim)
    
    function (x, mask = NULL) {
      x %>%
        self$conv1() %>%
        self$conv2() %>%
        self$flatten() %>%
        self$dense() 
    }
  })
}

decoder_model <- function(name = NULL) {
  
  keras_model_custom(name = name, function(self) {
    self$dense <- layer_dense(units = 7 * 7 * 32, activation = "relu")
    self$reshape <- layer_reshape(target_shape = c(7, 7, 32))
    
    self$deconv1 <-
      layer_conv_2d_transpose(
        filters = 64,
        kernel_size = 3,
        strides = 2,
        padding = "same",
        activation = "relu"
      )
    self$deconv2 <-
      layer_conv_2d_transpose(
        filters = 32,
        kernel_size = 3,
        strides = 2,
        padding = "same",
        activation = "relu"
      )
    self$deconv3 <-
      layer_conv_2d_transpose(
        filters = 1,
        kernel_size = 3,
        strides = 1,
        padding = "same",
        activation = "sigmoid"
      )
    
    function (x, mask = NULL) {
      x %>%
        self$dense() %>%
        self$reshape() %>%
        self$deconv1() %>%
        self$deconv2() %>%
        self$deconv3()
    }
  })
}


# Loss and optimizer ------------------------------------------------------

optimizer <- tf$train$AdamOptimizer(1e-4)

compute_kernel <- function(x, y) {
  x_size <- k_shape(x)[1]
  y_size <- k_shape(y)[1]
  dim <- k_shape(x)[2]
  tiled_x <- k_tile(k_reshape(x, k_stack(list(x_size,  k_cast(1,tf$int32), dim))), k_stack(list(k_cast(1,tf$int32), y_size, k_cast(1,tf$int32))))
  tiled_y <- k_tile(k_reshape(y, k_stack(list(k_cast(1,tf$int32), y_size, dim))), k_stack(list(x_size, k_cast(1,tf$int32), k_cast(1,tf$int32))))
  k_exp(-k_mean(k_square(tiled_x - tiled_y), axis = 3) / k_cast(dim, tf$float32))
}

compute_mmd <- function(x, y, sigma_sqr = 1) {
  x_kernel <- compute_kernel(x, x)
  y_kernel <- compute_kernel(y, y)
  xy_kernel <- compute_kernel(x, y)
  k_mean(x_kernel) + k_mean(y_kernel) - 2 * k_mean(xy_kernel)
}


# Training loop -----------------------------------------------------------

num_epochs <- 50

encoder <- encoder_model()
decoder <- decoder_model()

checkpoint_dir <- "./checkpoints_fashion_cvae_mmd"
checkpoint_prefix <- file.path(checkpoint_dir, "ckpt")
checkpoint <-
  tf$train$Checkpoint(optimizer = optimizer,
                      encoder = encoder,
                      decoder = decoder)

TrainModel<- function(encoder, 
                      decoder, 
                      checkpoint,
                      checkpoint_prefix,
                      compute_mmd,
                      latent_dim = NULL,
                      generateShinyInputs = F){
  
  generator(0,64,2)
  if(latent_dim==2) plotLatentSpace(0)
  plotGrid(0)
  
  
  for (epoch in seq_len(num_epochs)) {
    iter <- make_iterator_one_shot(train_dataset)
    
    total_loss <- 0
    loss_nll_total <- 0
    loss_mmd_total <- 0
    
    until_out_of_range({
      x <-  iterator_get_next(iter)
      
      with(tf$GradientTape(persistent = TRUE) %as% tape, {
        
        mean <- encoder(x)
        preds <- decoder(mean)
        
        true_samples <- k_random_normal(shape = c(batch_size, latent_dim), dtype = tf$float32)
        loss_mmd <- compute_mmd(true_samples, mean)
        loss_nll <- k_mean(k_square(x - preds))
        loss <- loss_nll + loss_mmd
        
      })
      
      total_loss <- total_loss + loss
      loss_mmd_total <- loss_mmd + loss_mmd_total
      loss_nll_total <- loss_nll + loss_nll_total
      
      encoder_gradients <- tape$gradient(loss, encoder$variables)
      decoder_gradients <- tape$gradient(loss, decoder$variables)
      
      optimizer$apply_gradients(purrr::transpose(list(
        encoder_gradients, encoder$variables
      )),
      global_step = tf$train$get_or_create_global_step())
      optimizer$apply_gradients(purrr::transpose(list(
        decoder_gradients, decoder$variables
      )),
      global_step = tf$train$get_or_create_global_step())
      
    })
    
    checkpoint$save(file_prefix = checkpoint_prefix)
    
    cat(
      glue(
        "Losses (epoch): {epoch}:",
        "  {(as.numeric(loss_nll_total)/batches_per_epoch) %>% round(4)} loss_nll_total,",
        "  {(as.numeric(loss_mmd_total)/batches_per_epoch) %>% round(4)} loss_mmd_total,",
        "  {(as.numeric(total_loss)/batches_per_epoch) %>% round(4)} total"
      ),
      "\n"
    )
    
    if (epoch %% 10 == 0) {
      generator(epoch,64,2)
      if (latent_dim==2) plotLatentSpace(epoch)
      plotGrid(epoch)
    }
  }
  
  if(generateShinyInputs){
    
    for(i in 1:100){ generator(epoch,1,2,filename = "shiny_generated_") }
  
  }
  
}

TrainModel(encoder = encoder, 
           decoder = decoder, 
           checkpoint = checkpoint,
           compute_mmd = compute_mmd,
           latent_dim = latent_dim,
           generateShinyInputs = F)

