library(tensorflow)
library(keras)
library(reticulate)
library(tfdatasets)
library(tidyverse)
library(glue)

tf$keras$backend$set_floatx('float32')
source('utils.R')


# Obtenció de dades i creacio del split train / test ----------------------------------

mnist <- keras::dataset_mnist()

GetData <- function(dataset){
  
  c(train_images, train_labels) %<-% mnist$train
  c(test_images, test_labels) %<-% mnist$test
  
  train_x <-
    train_images %>% `/`(255) %>% 
    k_reshape(c(nrow(train_images), 784))%>% 
    k_cast(dtype='float32')
  
  test_x <-
    test_images %>% `/`(255) %>% 
    k_reshape(c(nrow(test_images), 784)) %>% 
    k_cast(dtype='float32')
  
  buffer_size <- 60000
  batch_size <- 6000
  batches_per_epoch <<- buffer_size / batch_size
  
  train_dataset <- tfdatasets::tensor_slices_dataset(train_x) %>%
    tfdatasets::dataset_shuffle(buffer_size) %>%
    tfdatasets::dataset_batch(batch_size)
  
  test_dataset <- tensor_slices_dataset(test_x) %>%
    dataset_batch(10000)
  
  return(list(train_dataset = train_dataset, 
              test_dataset = test_dataset))
  
}

c(train_dataset,test_dataset) %<-% GetData(dataset = mnist)


# Model -------------------------------------------------------------------

latent_dim <- 20

encoder_model <- function(x){
  
  keras_model_custom(name = 'encoder', function(self) {
    
    self$dense1 <- layer_dense(input_shape = c(NULL,784),
                               units = 512,
                               activation = "relu")
    
    self$dense2 <- layer_dense(units = 512,
                               activation = "relu")
    
    self$dense3 <- layer_dense(units = 512,
                               activation = "relu")
    
    self$mu_sigma <- layer_dense(units = 2 * latent_dim,
                                 activation = "linear")
    
    function (x, mask = NULL) {
      x %>%
        self$dense1() %>%
        self$dense2() %>%
        self$dense3() %>%
        self$mu_sigma() %>%
        tf$split(num_or_size_splits = 2L, axis = 1L)
    }
  })
  
}

decoder_model <- function(name = NULL) {
  keras_model_custom(name = name, function(self) {
    
    self$dense1 <- layer_dense(input_shape = c(NULL,latent_dim),
                               units = 512,
                               activation = "relu")
    
    self$dense2 <- layer_dense(units = 512,
                               activation = "relu")
    
    self$dense3 <- layer_dense(units = 784,
                               activation = "relu")
    
    
    
    function (x, mask = NULL) {
      x %>%
        self$dense1()%>%
        self$dense2()%>%
        self$dense3()
    }
  })
}

reparameterize <- function(mean, logvar) {
  eps <- k_random_normal(shape = mean$shape, dtype = tf$float32)
  eps * k_exp(logvar * 0.5) + mean
}


# Loss and optimizer ------------------------------------------------------

vae_loss <- function(x,preds,mean,logvar){
  
  K <- mean$shape[1]
  
  kl <- tf$reduce_sum(tf$exp(logvar))+tf$reduce_sum(tf$square(mean))-tf$constant(K,dtype = 'float32')-tf$reduce_sum(logvar)
  kl <- 0.5*tf$reduce_mean(kl)
  
  expectation <- tf$reduce_mean(tf$reduce_sum(tf$square(tf$squeeze(x)-tf$squeeze(preds))))
  
  loss <- kl + expectation
  return(loss)
}

optimizer <- tf$compat$v1$train$AdamOptimizer(learning_rate = 1e-4)



# Entrenament del model -----------------------------------------------------------

num_epochs <- 50

encoder <- encoder_model()
decoder <- decoder_model()

checkpoint_dir <- "./checkpoints_standard_vae"
checkpoint_prefix <- file.path(checkpoint_dir, "ckpt")
checkpoint <-
  tf$train$Checkpoint(optimizer = optimizer,
                      encoder = encoder,
                      decoder = decoder)

TrainModel<- function(encoder = encoder, 
                      decoder = decoder, 
                      checkpoint = checkpoint,
                      checkpoint_prefix = checkpoint_prefix,
                      reparameterize = reparameterize,
                      vae_loss = vae_loss,
                      latent_dim = NULL,
                      generateShinyInputs = F){
  
  generator(epoch = 0,
            num_examples_to_generate = 64,
            filename = "vae_generated_epoch_",
            latent_dim = latent_dim)
  
  if(latent_dim==2) plotLatentSpace(epoch = 0,
                                    filename = 'vae_grid_epoch_')
  plotGrid(epoch = 0,
           latent_dim = latent_dim,
           filename = 'vae_grid_epoch_')
  
  
  for (epoch in 1:num_epochs) {
    iter <- make_iterator_one_shot(train_dataset)
    
    total_loss <- 0
    
    until_out_of_range({
      x <-  iterator_get_next(iter)
      
      with(tf$GradientTape(persistent = TRUE) %as% tape, {
        c(mean, logvar) %<-% encoder(x)
        z <- reparameterize(mean, logvar)
        preds <- decoder(z)
        loss <- vae_loss(x,preds,mean,logvar)
        
      })
      
      tape$watch(loss)
      
      total_loss <- total_loss + loss
      
      encoder_gradients <- tape$gradient(loss, encoder$variables)
      decoder_gradients <- tape$gradient(loss, decoder$variables)
      
      optimizer$apply_gradients(purrr::transpose(list(
        encoder_gradients, encoder$variables
      )),
      global_step = tf$compat$v1$train$get_or_create_global_step())
      
      optimizer$apply_gradients(purrr::transpose(list(
        decoder_gradients, decoder$variables
      )),
      global_step = tf$compat$v1$train$get_or_create_global_step())
      
    })
    
    # checkpoint$save(file_prefix = checkpoint_prefix)
    
    cat(
      glue(
        "Losses (epoch): {epoch}:",
        "  {(as.numeric(total_loss)/batches_per_epoch) %>% round(2)} total"
      ),
      "\n"
    )
    
    if (epoch %% 10 == 0) {
      
      generator(epoch = epoch,
                num_examples_to_generate = 64,
                filename = "vae_generated_epoch_",
                latent_dim = latent_dim)
      
      if (latent_dim==2) plotLatentSpace(epoch = epoch, filename = "vae_latent_space_epoch_")
      
      plotGrid(epoch = epoch, 
               latent_dim = latent_dim, 
               filename = "vae_grid_epoch_")
      
    }
  }
  
  if(generateShinyInputs){
    
    for(i in 1:100){ generator(epoch = i,
                               num_examples_to_generate = 1,
                               toShiny = T,
                               latent_dim = latent_dim,
                               filename = "sample_") }
    
  }
}

TrainModel(encoder = encoder, 
           decoder = decoder, 
           checkpoint = checkpoint,
           checkpoint_prefix = checkpoint_prefix,
           reparameterize = reparameterize,
           vae_loss = vae_loss,
           latent_dim = latent_dim,
           generateShinyInputs = F)






