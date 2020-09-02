library(tensorflow)
library(keras)
library(reticulate)
library(tfdatasets)
library(tidyverse)
library(glue)


hp <- import("tensorboard.plugins.hparams.api")
tf$keras$backend$set_floatx('float32')

hp_latent_dim <- hp$HParam('latent_dim', hp$Discrete(list(2,20,50)))
hp_optimizer <- hp$HParam('optimizer', hp$Discrete(list('rmsprop','adam', 'adadelta')))
hp_learning_rate <- hp$HParam('learning_rate', hp$Discrete(list(0.0001,0.00075)))

metric_loss <- 'vae_loss'

with(tf$summary$create_file_writer("logs/hparam_tuning/standard_vae/")$as_default(), {
  hp$hparams_config(
    hparams = list(hp_latent_dim, 
                   hp_optimizer,
                   hp_learning_rate),
    metrics = list(hp$Metric(metric_loss, display_name = "VAE Loss"))
  )
})



mnist <- keras::dataset_mnist()
c(train_images, train_labels) %<-% mnist$train
c(test_images, test_labels) %<-% mnist$test

train_x <-
  train_images %>% `/`(255) %>% k_reshape(c(nrow(train_images), 784))%>% k_cast(dtype='float32')

test_x <-
  test_images %>% `/`(255) %>% k_reshape(c(nrow(test_images), 784)) %>% k_cast(dtype='float32')


reparameterize <- function(mean, logvar) {
  eps <- k_random_normal(shape = mean$shape, dtype = tf$float32)
  eps * k_exp(logvar * 0.5) + mean
}

vae_loss <- function(x,preds,mean,logvar){
  
  K <- mean$shape[1]
  
  kl <- tf$reduce_sum(tf$exp(logvar))+tf$reduce_sum(tf$square(mean))-tf$constant(K,dtype = 'float32')-tf$reduce_sum(logvar)
  kl <- 0.5*tf$reduce_mean(kl)
  
  expectation <- tf$reduce_mean(tf$reduce_sum(tf$square(tf$squeeze(x)-tf$squeeze(preds))))
  
  loss <- kl + expectation
  return(loss)
}


train_test_model <- function(hparams,run_dir) {
  
  # Hyperparameter search variable definition --------------------------------------------------
  
  
  latent_dim <- py_to_r(hparams[hp_latent_dim])
  num_epochs <- 50
  opt <- py_to_r(hparams[hp_optimizer])
  learning_rate <- py_to_r(hparams[hp_learning_rate])
  
  
  # Model -------------------------------------------------------------------
  
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
  
  encoder <- encoder_model()
  decoder <- decoder_model()
  
  if( opt == "adam" ) {
    
    optimizer <- tf$compat$v1$train$AdamOptimizer(learning_rate = learning_rate)
    
    
  } else if( opt == "adadelta" ) {
    
    optimizer <- tf$compat$v1$train$AdadeltaOptimizer(learning_rate = learning_rate)
    
  } else {
    
    optimizer <- tf$compat$v1$train$RMSPropOptimizer(learning_rate = learning_rate)
    
  }
  
  for (epoch in seq_len(num_epochs)) {
    
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
    
    cat(
      glue(
        "Losses (epoch): {epoch}:",
        "  {(as.numeric(total_loss)/batches_per_epoch) %>% round(2)} total"
      ),
      "\n"
    )
    
    with(tf$summary$create_file_writer(run_dir)$as_default(), {
      hp$hparams(hparams) # record the values used in this trial
      tf$summary$scalar(metric_loss, total_loss, step = as.integer(epoch))
      tf$summary$flush
    })
    
    
  }
  
  cat(
    glue(
      "\t Saving image to check results"
    ),
    "\n"
  )
  
  # with(tf$summary$create_file_writer(run_dir)$as_default(), {
  #   hp$hparams(hparams) # record the image generated in this run
  #   tf$summary$image("digits_generated", step = as.integer(1), data = hypersearchGenerator(num_examples_to_generate = 5,
  #                                                                                          latent_dim = latent_dim))
  #   tf$summary$flush
  # })
  
}


run <- function(run_dir, hparams) {
  
  train_test_model(hparams,run_dir)
  
}

# Batch and train size -------------------------------------------------------------------
session_num <- 7
batch_size <- 100
buffer_size <- 60000
batches_per_epoch <- buffer_size / batch_size


train_dataset <- tfdatasets::tensor_slices_dataset(train_x) %>%
  tfdatasets::dataset_shuffle(buffer_size) %>%
  tfdatasets::dataset_batch(batch_size)

test_dataset <- tensor_slices_dataset(test_x) %>%
  dataset_batch(10000)

for(latent_dim in hp_latent_dim$domain$values[2:3]) {
  for (optimizer in hp_optimizer$domain$values) {
    for (learning_rate in hp_learning_rate$domain$values) { 
      
      hparams <- dict(
        hp_latent_dim = latent_dim,
        hp_learning_rate = learning_rate,
        hp_optimizer = optimizer
      )
      
      run_name <- sprintf("run-%04d", session_num)
      print(sprintf('--- Starting trial: %s',  run_name))
      #purrr::iwalk(hparams, ~print(paste(.y, .x, sep = ": ")))
      run(paste0("logs/hparam_tuning/standard_vae/", run_name), hparams)
      session_num <- session_num + 1
    }
  }
  
}

tensorboard("logs/hparam_tuning/standard_vae/")
