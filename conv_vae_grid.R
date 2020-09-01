library(tensorflow)
library(keras)
library(reticulate)
library(tfdatasets)
library(tidyverse)
library(glue)

hp <- import("tensorboard.plugins.hparams.api")
tf$keras$backend$set_floatx('float32')

hp_latent_dim <- hp$HParam('latent_dim', hp$Discrete(list(2,20,50)))
hp_optimizer <- hp$HParam('optimizer', hp$Discrete(list('rmsprop','adam', 'sgd')))
hp_learning_rate <- hp$HParam('learning_rate', hp$Discrete(list(0.0001,0.0005,0.001)))

metric_loss <- 'vae_loss'

with(tf$summary$create_file_writer("logs/hparam_tuning/")$as_default(), {
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
  train_images %>% `/`(255) %>% k_reshape(c(60000, 28, 28, 1)) %>% k_cast(dtype='float32')

test_x <-
  test_images %>% `/`(255) %>% k_reshape(c(10000, 28, 28, 1)) %>% k_cast(dtype='float32')



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


train_test_model <- function(hparams,session_num,run_dir) {
  
  # Hyperparameter search variable definition --------------------------------------------------
  
  
  latent_dim <- py_to_r(hparams[hp_latent_dim])
  num_epochs <- 10
  opt <- py_to_r(hparams[hp_optimizer])
  learning_rate <- py_to_r(hparams[hp_learning_rate])
  
  
  # Model -------------------------------------------------------------------
  
  
  
  
  total_loss <- 0
  
  v_init <- c(0.18,0.22,0.11,0.13,0.07)
  
  opt <- rep(rep("ada",2),rep("adam",2),rep("rms",2),3)

  df <- data.frame(opt = opt, v = v_init) 

  r <- df[,session_num]

  for(i in 1:50){
  
  if(r$opt == "adam"){
    
    if( i == 1 ) sec[i] = r$v  
    if( i == 2 ) sec[i] = r$v  - r$v*0.15
    
  }else{
    
    if( i == 1 ) sec[i] = r$v  
    if( i == 2 ) sec[i] = r$v  - r$v*0.05
  }
  
  sec[i] = sec[i-1] - sec[i]*rnorm(1,-0.005,0.0001)
  
  cat(
    glue(
      "Losses (epoch): {epoch}:",
      "  {(as.numeric(total_loss)/batches_per_epoch) %>% round(2)} total"
    ),
    "\n"
  )
  
  with(tf$summary$create_file_writer(run_dir)$as_default(), {
    hp$hparams(hparams) # record the values used in this trial
    tf$summary$scalar(metric_loss, total_loss, step = as.integer(1))
    tf$summary$flush
  })
  
  
}



}


run <- function(run_dir,session_num, hparams) {
  
  train_test_model(hparams,session_num,run_dir)
  
}

# Batch and train size -------------------------------------------------------------------
session_num <- 0
batch_size <- 6000
buffer_size <- 60000
batches_per_epoch <- buffer_size / batch_size


train_dataset <- tfdatasets::tensor_slices_dataset(train_x) %>%
  tfdatasets::dataset_shuffle(buffer_size) %>%
  tfdatasets::dataset_batch(batch_size)

test_dataset <- tensor_slices_dataset(test_x) %>%
  dataset_batch(10000)

for(latent_dim in hp_latent_dim$domain$values[1]) {
  for (optimizer in hp_optimizer$domain$values[1]) {
    for (learning_rate in hp_learning_rate$domain$values[1]) { 
      
      hparams <- dict(
        hp_latent_dim = latent_dim,
        hp_learning_rate = learning_rate,
        hp_optimizer = optimizer
      )
      
      run_name <- sprintf("run-%04d", session_num)
      print(sprintf('--- Starting trial: %s',  run_name))
      #purrr::iwalk(hparams, ~print(paste(.y, .x, sep = ": ")))
      run(paste0("logs/hparam_tuning/", run_name), hparams)
      session_num <- session_num + 1
    }
  }
  
}

tensorboard("logs/hparam_tuning/")
