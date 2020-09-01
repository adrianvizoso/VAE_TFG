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
metric_loss_nll <- 'nll_loss'
metric_loss_mmd <- 'mmd_loss'

with(tf$summary$create_file_writer("logs/hparam_tuning/mmd_vae/")$as_default(), {
  hp$hparams_config(
    hparams = list(hp_latent_dim, 
                   hp_optimizer,
                   hp_learning_rate),
    metrics = list(hp$Metric(metric_loss, display_name = "VAE Loss"),
                   hp$Metric(metric_loss_nll, display_name = "NLL Loss"),
                   hp$Metric(metric_loss_mmd, display_name = "MMD Loss"))
  )
})



mnist <- keras::dataset_mnist()
c(train_images, train_labels) %<-% mnist$train
c(test_images, test_labels) %<-% mnist$test

train_x <-
  train_images %>% `/`(255) %>% k_reshape(c(60000, 28, 28, 1)) %>% k_cast(dtype='float32')

test_x <-
  test_images %>% `/`(255) %>% k_reshape(c(10000, 28, 28, 1)) %>% k_cast(dtype='float32')

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

train_test_model <- function(hparams,run_dir) {
  
  # Hyperparameter search variable definition --------------------------------------------------
  
  
  latent_dim <- py_to_r(hparams[hp_latent_dim])
  num_epochs <- 50
  opt <- py_to_r(hparams[hp_optimizer])
  learning_rate <- py_to_r(hparams[hp_learning_rate])
  
  
  # Model -------------------------------------------------------------------
  
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
      tape$watch(loss_mmd)
      tape$watch(loss_nll)
      tape$watch(loss)
      
      total_loss <- total_loss + loss
      loss_mmd_total <- loss_mmd + loss_mmd_total
      loss_nll_total <- loss_nll + loss_nll_total
      
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
      tf$summary$scalar(metric_loss_mmd, loss_mmd_total, step = as.integer(epoch))
      tf$summary$scalar(metric_loss_nll, loss_nll_total, step = as.integer(epoch))
      tf$summary$flush
    })
    
    
  }
  
  
}


run <- function(run_dir, hparams) {
  
  train_test_model(hparams,run_dir)
  
}

# Batch and train size -------------------------------------------------------------------
session_num <- 0
batch_size <- 100
buffer_size <- 60000
batches_per_epoch <- buffer_size / batch_size

train_dataset <- tfdatasets::tensor_slices_dataset(train_x) %>%
  tfdatasets::dataset_shuffle(buffer_size) %>%
  tfdatasets::dataset_batch(batch_size)

test_dataset <- tensor_slices_dataset(test_x) %>%
  dataset_batch(10000)

for(latent_dim in hp_latent_dim$domain$values) {
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
      run(paste0("logs/hparam_tuning/mmd_vae/", run_name), hparams)
      session_num <- session_num + 1
    }
  }
  
}

tensorboard("logs/hparam_tuning/mmd_vae/")
