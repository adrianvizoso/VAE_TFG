# Output utilities --------------------------------------------------------

show_latent_space <- function(epoch) {
  iter <- make_iterator_one_shot(test_dataset)
  x <-  iterator_get_next(iter)
  x_test_encoded <- encoder(x)[[1]]
  x_test_encoded %>%
    as.matrix() %>%
    as.data.frame() %>%
    mutate(class = class_names[fashion$test$y + 1]) %>%
    ggplot(aes(x = V1, y = V2, colour = class)) + geom_point() +
    theme(aspect.ratio = 1) +
    theme(plot.margin = unit(c(0, 0, 0, 0), "null")) +
    theme(panel.spacing = unit(c(0, 0, 0, 0), "null"))
  
  ggsave(
    paste0("cvae_latentspace_epoch_", epoch, ".png"),
    width = 10,
    height = 10,
    units = "cm"
  )
}

show_grid <- function(epoch) {
  png(paste0("cvae_grid_epoch_", epoch, ".png"))
  par(mar = c(0.5, 0.5, 0.5, 0.5),
      xaxs = 'i',
      yaxs = 'i')
  n <- 16
  img_size <- 28
  
  grid_x <- seq(from = -4,to =  4, length.out = n)
  grid_y <- seq(-4, 4, length.out = n)
  rows <- NULL
  
  for (i in 1:length(grid_x)) {
    column <- NULL
    for (j in 1:length(grid_y)) {
      z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 20)
      column <-
        rbind(column,
              (matrix(decoder(k_cast(z_sample,'float32')) %>% tf$nn$sigmoid() %>% as.numeric(),nrow=img_size)))
    }
    rows <- cbind(rows, column)
  }
  rows %>% as.raster() %>% plot()
  dev.off()
  N}


generate_random_clothes <- function(epoch) {
  
  num_examples_to_generate <- 64
  latent_dim <- 2
  random_vector_for_generation <-
    k_random_normal(shape = list(num_examples_to_generate, latent_dim),
                    dtype = tf$float32)
  predictions <-
    decoder(random_vector_for_generation) %>% 
    tf$nn$sigmoid() %>%
    k_reshape(shape = c(num_examples_to_generate,28,28))
  
  
  
  png(paste0("cvae_clothes_epoch_", epoch, ".png"))
  par(mfcol = c(8, 8))
  par(mar = c(0.5, 0.5, 0.5, 0.5),
      xaxs = 'i',
      yaxs = 'i')
  for (i in 1:64) {
    img <- predictions[i, ,] %>% as.numeric() %>% matrix(nrow=28)
    img <- t(apply(img, 2, rev))
    image(
      1:28,
      1:28,
      img * 127.5 + 127.5,
      col = gray((0:255) / 255),
      xaxt = 'n',
      yaxt = 'n'
    )
  }
  dev.off()
}
