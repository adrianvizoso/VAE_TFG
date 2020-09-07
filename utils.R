# Output utilities --------------------------------------------------------

plotLatentSpace<- function(epoch,
                           MMD = F,
                           filename = 'vae_grid_epoch_') {
  iter <- make_iterator_one_shot(test_dataset)
  x <-  iterator_get_next(iter)
  
  if(MMD){
    x_test_encoded <- encoder(x)
  }else{
    x_test_encoded <- encoder(x)[[1]]
  }
  
  p <- x_test_encoded %>%
    as.matrix() %>%
    as.data.frame() %>%
    cbind(class = as.factor(mnist$test$y)) %>%
    ggplot(aes(x = V1, y = V2, colour = class)) + geom_point() +
    theme(aspect.ratio = 1) +
    theme(plot.margin = unit(c(0, 0, 0, 0), "null")) +
    theme(panel.spacing = unit(c(0, 0, 0, 0), "null"))
  
  p + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
            panel.background = element_blank()) + scale_color_brewer(palette="RdBu")
  
  ggsave(
    paste0(filename, epoch, ".png"),
    width = 10,
    height = 10,
    units = "cm"
  )
}

plotGrid <- function(epoch,
                     latent_dim = latent_dim,
                     filename = "vae_generated_grid_") {
  png(paste0(filename, epoch, ".png"))
  par(mar = c(0.5, 0.5, 0.5, 0.5),
      xaxs = 'i',
      yaxs = 'i')
  n <- 16
  img_size <- 28
  
  grid_x <- seq(-4, 4, length.out = n)
  grid_y <- seq(-4, 4, length.out = n)
  rows <- NULL
  
  for (i in 1:length(grid_x)) {
    column <- NULL
    for (j in 1:length(grid_y)) {
      z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = latent_dim)
      column <-
        rbind(column,
              (decoder(z_sample) %>% tf$nn$sigmoid() %>% as.numeric()) %>% matrix(ncol = img_size))
    }
    rows <- cbind(rows, column)
  }
  rows %>% as.raster() %>% plot()
  dev.off()
}


generator <- function(epoch,
                      num_examples_to_generate,
                      toShiny = F,
                      filename = "vae_generated_epoch_",
                      latent_dim) {
  
  random_vector_for_generation <-
    k_random_normal(shape = list(num_examples_to_generate, latent_dim),
                    dtype = tf$float32)
  predictions <-
    decoder(random_vector_for_generation) %>%  tf$nn$sigmoid() %>%
    k_reshape(shape = c(num_examples_to_generate,28,28))
  
  png(paste0(filename, epoch, ".png"))
  #png(paste0("sample_", epoch, ".png"))
  if(toShiny){
    par(mfcol = c(1, 1))
  }else{
    par(mfcol = c(8, 8))
  }
  par(mar = c(0.5, 0.5, 0.5, 0.5),
      xaxs = 'i',
      yaxs = 'i')
  for (i in 1:num_examples_to_generate) {
    img <- predictions[i, ,] %>% as.numeric() %>% matrix(nrow=28)
    img <- t(apply(img, 2, rev))
    img <- prepareImgToPlot(img)
    image(
      1:28,
      1:28,
      img * 127.5 + 127.5,
      col = gray((1:255) / 255),
      xaxt = 'n',
      yaxt = 'n'
    )
  }
  dev.off()
}


hypersearchGenerator <- function(num_examples_to_generate,
                                 latent_dim) {
  
  random_vector_for_generation <-
    k_random_normal(shape = list(num_examples_to_generate, latent_dim),
                    dtype = tf$float32)
  predictions <-
    decoder(random_vector_for_generation) %>%  tf$nn$sigmoid() %>%
    k_reshape(shape = c(num_examples_to_generate,28,28))
  
  imgs <- tf$Variable(tf$zeros(shape = c(nrow(predictions),28,28,1), 
                               dtype=tf$dtypes$float32))
  
  for(i in 1:nrow(predictions)){
    img <- matrix(as.array(predictions[i,,]),ncol=28)
    img <- t(apply(img, 2, rev))
    img <- prepareImgToPlot(img)
    
    img <- tf$Variable(tf$constant(img,dtype=tf$dtypes$float32)) %>% 
      k_reshape(shape = c(28,28,1))
    
    imgs[i,,,]$assign(img)
  }
  
  return(imgs)
  
}


prepareImgToPlot <- function(img){
  
  
  for(i in 1:nrow(img)){
    
    start <- head(which(img[i,]!=0),1)
    
    end <- tail(which(img[i,]!=0),1)
    
    if(length(start)!= 0 & length(end)!= 0){
      
      if(any(img[i,seq(start,end)]==0)){
        
        to.scale <- start + which(img[i,seq(start,end)]==0) - 1
        
        for(j in to.scale){
          
          val <- mean(img[i,j-1],img[i,j+1])*0.75
          
          if(val<0.15){val <- 0}
          
          img[i,j] <- val
          
        }
        
        
      }
      
    }
    
  }
  
  return(img)
  
}
