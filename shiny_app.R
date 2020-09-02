library(shinydashboard)
library(dashboardthemes)
library(shiny)


ui <- dashboardPage(
  dashboardHeader(title = "Resultats dels models",titleWidth = 200),
  dashboardSidebar(disable = TRUE),
  dashboardBody(
    
    # Apply theme 
    shinyDashboardThemes(
      theme = "poor_mans_flatly"
    ),
    
    # Boxes need to be put in a row (or column)
    fluidRow(
      
      column(8, align="center",
             selectInput("model", "Model escollit:",
                         c("VAE" = "standard_vae",
                           "CVAE" = "cvae",
                           "MMD-CVAE" = "mmd_vae"),selected = "standard_vae")),
      
      column(8, align="center",
             box(
               title = "Nombres generats pel model",
               sliderInput("slider1", "Nombre de epochs:", 0, 50, 0,step = 10),
             )),
      
      column(8, align="center",
             
             uiOutput("img1", height = 250),style="margin-bottom:50px;"
      ),
      
      column(8, align="center",
             box(
               title = "Espai latent",
               sliderInput("slider2", "Nombre de epochs:", 0, 50, 0,step = 10),
             )),
      
      column(8, align="center",
             uiOutput("img2", height = 250),style="margin-bottom:50px;"
      ),
      
      column(8, align="center",
             box(
               title = "Grid",
               sliderInput("slider3", "Nombre de epochs:", 0, 50, 0,step = 10),
             )),
      
      column(8, align="center",
             
             uiOutput("img3", height = 250),style="margin-bottom:50px;"
      ),
      
      column(8, align="center",
             box(
               uiOutput("img_generator", height = 250)
             )),
      column(8, align="center",
             
             actionButton("button", "Genera nou digit"))
      
    )
  )
)

server <- function(input, output) {
  
  output$out <- renderText({
    dir <- paste0(getwd(),"/",as.character(input$model))
    
    dir.to.plot.1 <- paste0(dir,"/cvae_grid_epoch_",input$slider1,".png")
  })
  
  
  output$img1 <- renderUI({
    
    dir <- paste0(getwd(),"/",as.character(input$model))
    
    dir.to.plot.1 <- paste0(dir,"/cvae_generated_epoch_",input$slider1,".png")
    
    b64 <- base64enc::dataURI(file=dir.to.plot.1, mime="image/png")
    
    img(height = 240, width = 300, src = b64)
    
    
  })
  
  
  output$img2 <- renderUI({
    
    dir <- paste0(getwd(),"/",as.character(input$model))
    
    dir.to.plot.2 <- paste0(dir,"/cvae_latentspace_epoch_",input$slider2,".png")
    
    b64 <- base64enc::dataURI(file=dir.to.plot.2, mime="image/png")
    
    img(height = 240, width = 300, src = b64)
    
    
  })
  
  output$img3 <- renderUI({
    
    dir <- paste0(getwd(),"/",as.character(input$model))
    
    dir.to.plot.3 <- paste0(dir,"/cvae_grid_epoch_",input$slider3,".png")
    
    b64 <- base64enc::dataURI(file=dir.to.plot.3, mime="image/png")
    
    img(height = 240, width = 300, src = b64)
    
    
  })
  
  output$img_generator <- renderUI({
    
    dir <- paste0(getwd(),"/",as.character(input$model))
    
    r <- round(runif(1,min=1,max=200))
    
    if(input$button){ r <- round(runif(1,min=1,max=200)) }
    
    dir.to.plot.generator <- paste0(dir,"/random_outputs/sample_",r,".png")
    
    b64 <- base64enc::dataURI(file=dir.to.plot.generator, mime="image/png")
    
    img(height = 240, width = 300, src = b64)
    
  })
  
  
  
  
  
}

shinyApp(ui = ui, server = server)
