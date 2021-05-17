library(reticulate)
library(tidyr)
library(purrr)
library(dplyr)
library(ggplot2)
library(patchwork)

pd <- import("pandas")

history_names <- c("hist1.pickle", 
                   "hist2.pickle", 
                   "conv_layers_test_hist.pickle", 
                   "bilstm_test_hist.pickle", 
                   "bilstm_size_test_hist.pickle",
                   "bilstm_n_layers_test_hist.pickle")
history_pickles <- map(paste0("data/", history_names), function(hist) pd$read_pickle(hist))
tested_parameters <- c("dropout rate", 
                       "kernel size", 
                       "number of conv layers", 
                       "type of layer",
                       "size of layers",
                       "number of layers")
tested_parameter_values <- list(
  c(0, 0.2, 0.4),
  c(5, 15, 10),
  c(2, 1),
  c("BILSTM", "GRU", "LSTM"),
  c(64, 256, 128),
  c(1, 3, 2)
)

histories <- map(1:6, function(i) {
  map2_dfr(history_pickles[[i]], rep(tested_parameter_values[[i]], each = 3), 
           function(df, tested_parameter_value) {
    df %>%
      as.data.frame %>%
      mutate(epoch = cur_group_rows()) %>%
      pivot_longer(cols = -epoch, names_to = "measure_name", values_to = "measure_value") %>%
      mutate(tetsted_parameter_name = !!tested_parameters[[i]],
             tested_parameter_value = factor(tested_parameter_value))
  })
})

plots <- map2(histories, tested_parameters, function(history, parameter) {
  subplots <- map2(c("loss", "accuracy", "val_loss", "val_accuracy"),
                   c("Loss on training set",
                     "Accuracy on training set",
                     "Loss on validation set",
                     "Accuracy on validation set"), function(measure, measure_disp_name) {
    history %>%
      filter(measure_name == measure) %>%
      ggplot(aes(x = epoch, 
                 y = measure_value, 
                 group = tested_parameter_value, 
                 color = tested_parameter_value)) +
      geom_smooth() +
      ggtitle(measure_disp_name) +
      guides(color = guide_legend(title = parameter)) +
      theme_minimal()
  })
  plt <- (subplots[[1]] + subplots[[3]]) / (subplots[[2]] + subplots[[4]]) +
    plot_annotation(parameter) +
    plot_layout(guides = "collect", ) &
    theme(legend.position = "bottom")
  ggsave(paste0("plots/", sub(" ", "_", parameter), ".png"), plt, "png", width = 12, height = 9)
})
