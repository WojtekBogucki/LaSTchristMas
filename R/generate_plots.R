library(reticulate)
library(tidyr)
library(purrr)
library(dplyr)
library(ggplot2)

pd <- import("pandas")

hist1 <- pd$read_pickle("hist1.pickle")
hist2 <- pd$read_pickle("hist2.pickle")

dropout_values <- c(0, 0.2, 0.4)

hist1[[1]] %>%
  as.data.frame %>%
  mutate(epoch = cur_group_rows()) %>%
  pivot_longer(cols = -epoch, names_to = "measure") %>%
  mutate(dropout = dropout_value)


hist1_tf <- map2_dfr(hist1, rep(c(0, 0.2, 0.4), each = 3), function(df, dropout_value) {
  df %>%
    as.data.frame %>%
    mutate(epoch = cur_group_rows()) %>%
    pivot_longer(cols = -epoch, names_to = "measure") %>%
    mutate(dropout = dropout_value)
})

hist2_tf <- map2_dfr(hist2, rep(c(5, 15, 10), each = 3), function(df, kernel_size_value) {
  df %>%
    as.data.frame %>%
    mutate(epoch = cur_group_rows()) %>%
    pivot_longer(cols = -epoch, names_to = "measure") %>%
    mutate(kernel_size = kernel_size_value)
})


hist1_tf %>%
  filter(measure == "loss") %>%
  ggplot(aes(x = epoch, y = value, group = factor(dropout), color = factor(dropout))) +
  geom_smooth() +
  theme_minimal()

hist1_tf %>%
  filter(measure == "val_loss") %>%
  ggplot(aes(x = epoch, y = value, group = factor(dropout), color = factor(dropout))) +
  geom_smooth() +
  theme_minimal()

hist2_tf %>%
  filter(measure == "loss") %>%
  ggplot(aes(x = epoch, y = value, group = factor(kernel_size), color = factor(kernel_size))) +
  geom_smooth(alpha = 0.1) +
  theme_minimal()

hist2_tf %>%
  filter(measure == "accuracy") %>%
  ggplot(aes(x = epoch, y = value, group = factor(kernel_size), color = factor(kernel_size))) +
  geom_smooth(alpha = 0.1) +
  theme_minimal()
  