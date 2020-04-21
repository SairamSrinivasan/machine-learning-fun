FLAGS <- flags(
  flag_numeric("nodes_input", 500),
  flag_numeric("nodes_hidden_one", 60),
  flag_numeric("nodes_hidden_two", 23),
  flag_string("activation_input", "relu"),
  flag_string("activation_one", "relu"),
  flag_string("activation_two", "relu"),
  flag_numeric("batch_size", 100),
  flag_string("layer_dropout", 0.5),
  flag_numeric("output_dim", 1),
  flag_numeric("learning_rate", 0.01),
  flag_numeric("epochs", 30)
)

inp_region <- layer_input(shape = c(1), name="region")

inp_other <- layer_input(shape = c(12), name="other")

embedding_region <- inp_region %>% layer_embedding(input_dim = 54 + 1, output_dim = 20, input_length = 1, name = "region_embedding") %>% layer_flatten()

merged_model <- layer_concatenate(c(embedding_region, inp_other)) %>% layer_dense(units = FLAGS$nodes_input, activation = FLAGS$activation_input) %>% layer_dropout(0.5) %>% layer_dense(units = FLAGS$nodes_hidden_one, activation = FLAGS$activation_one) %>% layer_dropout(0.5) %>% layer_dense(units = FLAGS$nodes_hidden_two, activation = FLAGS$activation_two) %>% layer_dropout(0.5) %>% layer_dense(units = 1)

nnet_model_with_embedding <- keras::keras_model(inputs = c(inp_region, inp_other), outputs = merged_model)

nnet_model_with_embedding %>% compile(loss = "mse", optimizer = optimizer_adam(lr = FLAGS$learning_rate))

input_tune_predictors <- list(as.matrix(avocado_tune_labels$region), as.matrix(avocado_tune_labels[not_embedded_columns_tune]))

input_val_predictors <- list(as.matrix(avocado_val_labels$region), as.matrix(avocado_val_labels[not_embedded_columns_tune]))

history <- nnet_model_with_embedding %>% fit(input_tune_predictors, avocado_tune_outcome, epochs = FLAGS$epochs, batch_size = FLAGS$batch_size, validation_data = list(input_val_predictors, avocado_val_outcome))