# mbc.R – Functions for Multi-dimensional Bayesian Classifier (MBC)

# Load required libraries (make sure these are installed in R):
library(bnlearn)
# gRain is optional, used for exact inference in some functions
tryCatch({
  library(gRain)
}, error = function(e) {
  # gRain not available, skipping (not essential for MBC)
})
library(arules)   # for discretize if needed

# Performance evaluation for multi-dimensional classification
test_multidimensional <- function(test_set, pred_set, classes) {
  # Ensure both true and predicted sets are character (to avoid factor mismatches)
  true <- data.frame(lapply(test_set[, classes], as.character), stringsAsFactors = FALSE)
  pred <- data.frame(lapply(pred_set[, classes], as.character), stringsAsFactors = FALSE)
  match_matrix <- (true == pred)
  exact_match <- apply(match_matrix, 1, all)  # TRUE if all class labels match for a case
  per_class <- colMeans(match_matrix)         # accuracy per class
  average_accuracy <- mean(per_class)         # average accuracy across classes
  global_accuracy  <- mean(exact_match)       # global accuracy (exact match of all classes)
  return(list(global = global_accuracy, average = average_accuracy, per_class = per_class))
}

# Learn an MBC using a hill-climbing search (filter approach, maximizing BIC)
learn_MBC <- function(training_set, classes, features) {
  # Blacklist: prevent arcs from features to classes (to keep model as a classifier)
  bl <- expand.grid(from = features, to = classes)  # all feature->class combinations
  # Learn network structure with hill-climbing (BIC score by default)
  net <- hc(training_set, blacklist = as.matrix(bl))
  # Fit parameters with Bayesian estimation (Laplace smoothing)
  MBC_model <- bn.fit(net, training_set, method = "bayes", iss = 1)
  return(MBC_model)
}

# Optional: Learn an MBC using a wrapper approach (greedy addition of arcs to improve accuracy)
learn_MBC_wrapper2 <- function(training_set, validation_set, classes, features, measure = "global", verbose = FALSE) {
  # Start with an empty network (no arcs)
  MBC_best <- empty.graph(nodes = c(classes, features))
  MBC_fit <- bn.fit(MBC_best, training_set, method = "bayes", iss = 1)
  # Pre-compute joint class configurations for fast inference
  joint_levels <- lapply(classes, function(c) dimnames(MBC_fit[[c]]$prob)[[1]])
  names(joint_levels) <- classes
  class_combinations <- expand.grid(joint_levels)  # all combinations of class values
  I <- nrow(class_combinations)
  # Function to compute predictions for all validation cases given a model
  get_validation_accuracy <- function(model) {
    # Compute log-likelihoods for all class combos for each validation case
    big_matrix <- cbind(
      validation_set[rep(1:nrow(validation_set), each = I), features, drop = FALSE],
      class_combinations[rep(1:I, nrow(validation_set)), ]
    )
    logL <- logLik(model, big_matrix, by.sample = TRUE)
    # Determine MPE (most probable explanation) for each case
    max_idx <- sapply(0:(nrow(validation_set)-1), function(j) {
      offset <- j * I
      which.max(logL[(offset+1):(offset+I)])
    })
    preds <- class_combinations[max_idx, , drop = FALSE]
    # Compute chosen measure (global or average accuracy) on validation set
    perf <- test_multidimensional(validation_set, preds, classes)
    if (measure == "average") return(perf$average) else return(perf$global)
  }
  # Initial performance with no arcs
  best_perf <- get_validation_accuracy(MBC_fit)
  improved <- TRUE
  # Greedily add the single best arc until no improvement
  while (improved) {
    improved <- FALSE
    best_arc <- NULL
    candidate_arcs <- MBC_possible_arcs(classes, features)
    current_arcs <- if (!is.null(arcs(MBC_best))) arcs(MBC_best) else matrix(nrow = 0, ncol = 2)
    best_candidate_perf <- best_perf
    # Try each possible new arc
    for (k in 1:nrow(candidate_arcs)) {
      arc <- candidate_arcs[k, ]
      # Skip if arc already present or creates a cycle
      if (any(apply(current_arcs, 1, function(x) all(x == arc)))) next
      MBC_temp <- set.arc(MBC_best, from = arc["from"], to = arc["to"], check.cycles = FALSE)
      if (!acyclic(MBC_temp)) next
      # Fit parameters for the temporary model
      MBC_temp_fit <- bn.fit(MBC_temp, training_set, method = "bayes", iss = 1)
      # Evaluate performance on validation set
      perf_val <- get_validation_accuracy(MBC_temp_fit)
      if (perf_val > best_candidate_perf + 1e-10) {  # use a tiny tolerance
        best_candidate_perf <- perf_val
        best_arc <- arc
      }
    }
    # If an arc improved performance, add it permanently
    if (!is.null(best_arc)) {
      MBC_best <- set.arc(MBC_best, from = best_arc["from"], to = best_arc["to"])
      MBC_best <- model2network(modelstring(MBC_best))  # ensure no duplicates, keep structure consistent
      MBC_fit  <- bn.fit(MBC_best, training_set, method = "bayes", iss = 1)
      best_perf <- best_candidate_perf
      improved <- TRUE
      if (verbose) message(sprintf("Added arc %s -> %s, %s accuracy = %.4f", 
                                   best_arc["from"], best_arc["to"], measure, best_perf))
    }
  }
  return(MBC_fit)
}

# Helper: List all possible arcs in an MBC (between class-class, feature-feature, and class->feature)
MBC_possible_arcs <- function(classes, features) {
  nodes <- c(classes, features)
  # All directed pairs except feature->class (which we disallow)
  all_pairs <- expand.grid(from = nodes, to = nodes, stringsAsFactors = FALSE)
  all_pairs <- all_pairs[all_pairs$from != all_pairs$to, ]
  # Remove feature -> class combinations (to avoid features as parents of classes)
  bad_idx <- which(all_pairs$from %in% features & all_pairs$to %in% classes)
  if (length(bad_idx) > 0) all_pairs <- all_pairs[-bad_idx, ]
  return(all_pairs)
}

# Predict class labels for a dataset using a trained MBC (exact MPE for each instance)
predict_MBC_dataset_veryfast <- function(MBC_model, data_set, classes, features) {
  options(warn = -1)  # suppress warnings (e.g., logLik warnings for missing combos)
  # Get all possible joint class outcomes
  class_levels <- lapply(classes, function(c) dimnames(MBC_model[[c]]$prob)[[1]])
  names(class_levels) <- classes
  class_joint <- expand.grid(class_levels)
  I <- nrow(class_joint)
  # Repeat each test case I times with each possible class combination
  big_mat <- cbind(
    data_set[rep(1:nrow(data_set), each = I), features, drop = FALSE],
    class_joint[rep(1:I, nrow(data_set)), ]
  )
  # Compute log-likelihood of each combination for each case
  logL <- logLik(MBC_model, big_mat, by.sample = TRUE)
  # For each case, find which class combination maximizes log-likelihood
  max_indices <- sapply(0:(nrow(data_set)-1), function(j) {
    idx_start <- j * I + 1
    idx_end   <- j * I + I
    which.max(logL[idx_start:idx_end])
  })
  # Extract the predicted class combination for each case
  preds <- class_joint[max_indices, , drop = FALSE]
  rownames(preds) <- NULL  # remove row names
  return(preds)  # data frame with one row per case, columns = classes
}
