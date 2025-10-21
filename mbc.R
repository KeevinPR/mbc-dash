# mbc.R – Multi-dimensional Bayesian Classifier (MBC)
# ======================================================================
# PURPOSE
#   Learn a single Multi-dimensional Bayesian Classifier (MBC) and use it
#   to predict a vector of class variables Y = (C1,…,Cd) from features X.
#
# THEORY SKETCH
#   A Bayesian network (BN) encodes a joint distribution over variables V
#   via factorization along a DAG G:      p(V) = ∏_{v∈V} p(v | Pa(v))
#   with local CPT parameters θ.
#
#   An MBC is a BN over V = {X1..Xm, C1..Cd} optimized for classification
#   of the d-dimensional label Y = (C1..Cd).  We constrain directions by
#   blacklisting feature→class arcs (features cannot be parents of classes),
#   which biases the model so that classes can be parents (or peers) and
#   we classify using the MPE rule:
#         ŷ(x) = argmax_{y ∈ Ω(C1)×…×Ω(Cd)}  p(y, x ; θ, G)
#   (equivalently argmax_y p(y | x) since p(x) is constant w.r.t. y).
#
#   Structure learning (filter mode) maximizes BIC:
#         BIC(G,θ;D) = log L(θ; D, G) - (k/2) log N
#   where k = #free parameters, N = #samples, and L is the likelihood.
#   Parameter learning uses Bayesian estimation with a Dirichlet prior:
#   bnlearn’s method = "bayes", iss = 1  ⇒ Laplace smoothing (α=1),
#   i.e. posterior mean: (N_ijk + 1) / (N_ij + r_i), robust to sparsity.
#
#   Wrapper mode explicitly optimizes validation accuracy (global or
#   average) by greedily adding arcs that improve:
#      J_global = (1/N) ∑_n 1[ ŷ_n == y_n ]         (exact-match)
#      J_avg    = (1/d) ∑_{k=1}^d (1/N) ∑_n 1[ ŷ_{n,k} == y_{n,k} ]
#
#   Prediction evaluates log-likelihoods of all joint class assignments:
#      For each test case x:
#        compute ℓ(y) = log p(y, x) by BN factorization
#        pick argmax_y ℓ(y).  (MPE for discrete BN.)
#
# PRACTICAL NOTES
#   • Discretization: numeric features are optionally discretized via arules::discretize
#     (frequency/cluster, 3 bins) to keep CPTs discrete.
#   • Complexity at prediction is O(I) per instance, with I = ∏_k |Ω(Ck)|.
#     Works well for small d or small label cardinalities.
#   • Code is adapted from CIG-style MBC / MBCTree implementations; tree parts removed.
#
# REFERENCES (for attribution; not pulled at runtime):
#   • Borchani, H. (2013). Multi-dimensional classification using Bayesian networks ...
#   • Gil-Begue, S., Larrañaga, P., Bielza, C. (2018). Multi-dimensional BN classifier trees.
# ======================================================================

# ---- Libraries ----
library(bnlearn)
# gRain is optional; not required for dataset-level MPE we use here
tryCatch({ library(gRain) }, error = function(e) {})
library(arules)   # only for discretize() if needed

# ----------------------------------------------------------------------
# METRICS: multi-dimensional accuracy measures used in CIG codebases
# ----------------------------------------------------------------------
# For true labels Y (test_set[,classes]) and predicted labels Ŷ (pred_set),
# we compute:
#   • Global accuracy  = P(Ŷ = Y)   (exact-match across all d labels)
#   • Per-class acc_k  = P(Ŷ_k = Y_k)
#   • Average accuracy = (1/d) ∑_k acc_k
test_multidimensional <- function(test_set, pred_set, classes) {
  # convert to character to avoid level-mismatch problems in factors
  true <- data.frame(lapply(test_set[, classes], as.character), stringsAsFactors = FALSE)
  pred <- data.frame(lapply(pred_set[, classes], as.character), stringsAsFactors = FALSE)

  M <- (true == pred)                   # indicator matrix (N × d)
  exact_match <- apply(M, 1, all)       # row-wise all-true?
  per_class <- colMeans(M)              # mean per column → acc per class
  average_accuracy <- mean(per_class)   # macro-average
  global_accuracy  <- mean(exact_match) # exact-match accuracy

  list(global = global_accuracy, average = average_accuracy, per_class = per_class)
}

# ----------------------------------------------------------------------
# STRUCTURE LEARNING (FILTER): maximize BIC via hill-climbing (hc)
# ----------------------------------------------------------------------
# We blacklist all feature→class arcs so features can't be direct parents
# of classes; this respects the “classifier” orientation and reduces
# label leakage. bnlearn::hc uses BIC score by default.
#
# BIC(G;D) = log L(θ̂_G; D) - (k_G / 2) log N
# with θ̂_G the MLE/MAP per local CPT (we use Bayesian estimates with iss=1).
learn_MBC <- function(training_set, classes, features) {
  bl <- expand.grid(from = features, to = classes)  # forbid feature→class
  net <- hc(training_set, blacklist = as.matrix(bl))  # hill-climb BIC
  MBC_model <- bn.fit(net, training_set, method = "bayes", iss = 1)  # Laplace smoothing
  return(MBC_model)
}

# ----------------------------------------------------------------------
# STRUCTURE LEARNING (WRAPPER): greedy arc addition to improve accuracy
# ----------------------------------------------------------------------
# Objective:
#   maximize  J(measure) on validation data, with measure ∈ {global, average}
#   J_global( G ) = (1/N_val) ∑ 1[ ŷ(x_i; G) = y_i ]
#   J_avg( G )    = (1/d) ∑_k (1/N_val) ∑ 1[ ŷ_k(x_i; G) = y_{i,k} ]
#
# We start from the empty graph and add the single arc (respecting acyclicity
# and the feature→class blacklist) that yields the largest ΔJ > 0.
learn_MBC_wrapper2 <- function(training_set, validation_set, classes, features,
                               measure = "global", verbose = FALSE) {
  MBC_best <- empty.graph(nodes = c(classes, features))
  MBC_fit  <- bn.fit(MBC_best, training_set, method = "bayes", iss = 1)

  # Enumerate joint label space Ω = Ω(C1)×…×Ω(Cd)
  joint_levels <- lapply(classes, function(c) dimnames(MBC_fit[[c]]$prob)[[1]])
  names(joint_levels) <- classes
  class_combinations <- expand.grid(joint_levels)  # I × d
  I <- nrow(class_combinations)

  # helper: compute validation accuracy for a fitted model by MPE on Ω
  get_validation_accuracy <- function(model) {
    big_matrix <- cbind(
      validation_set[rep(1:nrow(validation_set), each = I), features, drop = FALSE],
      class_combinations[rep(1:I, nrow(validation_set)), ]
    )
    # log-likelihood per row: log p(y, x) = ∑_v log p(v | Pa(v))
    logL <- logLik(model, big_matrix, by.sample = TRUE)

    # per case i, take argmax over its I rows
    max_idx <- sapply(0:(nrow(validation_set)-1), function(j) {
      offset <- j * I
      which.max(logL[(offset+1):(offset+I)])
    })
    preds <- class_combinations[max_idx, , drop = FALSE]
    perf <- test_multidimensional(validation_set, preds, classes)
    if (measure == "average") perf$average else perf$global
  }

  best_perf <- get_validation_accuracy(MBC_fit)
  improved <- TRUE
  while (improved) {
    improved <- FALSE
    best_arc <- NULL
    candidate_arcs <- MBC_possible_arcs(classes, features)
    current_arcs <- if (!is.null(arcs(MBC_best))) arcs(MBC_best) else matrix(nrow = 0, ncol = 2)
    best_candidate_perf <- best_perf

    for (k in 1:nrow(candidate_arcs)) {
      arc <- candidate_arcs[k, ]

      # skip if arc already present
      if (nrow(current_arcs) > 0 &&
          any(apply(current_arcs, 1, function(x) all(x == arc)))) next

      # check acyclicity
      MBC_temp <- set.arc(MBC_best, from = arc["from"], to = arc["to"], check.cycles = FALSE)
      if (!acyclic(MBC_temp)) next

      # fit and evaluate
      MBC_temp_fit <- bn.fit(MBC_temp, training_set, method = "bayes", iss = 1)
      perf_val <- get_validation_accuracy(MBC_temp_fit)

      # accept best positive improvement (tiny tol to avoid float ties)
      if (perf_val > best_candidate_perf + 1e-10) {
        best_candidate_perf <- perf_val
        best_arc <- arc
      }
    }

    if (!is.null(best_arc)) {
      MBC_best <- set.arc(MBC_best, from = best_arc["from"], to = best_arc["to"])
      # normalize model string to keep a clean DAG (removes duplicates)
      MBC_best <- model2network(modelstring(MBC_best))
      MBC_fit  <- bn.fit(MBC_best, training_set, method = "bayes", iss = 1)
      best_perf <- best_candidate_perf
      improved <- TRUE
      if (verbose) message(sprintf("Added arc %s -> %s | %s-acc=%.4f",
                                   best_arc["from"], best_arc["to"], measure, best_perf))
    }
  }
  return(MBC_fit)
}

# ----------------------------------------------------------------------
# ARC CANDIDATES respecting the classifier constraint
# ----------------------------------------------------------------------
# Allowable arcs:
#   • class ↔ class   (both directions, acyclicity enforced)
#   • feature ↔ feature
#   • class → feature
# Forbid feature → class (so classes aren’t children of features).
MBC_possible_arcs <- function(classes, features) {
  nodes <- c(classes, features)
  P <- expand.grid(from = nodes, to = nodes, stringsAsFactors = FALSE)
  P <- P[P$from != P$to, ]
  bad <- which(P$from %in% features & P$to %in% classes)
  if (length(bad) > 0) P <- P[-bad, ]
  P
}

# ----------------------------------------------------------------------
# PREDICTION (MPE over the joint label space)
# ----------------------------------------------------------------------
# For each test case x, enumerate Ω and compute:
#   ŷ(x) = argmax_y log p(y, x) = argmax_y ∑_v log p(v | Pa(v))
# We vectorize this by stacking every test case repeated I times with
# every y ∈ Ω, then use bnlearn::logLik with by.sample=TRUE.
predict_MBC_dataset_veryfast <- function(MBC_model, data_set, classes, features) {
  options(warn = -1)

  class_levels <- lapply(classes, function(c) dimnames(MBC_model[[c]]$prob)[[1]])
  names(class_levels) <- classes
  class_joint <- expand.grid(class_levels)
  I <- nrow(class_joint)

  big_mat <- cbind(
    data_set[rep(1:nrow(data_set), each = I), features, drop = FALSE],
    class_joint[rep(1:I, nrow(data_set)), ]
  )
  logL <- logLik(MBC_model, big_mat, by.sample = TRUE)

  max_indices <- sapply(0:(nrow(data_set)-1), function(j) {
    s <- j * I + 1; e <- j * I + I
    which.max(logL[s:e])
  })
  preds <- class_joint[max_indices, , drop = FALSE]
  rownames(preds) <- NULL
  preds
}
