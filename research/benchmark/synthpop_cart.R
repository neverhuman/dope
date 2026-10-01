# SPDX-License-Identifier: GPL-2.0-only OR GPL-3.0-only
# Restricted research adapter for synthpop 1.9-3 numeric CART synthesis.
# Author fit and native KPI: synthpop::syn and synthpop::utility.gen.
# Sampling follows the donor-leaf algorithm in synthpop's GPL-licensed
# syn.cart() (Nowok, Raab, and contributors); this file is separately GPL
# licensed and is not part of the MIT-licensed product runtime.

suppressPackageStartupMessages(library(synthpop))

read_numeric <- function(path) {
  x <- read.csv(path, header = FALSE, colClasses = "numeric", check.names = FALSE)
  if (nrow(x) < 2 || ncol(x) < 2 || any(!is.finite(as.matrix(x))))
    stop("invalid numeric benchmark partition")
  names(x) <- paste0("c", seq_len(ncol(x)))
  x
}

draw_cart <- function(fit, xp) {
  if (!inherits(fit, "rpart") || fit$method != "anova")
    stop("unsupported synthpop CART model")
  y <- fit$y
  # Repeated data-frame rows gain decimal suffixes in R row names; syn.cart
  # floors them back to their original rpart leaf identifiers.
  leaves <- floor(as.numeric(row.names(fit$frame[fit$where, , drop = FALSE])))
  nodes <- as.numeric(predict(fit, newdata = xp))
  if (length(nodes) != nrow(xp) || any(!is.finite(nodes)))
    stop("invalid CART leaf predictions")
  for (node in setdiff(unique(nodes), unique(leaves))) {
    replacement <- node
    while (!(replacement %in% leaves))
      replacement <- 2 * replacement + sample(0:1, 1)
    nodes[nodes == node] <- replacement
  }
  out <- numeric(nrow(xp))
  for (node in unique(nodes)) {
    donors <- y[leaves == node]
    if (!length(donors)) stop("CART leaf has no donors")
    out[nodes == node] <- donors[sample.int(length(donors), sum(nodes == node),
                                           replace = TRUE)]
  }
  out
}

sample_model <- function(artifact, rows, seed) {
  if (!identical(artifact$format, "dope-synthpop-cart-numeric-v1") ||
      !identical(artifact$package_version, as.character(packageVersion("synthpop"))) ||
      rows < 1 || rows > 1000000) stop("incompatible synthpop artifact")
  set.seed(seed)
  columns <- names(artifact$models)
  out <- as.data.frame(matrix(NA_real_, nrow = rows, ncol = length(columns)))
  names(out) <- columns
  for (j in artifact$visit_sequence) {
    name <- columns[j]
    method <- artifact$methods[[name]]
    fit <- artifact$models[[name]]
    if (method == "cart") {
      out[[name]] <- draw_cart(fit, out)
    } else if (method == "sample") {
      donor <- artifact$donors[[name]]
      out[[name]] <- donor[sample.int(length(donor), rows, replace = TRUE)]
    } else if (method == "constant") {
      donor <- artifact$donors[[name]]
      out[[name]] <- synthpop:::syn.constant(donor, rows)$res
    } else if (method == "collinear") {
      parent <- columns[which(artifact$predictor_matrix[j, ] == 1)[1]]
      pair <- artifact$donors[[name]]
      out[[name]] <- pair$y[match(out[[parent]], pair$x)]
    } else {
      stop("unsupported synthpop synthesis method")
    }
    if (any(!is.finite(out[[name]]))) stop("nonfinite synthpop sample")
  }
  out
}

fit_command <- function(args) {
  train <- read_numeric(args[2])
  valid <- read_numeric(args[3])
  if (!identical(names(train), names(valid))) stop("partition width changed")
  seed <- as.integer(args[6])
  minbucket <- as.integer(args[7])
  cp <- as.numeric(args[8])
  if (is.na(seed) || is.na(minbucket) || is.na(cp) || minbucket < 1 || cp <= 0)
    stop("invalid synthpop CART configuration")
  set.seed(seed)
  # sampler.syn captures method-specific dots unevaluated, so construct
  # literal numeric arguments before entering the author's syn() function.
  fitted <- do.call(synthpop::syn, list(data = train, method = "cart",
                 k = nrow(valid), models = TRUE, seed = seed,
                 cart.minbucket = minbucket, cart.cp = cp,
                 print.flag = FALSE))
  synthetic <- fitted$syn
  if (any(!is.finite(as.matrix(synthetic)))) stop("nonfinite fitted synthesis")
  utility <- synthpop::utility.gen(synthetic, valid, method = "cart",
                                    resamp.method = "none", print.flag = FALSE)
  value <- as.numeric(utility$pMSE)
  if (length(value) != 1 || !is.finite(value)) stop("invalid native CART pMSE")
  donors <- list()
  for (j in seq_along(fitted$method)) {
    name <- names(fitted$method)[j]
    if (fitted$method[j] %in% c("sample", "constant")) {
      donors[[name]] <- train[[name]]
    } else if (fitted$method[j] == "collinear") {
      parent <- names(train)[which(fitted$predictor.matrix[j, ] == 1)[1]]
      donors[[name]] <- list(x = train[[parent]], y = train[[name]])
    } else if (fitted$method[j] != "cart" ||
               !inherits(fitted$models[[name]], "rpart") ||
               fitted$models[[name]]$method != "anova") {
      stop("unsupported fitted synthpop method")
    }
  }
  artifact <- list(format = "dope-synthpop-cart-numeric-v1",
                   package_version = as.character(packageVersion("synthpop")),
                   models = fitted$models, methods = fitted$method,
                   visit_sequence = fitted$visit.sequence,
                   predictor_matrix = fitted$predictor.matrix,
                   donors = donors, source_rows_required_for_sampling = FALSE)
  saveRDS(artifact, args[4], compress = "xz")
  probe <- sample_model(readRDS(args[4]), 32L, seed + 1000L)
  if (ncol(probe) != ncol(train) || any(!is.finite(as.matrix(probe))))
    stop("artifact-only synthpop sample failed")
  metric <- sprintf(paste0('{"format":"dope-synthpop-cart-native-validation-kpi",',
                           '"version":1,"value":%.17g,"direction":"minimize",',
                           '"objective":"validation_cart_pMSE",',
                           '"package_version":"%s","seed":%d,',
                           '"minbucket":%d,"cp":%.17g,',
                           '"official_tests_opened":false}'),
                    value, as.character(packageVersion("synthpop")),
                    seed, minbucket, cp)
  writeLines(metric, args[5])
}

sample_command <- function(args) {
  artifact <- readRDS(args[2])
  rows <- as.integer(args[3])
  seed <- as.integer(args[4])
  if (is.na(rows) || is.na(seed)) stop("invalid sample request")
  generated <- sample_model(artifact, rows, seed)
  write.table(generated, args[5], sep = ",", row.names = FALSE,
              col.names = FALSE, quote = FALSE, na = "")
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 8 && args[1] == "fit") {
    fit_command(args)
  } else if (length(args) == 5 && args[1] == "sample") {
    sample_command(args)
  } else {
    stop("usage: synthpop_cart.R fit TRAIN VALID MODEL KPI SEED MINBUCKET CP | sample MODEL ROWS SEED OUTPUT")
  }
}
