# SPDX-License-Identifier: GPL-2.0-only OR GPL-3.0-only
# Contract probe for the GPL-licensed research-only synthpop CART adapter.

source("research/benchmark/synthpop_cart.R")
set.seed(42)
rows <- 300L
a <- runif(rows)
b <- as.numeric(a > 0.5)
fixture <- data.frame(c1 = a, c2 = b, c3 = 2 * b,
                      c4 = rep(0.3, rows),
                      c5 = pmin(1, pmax(0, 0.8 * a + 0.2 * b + rnorm(rows, sd = 0.03))))
valid <- fixture[1:100, , drop = FALSE]
fit <- do.call(synthpop::syn, list(data = fixture, method = "cart", k = 100L,
               models = TRUE, seed = 11L, cart.minbucket = 5L,
               cart.cp = 1e-8, print.flag = FALSE))
stopifnot(any(fit$method == "cart"), any(fit$method == "constant"),
          any(fit$method == "collinear"))
objective <- synthpop::utility.gen(fit$syn, valid, method = "cart",
                                   resamp.method = "none", print.flag = FALSE)$pMSE
stopifnot(length(objective) == 1, is.finite(objective))
donors <- list()
for (j in seq_along(fit$method)) {
  name <- names(fit$method)[j]
  if (fit$method[j] %in% c("sample", "constant")) {
    donors[[name]] <- fixture[[name]]
  } else if (fit$method[j] == "collinear") {
    parent <- names(fixture)[which(fit$predictor.matrix[j, ] == 1)[1]]
    donors[[name]] <- list(x = fixture[[parent]], y = fixture[[name]])
  }
}
model <- list(format = "dope-synthpop-cart-numeric-v1",
              package_version = as.character(packageVersion("synthpop")),
              models = fit$models, methods = fit$method,
              visit_sequence = fit$visit.sequence,
              predictor_matrix = fit$predictor.matrix, donors = donors)
generated <- sample_model(model, 100L, 101L)
again <- sample_model(unserialize(serialize(model, NULL)), 100L, 101L)
stopifnot(identical(generated, again), ncol(generated) == ncol(fixture),
          all(is.finite(as.matrix(generated))),
          all(generated$c3 == 2 * generated$c2),
          all(generated$c4 == 0.3))
first_cart <- names(fit$method)[which(fit$method == "cart")[1]]
cart_model <- fit$models[[first_cart]]
set.seed(101)
probe <- draw_cart(cart_model, generated)
nodes <- as.numeric(predict(cart_model, newdata = generated))
leaves <- as.numeric(row.names(cart_model$frame[cart_model$where, , drop = FALSE]))
stopifnot(all(vapply(seq_along(probe), function(i) {
  probe[i] %in% cart_model$y[leaves == nodes[i]]
}, logical(1))))
cat("synthpop CART fit/native KPI/artifact-only sample contract: ok\n")
