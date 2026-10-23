"""Differentiable development proxies. They never certify exact learner transfer/drivers."""
import torch
from torch.nn import functional as F
from .contracts import require


def features(values, observed, schema, probabilities=None):
    parts, groups, start = [], [], 0
    for j, column in enumerate(schema["columns"][:-1]):
        present = observed[:, j:j+1].float()
        if column["kind"] == "numeric":
            value = values[:, j:j+1].clamp(-8, 8)*present
        else:
            count = column["categories"]
            value = (F.one_hot(values[:, j].long(), count).float() if probabilities is None
                     else probabilities[:, j, :count])*present
        block = torch.cat((value, 1-present), dim=1)
        parts.append(block); groups.append((start, start+block.shape[1])); start += block.shape[1]
    return torch.cat(parts, dim=1), groups


def labels(values, schema, probabilities=None):
    if schema["task"] == "regression": return values[:, -1:]
    count = schema["columns"][-1]["categories"]-2
    return (F.one_hot(values[:, -1].long()-2, count).float() if probabilities is None
            else probabilities[:, -1, 2:count+2])


def ridge(x, y):
    x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1).float()
    y = y.float()
    # Dual solve is bounded by proxy row count, including wide categorical inputs.
    kernel = x @ x.T + .1*torch.eye(len(x), device=x.device)
    coefficients = torch.linalg.solve(kernel, y)
    return x.T @ coefficients


def predict(x, coefficients):
    return torch.cat((x, torch.ones_like(x[:, :1])), dim=1).float() @ coefficients


def loss(prediction, target, schema):
    if schema["task"] == "regression": return (prediction-target).square().mean()
    return -(target*prediction.log_softmax(-1)).sum(-1).mean()


def neural_fit(x, y, schema, *, differentiable, iterations=3):
    """Matched fixed initialization and three explicit unrolls; no exact-learner claim."""
    with torch.enable_grad():
        generator = torch.Generator(device=x.device).manual_seed(91827)
        a = (torch.randn((x.shape[1], 16), generator=generator, device=x.device)*.05).requires_grad_()
        b = torch.zeros((16, y.shape[1]), device=x.device, requires_grad=True)
        for _ in range(iterations):
            prediction = torch.tanh(x @ a) @ b
            objective = loss(prediction, y, schema) + .001*(a.square().mean()+b.square().mean())
            da, db = torch.autograd.grad(objective, (a, b), create_graph=differentiable)
            a, b = a-.1*da, b-.1*db
        return (a, b) if differentiable else (a.detach(), b.detach())


def transfer_driver(real_values, real_observed, query_values, query_observed,
                    synthetic, soft_observed, probabilities, schema, *, differentiable, neural_unrolls=3):
    require(type(neural_unrolls) is int and 1 <= neural_unrolls <= 64, 'invalid_proxy_unrolls')
    xr, groups = features(real_values, real_observed, schema)
    xq, _ = features(query_values, query_observed, schema)
    xs, _ = features(synthetic, soft_observed, schema, probabilities)
    yr, yq, ys = labels(real_values, schema), labels(query_values, schema), labels(synthetic, schema, probabilities)
    reference = ridge(xr.detach(), yr.detach()).detach()
    fitted = ridge(xs, ys)
    real_prediction, syn_prediction = predict(xq, reference), predict(xq, fitted)
    real_loss, syn_loss = loss(real_prediction, yq, schema), loss(syn_prediction, yq, schema)
    null = yr.mean(0, keepdim=True).expand_as(yq)
    if schema["task"] == "classification":
        null_loss = -(yq*null.clamp_min(1e-6).log()).sum(-1).mean()
    else: null_loss = (null-yq).square().mean()
    gain = null_loss-real_loss
    valid = bool((gain > 1e-4).item())
    regret = (syn_loss-real_loss)/gain.clamp_min(1e-4) if valid else syn_loss*0
    real_nn = neural_fit(xr.detach(), yr.detach(), schema, differentiable=False, iterations=neural_unrolls)
    syn_nn = neural_fit(xs, ys, schema, differentiable=differentiable, iterations=neural_unrolls)
    nn_real = loss(torch.tanh(xq @ real_nn[0]) @ real_nn[1], yq, schema).detach()
    nn_syn = loss(torch.tanh(xq @ syn_nn[0]) @ syn_nn[1], yq, schema)
    nn_gain = null_loss-nn_real
    nn_valid = bool((nn_gain > 1e-4).item())
    nn_regret = (nn_syn-nn_real)/nn_gain.clamp_min(1e-4) if nn_valid else nn_syn*0
    informative = [v for v, good in [(regret, valid), (nn_regret, nn_valid)] if good]
    transfer = (torch.logsumexp(torch.stack(informative)*4, 0)/4
                + torch.stack(informative).mean()) if informative else syn_loss*0
    real_importance, syn_importance = [], []
    for begin, end in groups:
        perturbed = xq.clone()
        perturbed[:, begin:end] = xq.roll(1, dims=0)[:, begin:end]
        real_importance.append((predict(perturbed, reference)-real_prediction).square().mean())
        syn_importance.append((predict(perturbed, fitted)-syn_prediction).square().mean())
    real_importance, syn_importance = torch.stack(real_importance), torch.stack(syn_importance)
    driver_valid = bool((real_importance.sum() > 1e-6).item())
    driver = ((real_importance/real_importance.sum().clamp_min(1e-6)
               - syn_importance/syn_importance.sum().clamp_min(1e-6)).square().mean()
              if driver_valid else syn_loss*0)
    return transfer, driver, dict(linear_informative=valid, neural_informative=nn_valid,
                                 driver_proxy_informative=driver_valid, exact_tree="not_run",
                                 exact_driver_ceiling="not_measured", neural_unrolls=neural_unrolls)
