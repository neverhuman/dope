"""Per-type/per-level normalized objectives and explicit masked auxiliary supervision."""
import torch
from torch.nn import functional as F
from .contracts import level_widths, require
from .decoder import alpha
from .model import tensors
from .proxies import transfer_driver
from .table_encoder import quantized_code


def generation(decoder, code, schema, values, observed):
    time = torch.rand((), device=code.device)*.9+.05
    a, noise = alpha(time), torch.randn_like(values)
    noisy = a.sqrt()*values + (1-a).sqrt()*noise
    kinds = torch.tensor([c["kind"] == "categorical" for c in schema["columns"]], device=code.device)
    corrupted = torch.rand(values.shape, device=code.device) < time
    noisy[:, kinds] = torch.where(corrupted[:, kinds], 0., values[:, kinds])
    noisy_observed = observed & ~corrupted
    epsilon, logits, missing = decoder(code, schema, noisy, noisy_observed, time)
    numeric_mask = observed & ~kinds
    numeric = (epsilon-noise).square()[numeric_mask].mean() if numeric_mask.any() else epsilon.sum()*0
    categorical_losses = []
    for j, column in enumerate(schema["columns"]):
        use = observed[:, j] & corrupted[:, j]
        if column["kind"] == "categorical" and use.any():
            categorical_losses.append(F.cross_entropy(logits[use, j], values[use, j].long()))
    categorical = torch.stack(categorical_losses).mean() if categorical_losses else logits.sum()*0
    missing_loss = F.binary_cross_entropy_with_logits(missing, (~observed).float())
    return numeric+categorical+missing_loss, dict(numeric=numeric, categorical=categorical, missing=missing_loss)


def soft_rows(decoder, code, schema, count):
    noise = torch.randn((count, len(schema["columns"])), device=code.device)
    kinds = torch.tensor([c["kind"] == "categorical" for c in schema["columns"]], device=code.device)
    noisy = noise.clone(); noisy[:, kinds] = 0
    a = alpha(torch.tensor(.5, device=code.device))
    epsilon, logits, missing = decoder(code, schema, noisy, torch.zeros_like(noisy, dtype=torch.bool), .5)
    values = ((noise-(1-a).sqrt()*epsilon)/a.sqrt()).clamp(-8,8)
    return values, 1-missing.sigmoid(), logits.softmax(-1)


def objective(model, context, query, profile, config, step, *, phase="ordinary", differentiable=True):
    device = next(model.parameters()).device
    cv, co = tensors(context, device); qv, qo = tensors(query, device)
    schema = context.schema
    code = quantized_code(model.encoder(cv, co, schema, through="ultra" if phase == "ultra-residual" else "high"))
    selected = ["low","medium","high"] if phase == "ordinary" else ["ultra"]
    ramp = min(1., step/max(1,config.proxy_ramp_steps))
    total, records = code.sum()*0, {}
    for level in selected:
        prefix = code[:level_widths(model.config.layout)[level]]
        gen, terms = generation(model.decoder, prefix, schema, qv, qo)
        transfer, driver, state = gen*0, gen*0, {}
        if ramp > 0 and (config.transfer_weight or config.driver_weight):
            n = min(config.proxy_rows, len(cv))
            synthetic, present, probabilities = soft_rows(model.decoder, prefix, schema, n)
            with torch.autocast(device_type=device.type,enabled=False):
                transfer, driver, state = transfer_driver(cv[:n].float(),co[:n],qv.float(),qo,
                    synthetic.float(),present.float(),probabilities.float(),schema,differentiable=differentiable,
                    neural_unrolls=config.neural_unrolls)
        labels = torch.tensor(profile[0], dtype=torch.float32, device=device)
        mask = torch.tensor(profile[1], dtype=torch.bool, device=device)
        predictions = model.profiles[level](prefix)
        profile_loss = F.huber_loss(predictions[mask],labels[mask]) if mask.any() else predictions.sum()*0
        consistency = gen*0
        if config.consistency_weight and len(cv) >= 8:
            small = quantized_code(model.encoder(cv[::2], co[::2], schema,
                                   through="ultra" if level == "ultra" else level))
            consistency = F.mse_loss(F.normalize(prefix, dim=0), F.normalize(small,dim=0))
        value = gen + ramp*(config.transfer_weight*transfer+config.driver_weight*driver)
        value = value + config.profile_weight*profile_loss+config.consistency_weight*consistency
        require(torch.isfinite(value).item(), "nonfinite_objective")
        total = total+value/len(selected)
        record = dict(total=float(value.detach()),generation=float(gen.detach()),
                      transfer_proxy=float(transfer.detach()),driver_proxy=float(driver.detach()),
                      profile=float(profile_loss.detach()),profile_slots=int(mask.sum()),
                      consistency=float(consistency.detach()),proxy_ramp=ramp,**state)
        record.update({name:float(v.detach()) for name,v in terms.items()})
        records[level] = record
    return total, records
