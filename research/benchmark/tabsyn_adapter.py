"""Numeric TabSyn wrapper; author models and sampler stay byte-identical.

The whole fit includes preprocessing, both full author epoch budgets and safe
export. Generated controls may explicitly request two epochs per stage. A
deadline never converts partial training into an eligible model.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import time

from research.benchmark.tabsyn_runtime_guard import verify as verify_runtime

AUTHOR_COMMIT = "cb5ac0f74ec36ee88e7a974a393dfbef50d42da7"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('whole fit deadline exceeded')


def validate_config(config, fixture=False):
    require(type(config) is dict and set(config) == {'max_beta', 'min_beta', 'lambd',
        'vae_epochs', 'diffusion_epochs', 'batch_size', 'token_dimension',
        'transformer_layers', 'attention_heads', 'factor', 'diffusion_hidden_dimension',
        'vae_lr', 'diffusion_lr', 'weight_decay', 'sampling_steps', 'config_sha256'},
        'TabSyn configuration fields differ')
    integer_fields = {'vae_epochs', 'diffusion_epochs', 'batch_size', 'token_dimension',
                      'transformer_layers', 'attention_heads', 'factor',
                      'diffusion_hidden_dimension', 'sampling_steps', 'weight_decay'}
    require(all(type(config[k]) is int for k in integer_fields)
            and all(type(config[k]) is float for k in ['max_beta', 'min_beta', 'lambd',
                                                       'vae_lr', 'diffusion_lr']),
            'TabSyn configuration types differ')
    canonical = {k: v for k, v in config.items() if k != 'config_sha256'}
    expected = hashlib.sha256(json.dumps(canonical, separators=(',', ':'), sort_keys=True).encode()).hexdigest()
    require(config['config_sha256'] == expected, 'TabSyn configuration digest differs')
    require(config['max_beta'] in [.01, .001] and config['min_beta'] == 1e-5
            and config['lambd'] == .7 and config['batch_size'] == 4096
            and config['token_dimension'] == 4 and config['transformer_layers'] == 2
            and config['attention_heads'] == 1 and config['factor'] == 32
            and config['diffusion_hidden_dimension'] == 1024 and config['vae_lr'] == .001
            and config['diffusion_lr'] == .001 and config['weight_decay'] == 0
            and config['sampling_steps'] == 50, 'TabSyn author-supported configuration differs')
    require(config['vae_epochs'] == (2 if fixture else 4000)
            and config['diffusion_epochs'] == (2 if fixture else 10001),
            'TabSyn author epoch budget differs')


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def numeric_loss(x, reconstructed, mu, logvar):
    """Empty categorical reconstruction means zero CE and zero accuracy."""
    import torch
    mse = (x-reconstructed).pow(2).mean()
    kl = -.5*torch.mean((1+logvar-mu.pow(2)-logvar.exp()).mean(-1).mean())
    zero = torch.zeros((), dtype=mse.dtype, device=mse.device)
    return mse, zero, kl, zero


def fit(train, validation, projection_bytes, output, config, *, seed=11,
        device='cpu', timeout_seconds=600, generated_fixture=False,
        runtime_path, runtime_sha256, code_path, code_sha256):
    require(type(seed) is int and seed == 11 and type(timeout_seconds) is int
            and 0 < timeout_seconds <= 600, 'TabSyn fit seed or deadline differs')
    require(type(generated_fixture) is bool and type(projection_bytes) is bytes,
            'TabSyn fit declaration differs')
    validate_config(config, generated_fixture)
    require(generated_fixture and device == 'cpu',
            'TabSyn benchmark execution needs independent request and GPU admission')
    started = time.monotonic()
    deadline = started+timeout_seconds
    verify_runtime(runtime_path, runtime_sha256, code_path, code_sha256)
    check_deadline(deadline)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from sklearn.preprocessing import QuantileTransformer
    from safetensors.torch import save_file
    from tabsyn.vae.model import Model_VAE, Encoder_model, Decoder_model
    from tabsyn.model import MLPDiffusion, Model
    require(isinstance(train, np.ndarray) and isinstance(validation, np.ndarray)
            and train.ndim == validation.ndim == 2 and train.shape[1] == validation.shape[1]
            and train.shape[0] >= 9 and validation.shape[0] >= 2 and train.shape[1] >= 2
            and np.isfinite(train).all() and np.isfinite(validation).all(),
            'TabSyn numeric input contract differs')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if device.startswith('cuda:'):
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    # Joint column order is target first followed by projected inputs, matching
    # utils_train.concat_y_to_X and src.data.normalize for finite numeric rows.
    normalizer = QuantileTransformer(output_distribution='normal',
        n_quantiles=max(min(len(train)//30, 1000), 10), subsample=int(1e9), random_state=0)
    x = torch.as_tensor(normalizer.fit_transform(train), dtype=torch.float32)
    v = torch.as_tensor(normalizer.transform(validation), dtype=torch.float32, device=device)
    loader = DataLoader(x, batch_size=4096, shuffle=True, num_workers=0)
    model = Model_VAE(2, train.shape[1], [], 4, n_head=1, factor=32, bias=True).to(device)
    # Preserve author RNG consumption: encoder/decoder are constructed before training.
    encoder = Encoder_model(2, train.shape[1], [], 4, n_head=1, factor=32).to(device).eval()
    decoder = Decoder_model(2, train.shape[1], [], 4, n_head=1, factor=32).to(device).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=.95, patience=10, verbose=True)
    beta, best, patience = config['max_beta'], float('inf'), 0
    selected_epoch = None
    for epoch in range(config['vae_epochs']):
        check_deadline(deadline)
        for batch in loader:
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)
            reconstruction, _, mu, logvar = model(batch, None)
            mse, ce, kl, _ = numeric_loss(batch, reconstruction, mu, logvar)
            (mse+ce+beta*kl).backward()
            optimizer.step()
            check_deadline(deadline)
        model.eval()
        with torch.no_grad():
            reconstruction, _, mu, logvar = model(v, None)
            val_mse, val_ce, _, _ = numeric_loss(v, reconstruction, mu, logvar)
            val_loss = val_mse.item()*0+val_ce.item()
        scheduler.step(val_loss)
        if val_loss < best:
            best, patience, selected_epoch = val_loss, 0, epoch
            save_file({k: t.detach().cpu().contiguous() for k, t in model.state_dict().items()},
                      str(root/'vae-selected.safetensors'))
        else:
            patience += 1
            if patience == 10 and beta > config['min_beta']:
                beta *= config['lambd']
    encoder.load_weights(model)
    decoder.load_weights(model)
    with torch.no_grad():
        latent = encoder(x.to(device), None)[:, 1:, :].reshape(len(train), -1)
        mean = latent.mean(0)
        latent = ((latent-mean)/2).detach().cpu()
    diffusion = Model(MLPDiffusion(latent.shape[1], 1024).to(device), latent.shape[1]).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=.9, patience=20, verbose=True)
    loader = DataLoader(latent, batch_size=4096, shuffle=True, num_workers=0)
    best, patience, diffusion_selected_epoch = float('inf'), 0, None
    diffusion.train()
    completed_diffusion_epochs = 0
    for epoch in range(config['diffusion_epochs']):
        check_deadline(deadline)
        total, count = 0., 0
        for batch in loader:
            loss = diffusion(batch.float().to(device)).mean()
            total += loss.item()*len(batch)
            count += len(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            check_deadline(deadline)
        current = total/count
        scheduler.step(current)
        completed_diffusion_epochs += 1
        if current < best:
            best, patience, diffusion_selected_epoch = current, 0, epoch
            save_file({k: t.detach().cpu().contiguous() for k, t in diffusion.state_dict().items()},
                      str(root/'diffusion.safetensors'))
        else:
            patience += 1
            if patience == 500:
                break
        if epoch % 1000 == 0:
            save_file({k: t.detach().cpu().contiguous() for k, t in diffusion.state_dict().items()},
                      str(root/f'diffusion-epoch-{epoch}.safetensors'))
    save_file({k: t.detach().cpu().contiguous() for k, t in encoder.state_dict().items()},
              str(root/'encoder-final.safetensors'))
    save_file({k: t.detach().cpu().contiguous() for k, t in decoder.state_dict().items()},
              str(root/'decoder-final.safetensors'))
    np.savez(root/'numeric-state.npz', quantiles=normalizer.quantiles_, references=normalizer.references_,
             latent_mean=mean.detach().cpu().numpy())
    (root/'projection.json').write_bytes(projection_bytes)
    peak = torch.cuda.max_memory_allocated(device) if device.startswith('cuda:') else 0
    require(peak <= 16*2**30, 'TabSyn VRAM budget exceeded')
    receipt = dict(format='dope-tabsyn-numeric-artifact', version=1, seed=seed,
        rows=len(train), columns=train.shape[1], config=config, generated_fixture=generated_fixture,
        source_commit=AUTHOR_COMMIT, runtime_sha256=runtime_sha256, code_sha256=code_sha256,
        complete=True, execution_admitted=False,
        vae_selected_epoch=selected_epoch, encoder_decoder_export='final_epoch',
        completed_vae_epochs=config['vae_epochs'], completed_diffusion_epochs=completed_diffusion_epochs,
        diffusion_selected_epoch=diffusion_selected_epoch, numeric_empty_ce=0,
        original_numeric_weight_in_validation=0, data_loader_workers=0,
        data_loader_adaptation='zero workers for isolated worker/no subprocess contract',
        whole_fit_elapsed_seconds=time.monotonic()-started, peak_allocated_vram_bytes=peak,
        official_tests_opened=False, mfs_v2=None, ptf_v1=None, release_safe_l3=None, superiority=None)
    check_deadline(deadline)
    (root/'model.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    try:
        check_deadline(deadline)
    except TimeoutError:
        (root/'model.json').unlink()
        raise
    return receipt


def sample(artifact, rows, seed, expected_inventory, *, device='cpu',
           runtime_path, runtime_sha256, code_path, code_sha256):
    require(type(rows) is int and 0 < rows <= 10_000_000 and type(seed) is int and seed in [101, 211, 307],
            'TabSyn sample identity differs')
    verify_runtime(runtime_path, runtime_sha256, code_path, code_sha256)
    require(device == 'cpu', 'TabSyn GPU sampling needs independent admission')
    require(artifact_inventory(artifact) == expected_inventory, 'TabSyn artifact custody differs')
    root = Path(artifact)
    info = json.loads((root/'model.json').read_text())
    require(info['complete'] is True and info['source_commit'] == AUTHOR_COMMIT
            and info['runtime_sha256'] == runtime_sha256 and info['code_sha256'] == code_sha256
            and info['generated_fixture'] is True and info['execution_admitted'] is False,
            'TabSyn artifact lineage or completion differs')
    validate_config(info['config'], fixture=True)
    require(rows in [info['rows']*m for m in [1, 2, 4, 8]], 'TabSyn sample row multiplier differs')
    import numpy as np
    import torch
    from sklearn.preprocessing import QuantileTransformer
    from safetensors.torch import load_file
    from tabsyn.vae.model import Decoder_model
    from tabsyn.model import MLPDiffusion, Model
    from tabsyn.diffusion_utils import sample as author_sample
    seed_all(seed)
    if device.startswith('cuda:'):
        torch.cuda.manual_seed_all(seed)
    state = np.load(root/'numeric-state.npz', allow_pickle=False)
    normalizer = QuantileTransformer(output_distribution='normal')
    normalizer.quantiles_ = state['quantiles']
    normalizer.references_ = state['references']
    normalizer.n_quantiles_ = len(normalizer.references_)
    normalizer.n_features_in_ = info['columns']
    latent_dim = info['columns']*4
    decoder = Decoder_model(2, info['columns'], [], 4, n_head=1, factor=32).to(device).eval()
    decoder.load_state_dict(load_file(str(root/'decoder-final.safetensors'), device=device))
    diffusion = Model(MLPDiffusion(latent_dim, 1024).to(device), latent_dim).to(device).eval()
    diffusion.load_state_dict(load_file(str(root/'diffusion.safetensors'), device=device))
    with torch.no_grad():
        latent = author_sample(diffusion.denoise_fn_D, rows, latent_dim,
                               num_steps=50, device=device)
        latent = latent*2+torch.as_tensor(state['latent_mean'], device=device)
        numeric, categorical = decoder(latent.reshape(rows, info['columns'], 4))
        require(not categorical, 'TabSyn numeric decoder contract differs')
    result = normalizer.inverse_transform(numeric.cpu().numpy())
    require(result.shape == (rows, info['columns']) and np.isfinite(result).all(),
            'TabSyn sample shape or finiteness differs')
    return result


def artifact_inventory(artifact):
    root = Path(artifact)
    require(root.is_dir() and not root.is_symlink(), 'TabSyn artifact root differs')
    files = {}
    for p in sorted(root.iterdir()):
        require(p.is_file() and not p.is_symlink(), 'TabSyn artifact member differs')
        digest = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                digest.update(block)
        files[p.name] = dict(bytes=p.stat().st_size, sha256=digest.hexdigest())
    expected = {'vae-selected.safetensors', 'diffusion.safetensors', 'encoder-final.safetensors',
        'decoder-final.safetensors', 'numeric-state.npz', 'projection.json', 'model.json'}
    require(expected <= set(files) and all(n in expected or
            (n.startswith('diffusion-epoch-') and n.endswith('.safetensors')) for n in files),
            'TabSyn artifact inventory differs')
    return dict(files=files, artifact_bytes=sum(r['bytes'] for r in files.values()),
                projection_bytes=files['projection.json']['bytes'])


def native_objective(synthetic, validation, metadata, seed=11, *,
                     runtime_path, runtime_sha256, code_path, code_sha256):
    """Call the immutable author's evaluator, preserving its label behavior."""
    require(type(seed) is int and seed == 11, 'TabSyn native seed differs')
    lock = verify_runtime(runtime_path, runtime_sha256, code_path, code_sha256)
    require(lock['gpu_runtime_closure_certified'] is True and lock['execution_admitted'] is True,
            'TabSyn native GPU evaluator needs independent admission')
    import numpy as np
    from eval.mle.mle import _evaluate_regression
    require(metadata['task_type'] == 'regression' and len(metadata['target_col_idx']) == 1,
            'TabSyn native target metadata differs')
    transformed = np.log(np.clip(validation[:, metadata['target_col_idx'][0]], 1, 20000))
    require(len(transformed) >= 2 and np.isfinite(transformed).all(),
            'TabSyn native validation contract differs')
    if np.unique(transformed).size < 2:
        return dict(status='tuning_inapplicable_uninformative_author_target', native_value=None)
    seed_all(seed)
    best_r2, _ = _evaluate_regression(synthetic, validation, metadata)
    value = float(next(r['r2'] for r in best_r2 if r['name'] == 'XGBRegressor'))
    require(math.isfinite(value), 'TabSyn native objective nonfinite')
    return dict(status='ok', native_value=value, objective='best_r2_scores.XGBRegressor.r2',
                direction='maximize', source_transform='log_clip_fit_and_real_raw_synthetic_validation')


def select_native(trials):
    require(len(trials) <= 8, 'TabSyn native trial budget exceeded')
    eligible = []
    for row in trials:
        require(type(row['elapsed_seconds']) in [int, float] and math.isfinite(row['elapsed_seconds'])
                and row['elapsed_seconds'] >= 0,
                'TabSyn trial cost differs')
        if row['status'] != 'ok':
            continue
        require(type(row['native_value']) in [int, float] and math.isfinite(row['native_value'])
                and type(row['artifact_bytes']) is int
                and row['artifact_bytes'] > 0 and type(row['config_sha256']) is str
                and len(row['config_sha256']) == 64
                and all(c in '0123456789abcdef' for c in row['config_sha256']),
                'TabSyn native trial fields differ')
        eligible.append(row)
    require(sum(r['elapsed_seconds'] for r in trials) <= 43200, 'TabSyn native cell deadline exceeded')
    if not eligible:
        return None
    return min(eligible, key=lambda r: (-r['native_value'], r['artifact_bytes'], r['config_sha256']))
