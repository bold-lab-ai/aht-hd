# Ad-Hoc Teamwork from Human Demonstrations

Research code for learning latent-conditioned Hanabi partners from human demonstrations, reconstructing them with KL-regularised PPO, and auditing proxy-suite coverage and evaluation sensitivity.

Start with the CPU examples below. For the Hanabi pipeline, follow the stages in order and supply your own authorized data and checkpoints. The package preserves the original `ah2ac2` module paths so the research implementations can be inspected and used together.

## Find the relevant code

| What you want to do | Start here |
| --- | --- |
| Understand the layout | [Pipeline and artifact flow](#hanabi-pipeline) |
| Run an example without human data or a GPU | [CPU quick start](#cpu-quick-start) |
| Train pooled BC | [`bc.py`](ah2ac2/training/bc.py) |
| Train the masked trajectory VAE | [`vae_2p_masked.py`](ah2ac2/training/vae_2p_masked.py) |
| Fit a latent sampling prior | [`fit_latent_prior.py`](ah2ac2/tools/fit_latent_prior.py) |
| Initialize references from demonstrated trajectories | [`latent_search_from_dataset.py`](ah2ac2/tools/latent_search_from_dataset.py) |
| Refine a finite reference set using SP and XP | [`latent_es_refine.py`](ah2ac2/tools/latent_es_refine.py) |
| Reconstruct latent references with PPO and a KL controller | [`rl_ppo_vae_anchor_vae_params_to_init.py`](ah2ac2/training/rl_ppo_vae_anchor_vae_params_to_init.py) |
| Reconstruct BC references | [`rl_ppo_bc_anchor.py`](ah2ac2/training/rl_ppo_bc_anchor.py) |
| Train best responses | [`br_bc_lstm.py`](ah2ac2/training/br_bc_lstm.py), [`br_latent_lstm.py`](ah2ac2/training/br_latent_lstm.py), [`br_rl_bc_lstm.py`](ah2ac2/training/br_rl_bc_lstm.py), [`br_rl_vae_lstm.py`](ah2ac2/training/br_rl_vae_lstm.py) |
| Evaluate fixed agents and proxy families | [`crossplay_eval.py`](ah2ac2/tools/crossplay_eval.py) |
| Inspect reference-quality/target sweeps | [`evaluate_kl_anchor_sweep.py`](ah2ac2/tools/evaluate_kl_anchor_sweep.py), [`calibration.py`](ah2ac2/tools/calibration.py) |
| Compute interpretable behaviour coverage | [`coverage_behaviour.py`](ah2ac2/tools/coverage_behaviour.py), [`coverage_drop_dims.py`](ah2ac2/tools/coverage_drop_dims.py), [`coverage_analyze.py`](ah2ac2/tools/coverage_analyze.py) |
| Check coverage sensitivity and human support | [`coverage_sensitivity.py`](ah2ac2/tools/coverage_sensitivity.py), [`human_grounding_certificate.py`](ah2ac2/tools/human_grounding_certificate.py) |
| Understand paired tests, ranks and full-dimensional MMD | [`experiment3_stats.py`](ah2ac2/tools/experiment3_stats.py) |
| Inspect the signaling-game noise analysis | [`signaling.py`](ah2ac2/theory/signaling.py) |

## CPU quick start

From this repository's root, with Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python scripts/signaling_demo.py
python scripts/analyze_scores.py --demo
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

The score demo generates **synthetic scores**, not the paper's measurements. It exercises the same statistical primitives without requiring private data. The output distinguishes suite easiness from the training-family compatibility interaction and records paired game uncertainty.

To analyze a method pair from your own evaluation, save four arrays of shape `[game]` in an NPZ: `bc_on_bc`, `bc_on_latent`, `latent_on_bc`, and `latent_on_latent`. Average over proxies within each family before saving; use the same ordered game keys for all arrays.

```bash
python scripts/analyze_scores.py --input artifacts/paired_scores.npz \
  --replicates 50000 --output outputs/paired-analysis.json
```

For a single pair, the script reports raw two-sided p-values. For multiple planned pairs, use `holm_adjust` across the test family. Bootstrap intervals condition on fixed checkpoints and do not estimate training-seed or suite-construction variance.

## Hanabi pipeline

```text
Authorized human games
  -> masked trajectory VAE checkpoint and its architecture config
  -> initial latent codes (dataset selection or fitted-prior sampling)
  -> SP/XP evolutionary refinement of the complete code set
  -> one fixed reference per selected code
  -> PPO reconstruction with a requested forward-KL target
  -> fixed evaluation partners
  -> common-game returns and rollout behaviour
  -> paired statistical and coverage audits
```

Install the additional dependencies for simulation and training:

```bash
python -m pip install -e '.[hanabi]'
```

Install the JAX build appropriate for your accelerator separately if required. This dependency list is not an archived lockfile of the experiment environment. The CPU examples and tests have been exercised in the preparation environment; full GPU training has not been rerun for this release.

### 1. Supply authorized data

Obtain the benchmark-permissible dataset and access information through [AH2AC2](https://github.com/FLAIROx/ah2ac2). This repository contains dataset readers, not human logs or benchmark test partners. Place your authorized inputs under `artifacts/`, which is ignored by Git.

The expected safetensors schema is documented in [`dataset.py`](ah2ac2/datasets/dataset.py): game IDs, scores, initial decks, padded actions, action counts and game-length masks. The simulator needs Hanabi's `reset_from_deck_of_pairs` interface to replay those decks. Keep training, development/calibration and disjoint audit games separate. Public benchmark data do not grant access to the complete private corpus used to construct evaluation suites.

### 2. Train the masked VAE

```bash
python -m ah2ac2.training.vae_2p_masked \
  train_data_path=artifacts/data/train.safetensors \
  val_data_path=artifacts/data/development.safetensors \
  save_path=artifacts/vae/
```

Training emits a checkpoint and YAML sidecar. Use their actual filenames below. The encoder input is a masked seat-observation sequence; actions are decoder targets. A fixed code still permits history-dependent actions through the recurrent decoder. The plain `vae_2p.py` module is retained where other tools import its shared model interface.

### 3. Initialize and refine references

Set paths to the checkpoint and matching sidecar produced in stage 2:

```bash
export VAE_CKPT=artifacts/vae/your_checkpoint.safetensors
export VAE_CONFIG=artifacts/vae/your_checkpoint.yaml
```

For the dataset-selection initializer used in the original construction:

```bash
python -m ah2ac2.tools.latent_search_from_dataset \
  --vae_ckpt "$VAE_CKPT" --train_data_path artifacts/data/train.safetensors \
  --latent_dim 32 --final_set_size 3 --act_fn gelu \
  --out_dir artifacts/initial

python -m ah2ac2.tools.latent_es_refine \
  --vae_ckpt "$VAE_CKPT" --init_z_path artifacts/initial/selected_z.npy \
  --num_latents 3 --latent_dim 32 --xp_weight 0.75 \
  --num_pairs_xp 3 --num_iterations 1000 --population_size 256 \
  --num_rollouts_sp 32 --num_rollouts_xp 32 --out_dir artifacts/search
```

For a fitted-prior initializer, fit the prior from your own authorized training set:

```bash
python -m ah2ac2.tools.fit_latent_prior \
  --ckpt "$VAE_CKPT" --cfg "$VAE_CONFIG" \
  --train-data artifacts/data/train.safetensors --outdir artifacts/prior
```

Then replace `--init_z_path` with `--prior_json artifacts/prior/prior_spherical.json` in the ES command. The second construction additionally refits the encoder with seed 2 and uses 3,000 ES iterations. Examples specify central search settings; inspect `--help` and the training configs for the remaining budgets. They do not reconstruct the exact historical run solely from public inputs.

ES emits `es_final_Z.npy`, per-code latent files, SP/XP diagnostics and a log. Keep every selected reference when constructing the evaluation suite. Search optimizes competence and mutual incompatibility; it does not optimize the behaviour-space coverage statistic or downstream method rankings.

### 4. Calibrate and reconstruct

```bash
python -m ah2ac2.training.rl_ppo_vae_anchor_vae_params_to_init \
  anchor_decoder_params_path="$VAE_CKPT" \
  anchor_z_npy_path=artifacts/search/YOUR_PER_CODE_LATENT.npy \
  kl_target=YOUR_CALIBRATED_TARGET \
  save_path=artifacts/proxies/reference_0/ wandb_mode=disabled
```

Substitute a numerical target and a per-code filename actually emitted by search. Run reconstruction for each selected code. Defaults in historical configs refer to absent research artifacts and must be overridden. Configs are in [`ah2ac2/training/config`](ah2ac2/training/config).

Calibration is a reference-quality × requested-target × seed sweep. Select each reference's target using external fidelity, then fit a nonincreasing quality-to-target map using `monotone_piecewise_fit`; `piecewise_predict` applies the fit. Inputs have shape `[reference, 2]`, with columns **selected requested target, reference quality**. These helpers implement the fitting rule; they do not contain the historical fitted map.

The map is domain/reference-family specific. Full-corpus maps, their point sets and learned priors are restricted benchmark artifacts and are excluded. Targets recorded for the original suite were 0.064/0.064/0.064; the second used 0.064/0.064/0.065. These are historical requested targets, not universal recommendations or achieved rollout KL values. Using permitted-data training partners requires calibrating on permitted data.

### 5. Evaluate and audit

Use `python -m ah2ac2.tools.crossplay_eval --help` for the supported checkpoint loaders and CLI. Preserve each checkpoint's architecture, activation and fixed latent input. Check self-play before cross-play, share game keys across cells and retain per-game scores. Use the paired score analyzer above for the resulting family averages.

For behaviour coverage, first extract the raw 24-coordinate statistics, then remove the event-sparse blind-play-slot block M10 to obtain the paper's 19-D representation:

```bash
python -m ah2ac2.tools.coverage_behaviour \
  --human_games artifacts/data/audit.safetensors \
  --bot_rollouts_dir artifacts/rollouts --out_dir outputs/behaviour24
python -m ah2ac2.tools.coverage_drop_dims \
  --src_dir outputs/behaviour24 --dst_dir outputs/behaviour19 --drop_blocks M10
python -m ah2ac2.tools.coverage_analyze --help
```

Proxy rollout subfolders use `BC_proxy_1`, `BC_proxy_2`, `Latent_proxy_1`, `Latent_proxy_2`, `Latent_proxy_3`, each containing `games.safetensors`. `coverage_analyze` takes the resulting human/proxy arrays and produces fitted-density diagnostics. Configure its input paths from `--help`.

The frozen paper coverage cloud and disjoint human support audit are distinct populations. For new experiments, disclose which data fit densities/weights and which data evaluate support. MMD measures symmetric distributional discrepancy; it does not establish directional coverage. Use human-to-mixture reverse KL and BC-tail rescue for that question.

## Scope of this source release

Included: the core latent construction and reconstruction pipeline, needed shared AH2AC2 modules, BC/best-response training, reference-sweep evaluation, coverage/support tools, CPU statistical primitives, calibration helpers and the signaling-game numerical analysis.

Excluded: private trajectories, learned checkpoint weights, fitted priors/calibration artifacts, raw score arrays, experiment outputs, reviewer correspondence, internal response-generation machinery and unrelated exploratory model families. The temporal audit and full experiment orchestration are not included in this minimal package; their main-paper findings are not being advertised as independently reproducible from these files alone.

The latent suite is a complementary evaluation probe. This release does not establish that its scores better predict human-agent coordination or that selected codes estimate convention prevalence.

## Provenance and publication

[`source_inventory.json`](source_inventory.json) records the originating workspace commit and hashes of copied modules, including research files that were uncommitted in that workspace. Numerical signaling/calibration helpers were extracted from the original experiment implementations; training objectives were retained. The release cleanup initializes the optional PPCA mixture fit, updates deprecated JAX tree-map calls, removes unused imports and applies consistent formatting. Workstation path prefixes were made relative and W&B defaults disabled where configured.

See [attribution and license status](ATTRIBUTION.md) before publication. Public repository: https://github.com/gfppoy/aht-hd.

### Naming and checkpoint compatibility

The model is named `VAE`; training modules and configs use `vae_2p`. PPO's discount coefficient is named `discount_factor` in code and YAML. Historical serialized VAE wrappers remain supported by the checkpoint loader; tensor names are preserved. Original filenames remain in `source_inventory.json` for provenance.

The NumPy-only [prior fitting helpers](ah2ac2/tools/prior_math.py) can also be exercised without the Hanabi training stack.
