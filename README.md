# DrugFlow-RL

This repository contains the implementation of the DrugFlow-RL framework, which implements online reinforcement learning for pocket-conditioned molecular generation. We build on top of the structure-based drug design model [DrugFlow](https://arxiv.org/abs/2508.17815). The main online RL framework implement is based on [DiffusionNFT](https://arxiv.org/abs/2509.16117) and code for our [FlowGRPO](https://arxiv.org/abs/2505.05470) adaptation is also present. Code to perform direct preference optimization was implemented previously by DrugFlow and is also available in this repository.

## Abstract

Structure-based drug design has strong potential to accelerate early-stage drug discovery enabling conditional generation of target-specific molecules. However, a major bottleneck remains the poor synthesizability of generated compounds, which limits their direct experimental validation. We address this challenge by optimizing for membership in make-on-demand chemical libraries, ensuring that generated molecules can be directly purchased and tested. To this end, we apply reinforcement learning to a pocket-conditioned generative molecular model, developing techniques to handle the mixed continuous-discrete action space of molecular generation. This approach substantially increases the fraction of samples found in virtual chemical spaces, raising it from 11.8% to 82.5% purchasable candidates for the KRAS protein.

## Setup

A conda environment can be set up using the provided `environment.yaml` file. It is almost identical to the one provided by DrugFlow with the exception of the `hydra` library, which is used for configuration management in this codebase, and the `rdkit` version, which is updated to 2024.09.5 to ensure compatibility with the synthon search for querying the Freedom space.

```bash
conda env create -f environment.yaml -n drugflow-rl
conda activate drugflow-rl
```

The docker container provided by DrugFlow can still be used for sampling. For further information, please refer to the [DrugFlow repository](https://github.com/LPDI-EPFL/DrugFlow).

## Performing RL Optimization

RL optimization can be performed by running the `main.py` script and all relevant parameters can be set through the provided configuration files in the `configs` directory.

```bash
python main.py
```

A prior DrugFlow checkpoint is required to perform RL optimization, which can be downloaded from the [DrugFlow repository](https://github.com/LPDI-EPFL/DrugFlow/tree/main?tab=readme-ov-file#models) (checkpoint `DrugFlow`). Save the checkpoint under `checkpoints/drugflow.ckpt` or alter the path in `configs/model/drugflow.yaml` under `model_path`.

### Diversity filter

To counteract mode collapse (the policy repeatedly sampling the same molecules or chemotypes), a diversity filter is available inspired by [REINVENT](https://doi.org/10.1021/acs.jcim.0c00915) and [Augmented Hill-Climb](https://doi.org/10.1186/s13321-022-00646-z) (`configs/diversity_filter/`). Generated molecules are assigned a bucket key, an occurrence count per bucket is maintained over a configurable window, and the reward of over-represented buckets is discounted **before advantages are computed**, pushing the policy to spread its mass. It is **disabled by default** (`diversity_filter=none`).

Two ready-made presets are provided:

```bash
# molecular-uniqueness preset (penalizes re-generating the same molecule)
python main.py diversity_filter=uniqueness

# scaffold-diversity preset (REINVENT-like, penalizes over-used scaffolds)
python main.py diversity_filter=reinvent
```

Any option can be overridden on the command line, e.g.:

```bash
python main.py diversity_filter=uniqueness \
       diversity_filter.scale=3 diversity_filter.window=40 diversity_filter.minscore=0.4
```

**Options** (three orthogonal knobs plus thresholds):

| Option | Choices / type | Meaning |
| --- | --- | --- |
| `name` | `NoFilter`, `IdenticalMolecule`, `IdenticalMurckoScaffold`, `IdenticalTopologicalScaffold`, `ScaffoldSimilarity` | Bucket key. `IdenticalMolecule` (canonical SMILES) targets molecular uniqueness; the scaffold variants target chemotype diversity; `NoFilter` disables the filter. |
| `scope` | `batch`, `window`, `global` | Window over which occurrences are counted. `global` = whole run (REINVENT-like, non-stationary, saturates late); `window` = last `window` steps (stationary signal that shapes the final policy); `batch` = current step only. |
| `penalty` | `ramp`, `binary`, `continuous` | Count→multiplier mapping. `ramp` = full reward up to `tol`, linear decay to 0 at `buff`; `binary` = full reward below `buff`, 0 above (REINVENT hard cutoff); `continuous` = `1 / (1 + count/scale)`, penalizes from the first repeat, never exactly 0, always keeps an ordered gradient. |
| `minscore` | float | Only molecules with reward ≥ `minscore` count / get penalized (keeps low-quality samples out of the buckets). |
| `tol`, `buff` | int | Thresholds for `ramp` / `binary`. |
| `scale` | float | Parameter for `continuous` (smaller → harsher; multiplier = 0.5 at `count = scale`). |
| `window` | int | Number of steps kept when `scope=window`. |
| `minsimilarity`, `radius`, `nbits` | float, int, int | `ScaffoldSimilarity` only: Tanimoto threshold and Morgan-fingerprint settings for clustering scaffolds. |

Notes:
- Within a step, samples are already deduplicated by uniqueness before scoring, so molecule-level (`IdenticalMolecule`) pressure only registers across steps — use `scope=window` or `global`, not `batch`.
- The `continuous` penalty never reaches exactly 0, so it cannot zero out a whole batch and collapse the batch-centered advantage signal — a safer default than the `binary`/`ramp` hard floor when tuning aggressively.
- Filter statistics (`penalized`, `buckets`, `mean_multiplier`) are logged under `diversity_filter/*`, alongside `reward/filtered_mean`.

## Sampling

To sample molecule de novo for a given pocket, the `generate.py` script can be used as follows:

```bash
python src/generate.py \
       --protein path/to/protein.pdb \
       --ref_ligand path/to/reference_ligand.sdf \
       --checkpoint path/to/checkpoint.ckpt \
       --datadir path/to/sbdd_rl_repo/data \
       --output path/to/samples.sdf \
       --n_samples 1000 \
       --n_steps 500 \
       --batch_size 64 \
       --add_virtual_max 5 \
       --noise_scale 10.0
```

The folder `data` contains the necessary files for sampling, including the reference ligand and pocket files for the KRAS protein. The `checkpoint` argument can be set to either a prior DrugFlow checkpoint or an RL-optimized checkpoint. `n_steps` controls the number of sampling steps with higher values leading to better sample quality at the cost of increased sampling time. `noise_scale` controls the amount of noise added for stochastic sampling (SDE), while setting it to None will perform deterministic sampling (ODE). `add_virtual_max` controls the maximum number of virtual atoms that can be added on top of the number of nodes sampled given the number of pocket residues. These parameters correspond to the ones used in the paper for evaluation with SDE sampling.

## Checkpoints

Optimized model checkpoints are provided on [figshare](10.6084/m9.figshare.31148611).