# PyTorch Pruning Benchmarks

ResNet-18 experiments comparing magnitude pruning, gradient-signal preservation and regression-based channel pruning on CIFAR-10.

Maintained by [Ahmed Jawed](https://github.com/ahmedjawedaj).

## Methods and engineering work

- Layer sensitivity and global/local magnitude pruning.
- Iterative magnitude and GraSP-style saliency pruning.
- LASSO channel selection and least-squares filter reconstruction.
- Accuracy, parameter count, sparse representation and batch-latency comparisons.

## Repository contents

- [`pruning_experiments.ipynb`](notebooks/pruning_experiments.ipynb)
- [`data/README.md`](data/README.md): required external inputs and data handling.
- [`PROVENANCE.md`](PROVENANCE.md): original source-date evidence, attribution and publication changes.
- [`scripts/validate_sources.py`](scripts/validate_sources.py): notebook/Python syntax and exclusion checks.

## Recorded observations

- The selected source run records 11,173,962 → 2,239,048 parameters, a 79.96% reduction.
- Test top-1 changes from 93.07% to 91.00%, with a 1.362x batch-latency speedup in that measured workload.

## Setup and use

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd notebooks
jupyter lab
```

Select the intended kernel and run the notebook top to bottom after providing its required inputs. GPU notebooks can consume substantial runtime; validation below does not start training.

## Data and weights

The notebook downloads CIFAR-10 and a public CIFAR-10 ResNet-18 checkpoint. Training images, checkpoints and caches are excluded.

## Validation

```bash
python -m pip install -r requirements-validation.txt
python scripts/validate_sources.py
```

The publication checks validate notebook structure and compile source cells/Python modules. They do not reproduce training, GPU timing or ROS hardware behaviour. Cleaned notebooks have no captured outputs or stale execution counters.

## Scope and limitations

- A separate clean-notebook run records 90.64% and 1.412x. Those numbers are not merged with the selected run.
- Unstructured zeros did not automatically improve measured dense-kernel latency.
- The public pretrained checkpoint saw the original training pool, including later validation samples.

## References and attribution

- [PyTorch pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [CIFAR-10 checkpoint source](https://github.com/huyvnphan/PyTorch_CIFAR10)
- [GraSP reference](https://github.com/alecwangcq/GraSP)

See PROVENANCE.md for the distinction between original work dates and actual publication dates. This repository excludes instructor handouts, raw data, model weights and private records.
