# MV-REID: Multi-View Reidentification

## Installation

## Usage

### Training

To start training a model, use the `train` script provided. For example, to train a `vit_tiny` model on the `veri776` dataset with a learning rate of $1.5 \times 10^{-4}$, use the following script.

```bash
mvreid-train model=vit_tiny dataset=veri776 lr=1.5e-4
```

> [!NOTE]
> All hyperparameter configurations are handled by `hydra`. It is also possible to perform hyperparameter sweeps with the `-m` flag.

## Contributing

To contribute to the project, follow the steps below.

1. Clone the repo
  ```bash
  git clone https://github.com/sattwik-sahu/multi-view-reid.git
  ```
2. Install the packages using `uv`
  ```bash
  uv sync --all-groups
  ```
3. Now you can add new features

