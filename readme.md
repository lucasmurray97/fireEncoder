# FireEncoder

FireEncoder is a research project that aims at combining latent representations and genetic algorithms in a novel way.

## Project structure:

```
├── data/                         # Datasets
├── src/                          # Source code
│   ├── algorithms/               # Source code for search algorithms
│   │   ├── eval/                 # Wrapper on simulator to evaluate solutions of search algorithms
│   │   └── GA/                   # Source code for genetic algorithms
│   ├── experiments/              # Generative model training results (logs, checkpoints, metrics)
│   ├── networks/                 # Architecture definitions for generative models
│   ├── utils/                    # Training utilities for generative models
│   └── train_ae.py               # Entry point to train generative models (VAE, CCVAE)
```

## Installation

To install FireEncoder, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/fireEncoder.git
cd fireEncoder
pip install -r requirements.txt
```


## Contributing

We welcome contributions to FireEncoder! Please fork the repository and submit a pull request with your changes. Make sure to follow the coding standards and include tests for any new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

