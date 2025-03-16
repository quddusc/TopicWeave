# TopicWeave

Enhancing scientific literature topic modeling through distilled model ensembles

## Overview

TopicWeave is a novel approach to scientific literature topic modeling that leverages distilled model ensembles. It combines the complementary strengths of domain-specific language models (SciBERT) and citation-aware representations to create more coherent, interpretable, and scientifically relevant topics while maintaining computational efficiency.

The project addresses limitations in current topic modeling approaches for scientific literature, which often fail to capture domain-specific nuances due to specialized vocabulary and conceptual frameworks.

## Architecture

TopicWeave uses a dual pathway architecture:

1. **Domain Knowledge Pathway**: A distilled version of SciBERT that captures specialized scientific vocabulary and semantics
2. **Document Relationship Pathway**: A citation-aware model trained using contrastive learning to understand relationships between scientific content

These complementary embeddings are combined through an optimized weighted ensemble to create a unified representation for topic modeling.

## Repository Structure

```
TopicWeave/
├── Notebooks/                 # Jupyter notebooks for experiments and analysis
│   ├── baseline.ipynb         # Baseline BERTopic implementation
│   └── distillation.ipynb     # Model distillation and embedding generation
│
├── Data/                      # Data storage
│   ├── raw/                   # Original arXiv dataset
│   └── processed/             # Preprocessed data
│
│   └── topicweave/            # TopicWeave ensemble embeddings
│
└── README.md                  # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- Transformers 4.34+
- Sentence-Transformers 2.2+
- BERTopic
- Scikit-learn
- Numpy, Pandas
- NLTK
- Plotly, Matplotlib (for visualizations)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/TopicWeave.git
cd TopicWeave

# Set up virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data

The project uses the arXiv metadata dataset, which should be placed in the `Data/raw/` directory. Due to size limitations, this file is not included in the repository.

To download the arXiv dataset:
1. Visit [Kaggle arXiv Dataset](https://www.kaggle.com/Cornell-University/arxiv)
2. Download `arxiv-metadata-oai-snapshot.json`
3. Place it in the `Data/raw/` directory

### Usage

1. **Run Baseline Model**:
   ```bash
   jupyter notebook Notebooks/baseline.ipynb
   ```
   This notebook implements a standard BERTopic model using general-purpose embeddings.

2. **Run TopicWeave Pipeline**:
   ```bash
   jupyter notebook Notebooks/distillation.ipynb
   ```
   This notebook implements the distillation of SciBERT and the citation-aware model, generates embeddings, and combines them using optimized weights.

## Methodology

### Distillation Approach

TopicWeave uses a hybrid approach of architectural pruning with knowledge distillation to create efficient models:

1. **SciBERT Distillation**: We reduce the original 12-layer SciBERT to 6 layers while preserving domain-specific knowledge through knowledge distillation.

2. **Citation-Aware Model**: Inspired by SimCSE (Gao et al., 2021), we train a 6-layer model using contrastive learning to understand relationships between sections of the same document versus sections from different documents, simulating citation knowledge.

### Topic Modeling Pipeline

The full TopicWeave pipeline:
1. Preprocess scientific documents
2. Generate embeddings using distilled models
3. Combine embeddings using optimized weights
4. Apply BERTopic for topic extraction and visualization

## Evaluation

We evaluate TopicWeave against a baseline using:

- **Quantitative metrics**: NMI, ARI
- **Qualitative metrics**: C_v coherence, NPMI
- **Topic interpretability**: Through manual inspection and visualization

## Citation

If you use TopicWeave in your research, please cite:

```bibtex
@misc{chong2025topicweave,
  title={TopicWeave: Enhancing Scientific Literature Topic Modeling through Distilled Model Ensembles},
  author={Chong, Quddus},
  year={2025},
  howpublished={GitHub repository},
  url={https://github.com/your-username/TopicWeave}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Jennifer Zhu for project guidance
- MIDS W266 course at UC Berkeley
- Authors of BERTopic, SciBERT, and SimCSE