#   TopicWeave

Enhancing scientific literature topic modeling through distilled model ensembles

##   Overview

TopicWeave is a novel approach to scientific literature topic modeling that leverages distilled model ensembles. It combines the complementary strengths of domain-specific language models (SciBERT) and citation-aware representations to create more coherent, interpretable, and scientifically relevant topics while maintaining computational efficiency.

The project addresses limitations in current topic modeling approaches for scientific literature, which often fail to capture domain-specific nuances due to specialized vocabulary and conceptual frameworks.

##   Architecture

TopicWeave uses a dual pathway architecture:

1.  **Domain Knowledge Pathway**: A distilled version of SciBERT that captures specialized scientific vocabulary and semantics.
2.  **Document Relationship Pathway**: A citation-aware model trained using contrastive learning to understand relationships between scientific content.

These complementary embeddings are combined through an optimized weighted ensemble to create a unified representation for topic modeling.

##   Repository Structure

    TopicWeave/
    ├── Data/                      # Data storage
    │   ├── processed/             # Preprocessed data (e.g., document_info.csv)
    │   │   └── document_info.csv  # CSV file containing document metadata and processed text
    │   └── raw/                   # Original arXiv dataset (to be downloaded)
    │
    ├── Notebooks/                 # Jupyter notebooks for experiments and analysis
    │   ├── 0_BERTopic_baseline.ipynb         # Baseline BERTopic implementation
    │   ├── 1_Topicweave_Distillation.ipynb   # Model distillation and SciBERT embedding generation
    │   ├── 2_Topicweave_Integration.ipynb   # SPECTER embedding generation and ensemble weighting
    │   ├── 3_Topicweave_Specter_Distillation.ipynb # SPECTER distillation
    │   ├── 4_TopicWeave_Finetune.ipynb      # Domain adaptation finetuning of SciBERT
    │   ├── 5_Topicweave_Finetune_Integration.ipynb # Integration of domain-adapted SciBERT
    │   ├── 6_TopicWeave_Evaluate.ipynb      # Evaluation of TopicWeave
    │   ├── 7_TopicWeave_Vizualizations.ipynb # Notebooks for visualizations
    │
    ├── Papers/                    # Project reports and papers
    │   ├── Final/                 # Final project paper (W266_Final_Project_Quddus_Chong.pdf)
    │   │   └── W266_Final_Project_Quddus_Chong.pdf
    │   ├── Milestone/             # Project milestone report (W266_Milestone.pdf)
    │   │   └── W266_Milestone.pdf
    │
    ├── Visualizations/            # Generated visualizations
    │   ├── architecture.png       # Diagram of the TopicWeave architecture
    │   ├── baseline_confusion_matrix.png # Confusion matrix for baseline
    │   ├── heatmap.png            # Heatmap visualization
    │   ├── top-topics.png         # Visualization of top topics
    │
    ├── README.md                  # Project documentation

##   Getting Started

###   Prerequisites

-   Python 3.8+
-   PyTorch 1.8+
-   Transformers 4.34+
-   Sentence-Transformers 2.2+
-   BERTopic
-   Scikit-learn
-   Numpy, Pandas
-   NLTK
-   Plotly, Matplotlib (for visualizations)

###   Installation

    # Clone the repository
    git clone https://github.com/your-username/TopicWeave.git
    cd TopicWeave
    
    # Set up virtual environment (Linux/macOS)
    python3 -m venv env
    source env/bin/activate
    
    # Set up virtual environment (Windows)
    python -m venv env
    .\env\Scripts\activate
    
    # Install dependencies
    pip install -r requirements.txt

###   Data

The project uses the arXiv metadata dataset, which should be placed in the `Data/raw/` directory. Due to size limitations, this file is not included in the repository.

To download the arXiv dataset:

1.  Visit [Kaggle arXiv Dataset](https://www.kaggle.com/Cornell-University/arxiv)
2.  Download `arxiv-metadata-oai-snapshot.json`
3.  Place it in the `Data/raw/` directory

###   Usage

1.  **Run Baseline Model**:

    ```bash
    jupyter notebook Notebooks/0_BERTopic_baseline.ipynb
    ```

    This notebook implements a standard BERTopic model using general-purpose embeddings.

2.  **Run TopicWeave Pipeline**:

    To run the full TopicWeave pipeline, execute the following notebooks in order:

    ```bash
    jupyter notebook Notebooks/1_Topicweave_Distillation.ipynb
    jupyter notebook Notebooks/3_Topicweave_Specter_Distillation.ipynb
    jupyter notebook Notebooks/4_TopicWeave_Finetune.ipynb
    jupyter notebook Notebooks/5_Topicweave_Finetune_Integration.ipynb
    jupyter notebook Notebooks/6_TopicWeave_Evaluate.ipynb
    jupyter notebook Notebooks/7_TopicWeave_Vizualizations.ipynb
    ```

    These notebooks cover the following steps:

    * `1_Topicweave_Distillation.ipynb`: Distills SciBERT and generates SciBERT embeddings.
    * `3_Topicweave_Specter_Distillation.ipynb`: Distills SPECTER.
    * `4_TopicWeave_Finetune.ipynb`: Fine-tunes SciBERT for domain adaptation.
    * `5_Topicweave_Finetune_Integration.ipynb`: Integrates SciBERT and SPECTER embeddings using optimized weights.
    * `6_TopicWeave_Evaluate.ipynb`: Evaluates the TopicWeave model.
    * `7_TopicWeave_Vizualizations.ipynb`: Generates visualizations.

##   Methodology

###   Distillation Approach

TopicWeave uses a hybrid approach of architectural pruning with knowledge distillation to create efficient models:

1.  **SciBERT Distillation**: We reduce the original 12-layer SciBERT to 6 layers while preserving domain-specific knowledge through knowledge distillation.

2.  **Citation-Aware Model**: We use a distilled version of SPECTER with 6 layers to capture semantic connections grounded in citation context. 

Sources and related content


###   Topic Modeling Pipeline

The full TopicWeave pipeline:

1.  Preprocess scientific documents
2.  Generate embeddings using distilled models
3.  Combine embeddings using optimized weights
4.  Apply BERTopic for topic extraction and visualization

##   Evaluation

We evaluate TopicWeave against a baseline using:

* **Quantitative metrics**: NMI, ARI
* **Qualitative metrics**: C\_v coherence, NPMI
* **Topic interpretability**: Through manual inspection and visualization

##   Citation

If you use TopicWeave in your research, please cite:

```bibtex
@misc{chong2025topicweave,
    title={TopicWeave: Enhancing Scientific Literature Topic Modeling through Distilled Model Ensembles},
    author={Chong, Quddus},
    year={2025},
    howpublished={GitHub repository},
    url={[https://github.com/your-username/TopicWeave](https://github.com/your-username/TopicWeave)}
  }
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* Jennifer Zhu for project guidance
* MIDS W266 course at UC Berkeley
* Authors of BERTopic, SciBERT, and SPECTER.