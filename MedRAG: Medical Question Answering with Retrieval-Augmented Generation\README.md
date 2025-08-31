# MedicalQA Retriever: A RAG Pipeline for Medical Literature

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Retrieval-Augmented Generation (RAG) system designed to answer complex medical questions by retrieving and synthesizing information from a corpus of scientific abstracts. This project demonstrates the application of modern NLP techniques to the biomedical domain.

## 🚀 Features

- **Data Processing:** Automated pipeline for cleaning, deduplication, and sentence-level chunking of medical abstracts.
- **Semantic Search:** Utilizes `all-MiniLM-L6-v2` sentence transformers and FAISS for efficient and accurate retrieval of relevant text passages.
- **Knowledge-Enhanced QA:** Integrates a quantized `BioMistral-7B` model to generate answers conditioned on the retrieved evidence, reducing hallucination.
- **Interactive Exploration:** Includes tools for visualizing the embedding space and testing the system with custom queries.

## Project Structure

**MedicalQA_Retriever.ipynb** – Main Jupyter notebook containing the entire pipeline

- **Data Loading & Cleaning**
- **Text Chunking & Embedding**
- **FAISS Index Construction**
- **Query Retrieval & Visualization**
- **Answer Generation with BioMistral-7B**


text

## 🛠️ Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone <your-repo-url>
    cd MedicalQA_Retriever
    ```

2.  **Create a Python environment and install dependencies**
    It is highly recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```
    *Note: A `requirements.txt` file is not provided in the notebook. You will need to create one from the imported libraries (e.g., `pandas`, `spacy`, `sentence-transformers`, `faiss-cpu` or `faiss-gpu`, `transformers`, `torch`, `matplotlib`, `scikit-learn`).*

3.  **Download the Data**
    The notebook expects data in `medical_articles_abstracts/raw/`. Please place your CSV files there. The dataset used was derived from the [PMC Articles dataset on Kaggle](https://www.kaggle.com/datasets/cvltmao/pmc-articles).

4.  **Download the SpaCy Model**
    ```bash
    python -m spacy download en_core_sci_sm
    ```

## 🏃‍♂️ Usage

1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook MedicalQA_Retriever.ipynb
    ```
2.  **Run the cells sequentially.** The notebook is designed to be executed from top to bottom.
3.  Key steps include:
    - Processing the raw abstract files into chunks.
    - Building and saving the FAISS index.
    - Embedding a query and retrieving the top-k most relevant chunks.
    - Generating a final answer using the BioMistral-7B model based on the retrieved context.

### Example

The notebook is configured with the example query:
> **"What were the health impacts of micronutrient deficiencies in South African children?"**

After running all cells, you will see the retrieved evidence chunks and the LLM's generated answer based on that evidence.

## 📊 Results & Evaluation

The system successfully retrieves relevant information about micronutrient deficiencies and child malnutrition in South Africa and generates a coherent summary. The t-SNE plot provides a visual confirmation that the retrieved passages are semantically close to the query in the embedding space.

*Note: This is a proof-of-concept. For a production system, formal evaluation using metrics like Hit Rate, Normalized Discounted Cumulative Gain (NDCG), or answer accuracy against a ground-truth dataset would be necessary.*

## ⚠️ Limitations & Future Work

- **Scale:** The current experiment runs on a subset of 1,000 abstracts per file due to computational constraints. Scaling to the full dataset would require more powerful hardware.
- **Model Quantization:** Using 4-bit quantization enables running the 7B model on consumer hardware but may slightly impair output quality.
- **Chunking Strategy:** The current sentence-based chunker can break context. Future work could experiment with more advanced strategies using LangChain or LlamaIndex.
- **Evaluation:** Implementing a rigorous quantitative evaluation framework is a key next step.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for the [Transformers](https://huggingface.co/docs/transformers) and [Sentence-Transformers](https://www.sbert.net/) libraries.
- Meta AI for [FAISS](https://github.com/facebookresearch/faiss).
- BioMistral for the [BioMistral-7B](https://huggingface.co/BioMistral/BioMistral-7B) model.
