# Hybrid Dense-Sparse RAG for ESG Document Analysis

This repository contains a complete pipeline for processing, retrieving, and answering questions from ESG (Environmental, Social, and Governance) PDF reports. It uses a **Hybrid Dense-Sparse Retrieval** approach combined with **Reciprocal Rank Fusion (RRF)** to maximize retrieval accuracy across complex financial documents.

## Architecture Diagram

```mermaid
graph TD
    %% Ingestion Flow
    subgraph Ingestion Pipeline
        A[ESG PDF Reports] -->|PyMuPDF + pdfplumber| B(Per-Page Images & OCR Text)
        B --> C[Dense Encoder]
        B --> D[Sparse Indexer]
        
        C -->|sentence-transformers| E[(Dense Vector Index)]
        D -->|BM25Okapi| F[(Keyword Index)]
    end

    %% Retrieval & Generation Flow
    subgraph Retrieval & Generation
        G[User Query] --> H{Query Embedder}
        G --> I{Keyword Matcher}
        
        H -->|Semantic Search| E
        I -->|Lexical Search| F
        
        E --> J(Top-K Dense Results)
        F --> K(Top-K Sparse Results)
        
        J --> L((Reciprocal Rank Fusion))
        K --> L
        
        L --> M[Top-K Hybrid Ranked Pages]
        
        M --> N[Gemini 2.5 Flash / GPT-4o]
        G --> N
        
        N --> O[Structured Answer with Citations]
    end
    
    classDef primary fill:#2980b9,stroke:#34495e,stroke-width:2px,color:white;
    classDef secondary fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:white;
    classDef database fill:#f39c12,stroke:#e67e22,stroke-width:2px,color:white;
    
    class L,N primary;
    class C,D secondary;
    class E,F database;
```

## What This Pipeline Does

1. **Ingestion**: Parses raw ESG PDFs into page-level images and extracts the text using Optical Character Recognition (OCR).
2. **Dense Retrieval**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to create semantic vector embeddings of the page text. This runs efficiently on a CPU.
3. **Sparse Retrieval**: Uses a `BM25` index on the extracted text for exact-keyword matching (essential for specific metrics and company names).
4. **Hybrid Fusion**: Merges the results of both dense and sparse retrieval using Reciprocal Rank Fusion (RRF) to ensure the best pages bubble to the top.
5. **Generation**: Passes the retrieved context and user query to an LLM (like **Gemini 2.5 Flash**) to generate a highly accurate, structured answer with strict citations to the source pages.

## Getting Started

### Running the Notebook

The easiest way to run the entire pipeline is using the provided Jupyter Notebook: `Hybrid_Dense_Sparse_RAG.ipynb`.

1. **Open the Notebook**: You can open this notebook in Jupyter, Google Colab, or Kaggle. 
2. **Set your API Key**: Make sure you have your LLM API Key (e.g., Google Gemini API Key) securely set in your environment variables or secret manager (Do **NOT** hardcode it into the notebook).
3. **Upload Data**: Place your target PDF documents in the specified dataset directory (e.g., a `data/` folder).
4. **Run All**: Execute the cells from top to bottom. The notebook will automatically install dependencies, build the indexes, and run test queries.

### Dependencies

The project relies on standard, lightweight Python libraries:
- `sentence-transformers`
- `rank-bm25`
- `PyMuPDF`
- `google-generativeai`

No heavy GPU dependencies are strictly required since the dense retrieval model is highly optimized.

## License

This project is open-source and available for research and educational purposes.
