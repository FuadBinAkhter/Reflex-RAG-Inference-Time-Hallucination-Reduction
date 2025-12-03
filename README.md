# Reflex-RAG: Inference-Time Hallucination Reduction Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Model](https://img.shields.io/badge/Llama--3.1--8B-Quantized-purple?style=for-the-badge)
![Technique](https://img.shields.io/badge/Inference--Time-Compute-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-grey?style=for-the-badge)

## 🔬 Abstract

**Reflex-RAG** is an Agentic Retrieval-Augmented Generation (RAG) system designed to mitigate hallucinations in Small Language Models (SLMs) without the need for expensive fine-tuning.

Leveraging the concept of **Inference-Time Compute**, this pipeline implements a **"Best-of-N" Rejection Sampling** algorithm. Instead of relying on a single generative pass, the system generates multiple candidate reasoning paths and rigorously evaluates them against dynamically retrieved ground-truth data (Wikipedia) using a weighted dual-metric scoring system.

Optimized for consumer hardware, the system utilizes **4-bit quantization (NF4)** to execute a **Llama-3.1-8B** reasoning engine efficiently on NVIDIA T4 GPU.

---

## 🏗 Architecture & Methodology

The pipeline follows a **Generate $\rightarrow$ Verify $\rightarrow$ Select** workflow:

### 1. Dynamic Context Retrieval
The system queries the **Wikipedia API** to fetch real-time, domain-specific context based on the user prompt. This step ensures the model is grounded in up-to-date facts before generation begins.

### 2. Parallel Candidate Generation (Diversity Phase)
Using a high temperature ($T=0.85$), the Llama-3 model generates $N$ distinct candidate answers. This introduces semantic entropy, compelling the model to explore diverse reasoning paths rather than converging on a single, potentially hallucinated mode.

### 3. Dual-Metric Scoring (Verification Phase)
Each candidate is graded based on a weighted formula designed to balance external grounding with internal coherence:

$$Score = \alpha \cdot Sim(A, C) + (1 - \alpha) \cdot Conf(A)$$

Where:
* **$Sim(A, C)$**: Semantic Vector Similarity (Cosine) between the Answer ($A$) and Context ($C$). This is calculated using the **BAAI/bge-base-en-v1.5** embedding model.
* **$Conf(A)$**: The LLM's intrinsic Self-Confidence Score (0.0 - 1.0), obtained via a secondary strict evaluation prompt.

### 4. Rejection Sampling
The candidate maximizing the $Score$ is selected as the final output. Candidates that fail to align with the retrieved context or exhibit low confidence are effectively filtered out.

---

## 📊 Performance & Optimization

To quantify the mitigation of hallucinations, the pipeline was benchmarked against the **TruthfulQA** dataset (Generation Task). A systematic **Hyperparameter Grid Search** was performed to determine the optimal equilibrium between the model's generative entropy (Temperature) and its reliance on retrieved verification (Alpha).

### Comparative Results

The following table contrasts the standard "out-of-the-box" model performance against the optimized Reflex-RAG pipeline.

| Configuration | Temperature ($T$) | Retrieval Weight ($\alpha$) | Accuracy Score |
| :--- | :---: | :---: | :---: |
| **Baseline (Standard)** | 0.1 | N/A | 68% |
| **Reflex-RAG (Optimized)** | **0.8** | **0.7** | **84%** |

### Key Findings

* **The Baseline (Control Group):**
    Utilizing standard **Greedy Decoding** ($T=0.1$), the model operates deterministically. In the absence of the Reflex pipeline, the model achieved a **68% accuracy score**, frequently hallucinating incorrect details despite exhibiting high internal confidence.

* **The Optimization (Reflex-RAG):**
    By increasing entropy ($T=0.8$) to generate diverse candidate paths and implementing the weighted voting mechanism, the system achieved an **84% accuracy score**.
    > **Impact:** This represents a **~16% absolute improvement** in factual grounding.

* **The "Optimal Point" ($\alpha = 0.7$):**
    Grid search analysis identified that an Alpha of **0.7** yields maximal performance. This indicates that for 8B-parameter models, the scoring algorithm must weight **retrieved context (70%)** significantly higher than the **model's intrinsic confidence (30%)** to effectively suppress hallucinations.

---

## 📂 Repository Structure

The project is structured as a modular Python package:

```text
Reflex-RAG/
├── src/
│   ├── __init__.py
│   ├── config.py           # Hardware settings (Device, Quantization) & Hyperparameters
│   ├── models.py           # Singleton loader for Llama-3 (4-bit) & BGE Embeddings
│   ├── retrieval.py        # External knowledge retrieval logic (Wikipedia)
│   ├── pipeline.py         # Best-of-N Voting & Scoring Algorithms
│   └── evaluation.py       # TruthfulQA Benchmarking Engine
├── main.py                 # Entry point for optimization & inference
├── requirements.txt        # Project dependencies
└── README.md               # Documentation
