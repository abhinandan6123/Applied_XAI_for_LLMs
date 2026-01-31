# Applied_XAI_for_LLMs
Applied study comparing explainability methods (IG, SHAP, Attention Rollout) on transformer-based NLP models.





---

## Overview

This repository contains an applied research study on explainability techniques for transformer-based language models. The goal of this work is to evaluate how common explainability methods behave in practice, rather than proposing new algorithms.

The project focuses on comparing Integrated Gradients, Attention Rollout, and SHAP on a fine-tuned DistilBERT model for sentiment analysis. The emphasis is on practical interpretability, stability, and usability in real-world NLP workflows.

This work was conducted as an academic research project in applied machine learning.

---

## Objectives

* Study practical explainability for transformer models
* Compare multiple XAI methods under the same setup
* Observe stability and interpretability differences
* Provide practitioner-oriented insights
* Maintain reproducible and structured experimentation

---

## Methods Used

### Model

* DistilBERT (fine-tuned)

### Dataset

* SST-2 Sentiment Classification
* Standard NLP benchmark dataset

### Explainability Techniques

* Integrated Gradients (Captum)
* Attention Rollout
* SHAP

---

## Project Workflow

### 1) Model Training

DistilBERT was fine-tuned on the SST-2 dataset using Hugging Face Transformers. The trained model and tokenizer were saved for reuse.

### 2) Explainability Analysis

Each explainability method was applied to the same model for consistency:

* Integrated Gradients for gradient-based attribution
* Attention Rollout for attention visualization
* SHAP for model-agnostic feature attribution

### 3) Observation and Comparison

Methods were compared based on:

* Stability
* Interpretability
* Practical usability

The focus was qualitative behavior rather than performance optimization.

---

## Key Observations

* Integrated Gradients produced relatively stable token attributions
* Attention explanations were efficient but not always aligned with sentiment words
* SHAP provided flexible explanations but with higher computation cost and variability

These observations highlight trade-offs practitioners may encounter when selecting XAI tools.

---

## Limitations

* SST-2 represents a narrow domain
* DistilBERT is smaller than modern large LLMs
* Explanations rely on human interpretation

This study does not claim generalization to all LLM settings.

---

## Repository Structure

```
1_Code/ → Training and XAI notebooks  
2_Models/ → Saved model and tokenizer  
3_Plots/ → Explanation visualizations  
4_Research_Information/ → Section notes and drafts  
Reports/ → Compiled reports  
Workflow/ → Pipeline documentation  
References/ → Research sources
```

---

## Reproducibility

Install dependencies:

```
pip install -r requirements.txt
```

Then run notebooks in order:

1. DistilBERT training
2. Integrated Gradients
3. Attention Rollout
4. SHAP analysis

---

## Intended Use

This repository is for:

* Academic study
* Learning explainability in NLP
* Demonstrating applied ML workflow

Not intended for production deployment.

---

## References

Core references include work on BERT, Integrated Gradients, SHAP, Attention analysis, and XAI evaluation. See the References folder for details.

---

## Author

Venkata Abhinandan Kancharla

---
