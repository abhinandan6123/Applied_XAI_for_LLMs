<div align="center">

<img src="https://img.shields.io/badge/Status-Preprint-yellow?style=for-the-badge&logo=checkmarx&logoColor=white" />
<img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19096514-blue?style=for-the-badge&logo=zenodo&logoColor=white" />
<img src="https://img.shields.io/badge/Model-DistilBERT-orange?style=for-the-badge&logo=huggingface&logoColor=white" />
<img src="https://img.shields.io/badge/Dataset-SST--2-purple?style=for-the-badge&logo=databricks&logoColor=white" />
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />

<br/>

<a href="https://github.com/abhinandan6123/Applied_XAI_for_LLMs/stargazers">
  <img src="https://img.shields.io/github/stars/abhinandan6123/Applied_XAI_for_LLMs?style=for-the-badge&logo=github&color=gold&logoColor=white" alt="GitHub Stars" />
</a>
<img src="https://img.shields.io/github/forks/abhinandan6123/Applied_XAI_for_LLMs?style=for-the-badge&logo=github&color=gray&logoColor=white" />

<br/><br/>

<h1>
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&duration=3000&pause=1000&color=4F8EF7&center=true&vCenter=true&width=800&lines=Applied+Explainability+for+LLMs;A+Comparative+Study+%E2%80%94+IG+%7C+SHAP+%7C+Attention+Rollout;Research+by+Venkata+Abhinandan+Kancharla" alt="Typing SVG" />
</h1>

<br/>

<blockquote>
<i>"Explainability is not optional — it is the foundation of trust, accountability, and responsible AI deployment."</i>
</blockquote>

<br/>

**📄 Research Paper** · **[DOI: 10.5281/zenodo.19096514](https://doi.org/10.5281/zenodo.19096514)** · **[Portfolio](https://abhikancharla.vercel.app)** · **[LinkedIn](https://www.linkedin.com/in/venkata-abhinandan-kancharla)**

<br/>

> ⚠️ **Note:** This work is a **research preprint** and has **not been peer-reviewed**. Results reflect findings from a controlled applied study on DistilBERT + SST-2.

<br/>

</div>

---

## 🧭 At a Glance

| Property | Detail |
|---|---|
| **Research Type** | Applied Comparative Study |
| **Domain** | Explainable AI · NLP · Transformer Models |
| **Model** | DistilBERT (fine-tuned) |
| **Dataset** | SST-2 Sentiment Classification |
| **XAI Methods** | Integrated Gradients · Attention Rollout · SHAP |
| **Framework** | PyTorch · Hugging Face · Captum |
| **Evaluation** | Faithfulness · Stability · Human Interpretability |
| **Author** | Venkata Abhinandan Kancharla |
| **Affiliation** | NRI Institute of Technology |
| **ORCID** | [0009-0000-2464-313X](https://orcid.org/0009-0000-2464-313X) |

---

## 🎯 Problem Statement

Large Language Models (LLMs) achieve remarkable performance across NLP tasks — yet their internal decision-making processes remain largely opaque. This **black-box nature** creates critical barriers in:

- 🐛 **Debugging** model failures and diagnosing misclassifications
- 🔍 **Auditing** for bias, spurious correlations, and data artifacts  
- ✅ **Validating** that model reasoning aligns with domain expectations
- 🏛️ **Accountability** in high-stakes deployment scenarios

Despite a rich XAI literature, **the practical gap remains**: existing explainability methods are rarely evaluated under consistent, reproducible settings that reflect real-world engineering constraints.

> This study fills that gap — not by proposing new methods, but by rigorously comparing three established techniques under identical conditions, providing **practitioner-grade insight** into what actually works.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     RESEARCH PIPELINE                           │
│                                                                 │
│  SST-2 Dataset ──► DistilBERT Fine-tuning ──► Frozen Checkpoint│
│                              │                                  │
│               ┌──────────────┼──────────────┐                  │
│               ▼              ▼              ▼                   │
│     Integrated Gradients  Attention     SHAP (Kernel)           │
│        (Captum)           Rollout       Model-Agnostic          │
│               │              │              │                   │
│               └──────────────┼──────────────┘                  │
│                              ▼                                  │
│              Qualitative & Quantitative Analysis                │
│           (Faithfulness · Stability · Interpretability)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Methods Deep Dive

### 1️⃣ Integrated Gradients (IG)

> Gradient-based attribution via path integrals from baseline to input.

**Theoretical basis:** Axiomatic attribution (Sundararajan et al., ICML 2017) — computes the integral of gradients along a straight-line path from a neutral baseline (zero embedding) to the actual input.

**How it works:**
```
Attribution(xᵢ) = (xᵢ - x'ᵢ) × ∫₀¹ [∂F(x' + α(x-x')) / ∂xᵢ] dα
```
Where `x'` is the baseline and `F` is the model output.

**Implementation:** Captum library · token-level attribution scores · bar plot visualisation

**Key finding:** Consistently highlighted sentiment-bearing tokens (adjectives, negations, intensifiers) with stable attribution magnitudes across repeated evaluations.

---

### 2️⃣ Attention Rollout

> Aggregated multi-layer attention propagation from [CLS] token.

**Theoretical basis:** Quantifying Attention Flow (Abnar & Zuidema, ACL 2020) — recursively multiplies attention matrices across transformer layers accounting for residual connections.

**How it works:**
```
Ã⁽ˡ⁾ = 0.5 × A⁽ˡ⁾ + 0.5 × I    (residual-adjusted attention)
Rollout = Ã⁽ᴸ⁾ × Ã⁽ᴸ⁻¹⁾ × ... × Ã⁽¹⁾
```

**Implementation:** Layer-wise attention matrix propagation · [CLS] token focus · visualised as token attribution heatmaps

**Key finding:** Computationally fastest method but frequently emphasised structural/syntactic tokens ([CLS], stopwords, punctuation) rather than semantically meaningful tokens.

---

### 3️⃣ SHAP (Kernel SHAP)

> Model-agnostic Shapley value attribution via coalition sampling.

**Theoretical basis:** SHAP (Lundberg & Lee, NeurIPS 2017) — grounded in cooperative game theory, assigns each token a Shapley value representing its marginal contribution to the prediction.

**How it works:**
```
φᵢ = Σ [|S|!(M-|S|-1)!/M!] × [f(S∪{i}) - f(S)]
     S⊆M\{i}
```
Where `M` is the feature set and `f(S)` is the model output with coalition `S`.

**Implementation:** Kernel SHAP with reshaped input representations · HTML visualisation output · model-agnostic (no gradient access required)

**Key finding:** Most flexible across model types, but exhibited significant variability across runs due to sensitivity to background data selection and input perturbation sampling.

---

## 📊 Results Summary

### Quantitative Observations

| Method | Attribution Stability | Computation Speed | Faithfulness | Scalability |
|---|---|---|---|---|
| **Integrated Gradients** | ✅ High | ⚡ Moderate | ✅ High | ✅ Good |
| **Attention Rollout** | ⚠️ Moderate | ✅ Fast | ❌ Low | ✅ Excellent |
| **SHAP (Kernel)** | ❌ Variable | ❌ Slow | ⚠️ Moderate | ❌ Limited |

### Qualitative Findings

**Integrated Gradients** consistently identified sentiment-relevant tokens:
```
Input:  "The movie was absolutely wonderful and engaging."
        [CLS]  the   movie  was  absolutely  wonderful  and  engaging  [SEP]
IG:      0.21  0.18  0.20  0.08    0.65        1.18     0.48   0.52    0.09
                                   ↑ Strong positive attribution on sentiment words
```

**Attention Rollout** showed structural bias:
```
Input:  "The movie was absolutely wonderful and engaging."
        [CLS]  the   movie  was  absolutely  wonderful  and  engaging  [SEP]
AR:      0.32  0.07  0.09  0.05    0.08        0.11     0.06   0.08    0.26
         ↑ Disproportionate focus on [CLS] and structural tokens
```

**SHAP** identified sentiment regions but with noise:
```
Output: Whole-sentence attribution → identified positive region
        Token-level: unstable across runs with varying background configurations
```

### Trade-off Summary

```
┌─────────────────┬────────────────────────┬──────────────────────────────┐
│ Method          │ Strengths              │ Limitations                  │
├─────────────────┼────────────────────────┼──────────────────────────────┤
│ Integrated      │ High faithfulness,     │ Requires gradient access     │
│ Gradients       │ stable, interpretable  │ and careful baseline choice  │
├─────────────────┼────────────────────────┼──────────────────────────────┤
│ Attention       │ Fast, easy to apply,   │ Poor alignment with          │
│ Rollout         │ no extra computation   │ prediction-relevant features │
├─────────────────┼────────────────────────┼──────────────────────────────┤
│ SHAP            │ Model-agnostic,        │ High compute cost, unstable  │
│ (Kernel)        │ flexible framework     │ in transformer text settings │
└─────────────────┴────────────────────────┴──────────────────────────────┘
```

---

## 📸 Result Screenshots

> Figures reproduced from the published paper (pages 8–9). All outputs generated from the **same frozen DistilBERT checkpoint** for cross-method consistency.

### Figure 1 — Integrated Gradients: Token Attribution (Positive Review)

![Integrated Gradients Attribution](3_Plots/ig_attribution_positive.png)

> *Sentiment-bearing words like **"wonderful"** (score ≈ 1.18) and **"engaging"** (score ≈ 0.52) receive the highest attribution scores. The method cleanly separates meaningful tokens from structural noise.*

**What this tells us:** IG directly tracks gradient sensitivity through the model — tokens that most influence the positive classification are highlighted with high fidelity. Attribution scores are stable across repeated evaluations of semantically similar inputs.

---

### Figure 2 — Attention Rollout: CLS Token Attribution

![Attention Rollout Attribution](3_Plots/attention_rollout_cls.png)

> *The [CLS] token (score ≈ 0.32) and [SEP] token (score ≈ 0.26) dominate — structural artifacts, not sentiment signals. Sentiment words like **"wonderful"** and **"engaging"** receive comparatively low attention weight.*

**What this tells us:** Attention Rollout propagates attention across all layers but inherits the architectural bias toward positional/special tokens. This confirms prior findings (Jain & Wallace, 2019) that attention ≠ explanation.

---

### Figure 3 — SHAP: Kernel Attribution Output

![SHAP Attribution](3_Plots/shap_html_output.png)

> *SHAP identifies the overall positive sentiment direction but produces noisy, less stable token-level attributions. Output value shown: **0.00004568** (base value) with the sentence scored as Output 1 (positive).*

**What this tells us:** SHAP's model-agnostic design allows it to work without gradient access, but Kernel SHAP applied to transformers suffers from sensitivity to background data sampling — making routine use impractical without careful configuration.

---

### Side-by-Side Comparison

| | Integrated Gradients | Attention Rollout | SHAP |
|---|---|---|---|
| **Top token** | "wonderful" (1.18) | [CLS] (0.32) | sentence-level |
| **Sentiment aligned?** | ✅ Yes | ❌ No | ⚠️ Partial |
| **Stable across runs?** | ✅ Yes | ⚠️ Moderate | ❌ Variable |
| **Compute cost** | Medium | Very Low | High |
| **Best use case** | Production debugging | Quick exploration | Model-agnostic audit |

---

## 📁 Repository Structure

```
Applied_XAI_for_LLMs/
│
├── 📂 1_Code/
│   ├── 01_distilbert_training.ipynb       # Fine-tuning pipeline
│   ├── 02_integrated_gradients.ipynb      # IG attribution analysis
│   ├── 03_attention_rollout.ipynb         # Layer-wise attention propagation
│   └── 04_shap_analysis.ipynb             # Kernel SHAP with HTML output
│
├── 📂 2_Models/
│   ├── tokenizer/                        # Saved tokenizer artifacts
│
├── 📂 3_Plots/
│   ├── ig_attribution_positive.png        # Figure 1: IG on positive review
│   ├── attention_rollout_cls.png          # Figure 2: Rollout [CLS] focus
│   └── shap_html_output.html              # Figure 3: SHAP token attribution
│
│
├── 📂 Research Paper/                    # End-to-end final research documentation
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites

```bash
Python >= 3.8
PyTorch >= 1.12
CUDA (optional, recommended for training)
```

### Installation

```bash
git clone https://github.com/abhinandan6123/Applied_XAI_for_LLMs.git
cd Applied_XAI_for_LLMs
pip install -r requirements.txt
```

### Dependencies

```txt
transformers>=4.30.0
torch>=1.12.0
captum>=0.6.0
shap>=0.41.0
datasets>=2.12.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
numpy>=1.24.0
jupyter>=1.0.0
```

### Run Pipeline (in order)

```bash
# Step 1 — Fine-tune DistilBERT on SST-2
jupyter nbconvert --to notebook --execute 1_Code/01_distilbert_training.ipynb

# Step 2 — Generate Integrated Gradients attributions
jupyter nbconvert --to notebook --execute 1_Code/02_integrated_gradients.ipynb

# Step 3 — Run Attention Rollout
jupyter nbconvert --to notebook --execute 1_Code/03_attention_rollout.ipynb

# Step 4 — Run SHAP analysis
jupyter nbconvert --to notebook --execute 1_Code/04_shap_analysis.ipynb
```

---

## 🧠 Key Insights for Practitioners

### When to use Integrated Gradients ✅
- Production debugging of transformer-based NLP models
- When stable, repeatable explanations are required
- Token-level attribution for classification tasks
- Model validation and bias investigation pipelines

### When to use Attention Rollout ⚡
- Rapid exploratory analysis at scale
- When computational budget is tight
- As a supplementary (not primary) interpretability lens
- Visualising broad attention flow patterns

### When to use SHAP ⚠️
- Model-agnostic scenarios (no gradient access)
- Targeted qualitative analysis on specific examples  
- Comparing across heterogeneous model architectures
- When theoretical grounding (Shapley values) is required for stakeholders

> **Core recommendation:** Treat explanations as **diagnostic aids, not ground truth**. No single method captures the full picture — use IG as primary, complement with SHAP for model-agnostic validation, and use Attention Rollout for speed only.

---

## 📚 Research Context

### Taxonomy of XAI Methods for LLMs

```
XAI Methods for Transformers
│
├── Attention-Based
│   ├── BERTViz — multi-head attention visualisation [Vig, 2019]
│   └── Attention Rollout — propagated layer attention [Abnar & Zuidema, 2020]
│
├── Gradient-Based  ← PRIMARY FOCUS
│   ├── Vanilla Gradients — input × gradient
│   ├── Integrated Gradients — path integral attribution [Sundararajan et al., 2017]
│   └── SmoothGrad — noise-averaged gradients
│
├── Feature Attribution (Model-Agnostic)
│   ├── SHAP — Shapley value framework [Lundberg & Lee, 2017]
│   ├── LIME — local surrogate models
│   └── Kernel SHAP — sampling-based Shapley estimation
│
└── Example-Based
    ├── TracIn — training data influence [Pruthi et al., 2020]
    └── Influence Functions — second-order influence estimation
```

### Selected References

| # | Authors | Title | Venue |
|---|---|---|---|
| [1] | Vaswani et al. | Attention Is All You Need | NeurIPS 2017 |
| [2] | Devlin et al. | BERT: Pre-training of Deep Bidirectional Transformers | NAACL-HLT 2019 |
| [3] | Sanh et al. | DistilBERT: A distilled version of BERT | 2019 |
| [4] | Sundararajan et al. | Axiomatic Attribution for Deep Networks | ICML 2017 |
| [5] | Lundberg & Lee | A Unified Approach to Interpreting Model Predictions | NeurIPS 2017 |
| [6] | Jain & Wallace | Attention is Not Explanation | NAACL-HLT 2019 |
| [7] | Abnar & Zuidema | Quantifying Attention Flow in Transformers | ACL 2020 |
| [8] | Pruthi et al. | Estimating Training Data Influence by Tracing Gradient Descent | NeurIPS 2020 |
| [9] | Lipton | The Mythos of Model Interpretability | 2016 |
| [10] | Mitchell et al. | Model Cards for Model Reporting | FAccT 2019 |

---

## ⚠️ Limitations

- **Dataset scope:** SST-2 is single-domain (movie reviews), binary-label; findings may not generalise to longer documents, multi-label tasks, or specialised domains (medical, legal)
- **Model scale:** DistilBERT is compact; explanation behaviours may differ in larger models (GPT-4, LLaMA, etc.) or instruction-tuned LLMs
- **Evaluation subjectivity:** Qualitative faithfulness assessment relies on human interpretation — no formal user study was conducted
- **Language:** English-only; multilingual explainability behaviour not assessed

---

## 🔭 Future Directions

- [ ] Scale analysis to larger models (LLaMA 2/3, Mistral, GPT-family)
- [ ] Extend to diverse tasks: NER, QA, summarisation, code generation
- [ ] Incorporate formal quantitative faithfulness metrics (AOPC, comprehensiveness)
- [ ] Explore multimodal explainability (vision-language models)
- [ ] Develop attention-causal alignment techniques to improve rollout faithfulness
- [ ] Human evaluation study with domain experts for practitioner-grade validation

---


## 👤 Author

<div align="center">

**Venkata Abhinandan Kancharla**  
NRI Institute of Technology  

[![Portfolio](https://img.shields.io/badge/Portfolio-abhikancharla.vercel.app-black?style=for-the-badge&logo=vercel&logoColor=white)](https://abhikancharla.vercel.app)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/venkata-abhinandan-kancharla)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0000--2464--313X-green?style=for-the-badge&logo=orcid&logoColor=white)](https://orcid.org/0009-0000-2464-313X)
[![Email](https://img.shields.io/badge/Email-abhinri6123%40gmail.com-red?style=for-the-badge&logo=gmail&logoColor=white)](mailto:abhinri6123@gmail.com)

</div>

---

## 📜 Citation

If this work is useful for your research or projects, please cite:

```bibtex
@article{kancharla2025appliedxai,
  title     = {Applied Explainability for Large Language Models: A Comparative Study},
  author    = {Kancharla, Venkata Abhinandan},
  year      = {2025},
  doi       = {10.5281/zenodo.19096514},
  url       = {https://doi.org/10.5281/zenodo.19096514},
  note      = {NRI Institute of Technology},
  orcid     = {0009-0000-2464-313X}
}
```

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

<div align="center">

**Intended for:** Academic study · ML Engineering · Applied NLP Research · Portfolio Demonstration

> ⚠️ **Preprint Notice:** This is a research preprint. It has **not undergone formal peer review**. Findings should be interpreted within the scope described in Section 7 (Limitations).

<br/>

*Built with rigor. Evaluated with honesty. Shared for the community.*

<br/>

⭐ **If this helped you — star the repo. It signals quality to recruiters and professors alike.**

</div>
