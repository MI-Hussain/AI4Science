# AI‑for‑Science – Foundation Models Focused Reading List & Survey Outline

> **Last updated:** 30 Jul 2025  
> Curated resources for a forthcoming **Survey** on *scientific foundation models* (SciFMs) – large, generally‑pre‑trained AI systems that can be adapted to diverse scientific problems with minimal additional data or compute.

---

## Table of Contents
1. [Survey Outline](#survey-outline)
2. [Foundation‑Model Breakthrough Timeline](#foundation-model-breakthrough-timeline)
3. [Landscape of Scientific Foundation Models](#landscape-of-scientific-foundation-models)  
   1. [Cross‑disciplinary Language & Multimodal](#31-cross-disciplinary-language--multimodal)  
   2. [Life‑Sciences & Structural Biology](#32-life-sciences--structural-biology)  
   3. [Materials & Chemistry](#33-materials--chemistry)  
   4. [Earth‑System & Climate Science](#34-earth-system--climate-science)
4. [Benchmarks & Datasets](#benchmarks--datasets)
5. [Software & Frameworks](#software--frameworks)
6. [Conferences & Community](#conferences--community)
7. [Staying Current](#staying-current)
8. [Progress Timeline](#progress-timeline)
9. [High‑Level Milestone Plan](#high-level-milestone-plan)
10. [Contributing](#contributing)
11. [License](#license)

---

## Survey Outline

### *Scientific Foundation Models: Catalysing the Next Wave of Discovery*

Foundation models (FMs) pre‑train on petabyte‑scale, heterogeneous data then specialise rapidly via fine‑tuning, prompting or low‑rank adaptation.  
This survey:  

* **Defines SciFMs** – language, vision‑language, graph and physics‑based backbones unified by transferable scientific representations.  
* **Maps impact trajectories** across hypothesis generation, simulation, instrumentation control and robotic experimentation.  
* **Dissects challenges** – data bias, evaluation, safety, compute equity and responsible open‑sourcing.  
* **Forecasts trends** – agentic SciFMs, hybrid neuro‑symbolic FMs, and quantum‑accelerated training.

---

## Foundation‑Model Breakthrough Timeline

| Year | Milestone | Why it matters |
|------|-----------|----------------|
| 2021 | **AlphaFold 2** – near‑atomic protein structures | Sparked the SciFM era :contentReference[oaicite:0]{index=0} |
| 2022 | **Galactica (120 B)** – large language model trained on 48 M scientific papers :contentReference[oaicite:1]{index=1} | Cross‑domain textual reasoning |
| 2023 | **GraphCast** – medium‑range global weather FM beating ECMWF :contentReference[oaicite:2]{index=2} |
| 2023 | **GNoME** – crystal‑stability FM predicts 2.2 M new materials :contentReference[oaicite:3]{index=3} |
| 2024 | **Prithvi WxC** – NASA‑IBM open‑source climate FM :contentReference[oaicite:4]{index=4} |
| 2024 | **AlphaFold 3** – complexes & interactions via diffusion/Pairformer :contentReference[oaicite:5]{index=5} |
| 2024 | **Aurora (1.3 B)** – 3‑D Earth‑system FM; Nature paper :contentReference[oaicite:6]{index=6} |
| 2025 | **WeatherNext** – diffusion‑based operational forecasts via DeepMind :contentReference[oaicite:7]{index=7} |
| 2025 | **IntFold** – controllable biomolecular FM rivaling AlphaFold 3 :contentReference[oaicite:8]{index=8} |

---

## Landscape of Scientific Foundation Models

### 3.1 Cross‑disciplinary Language & Multimodal
| Model | Scale | Modality | Highlight |
|-------|-------|----------|-----------|
| **Galactica** (Meta) :contentReference[oaicite:9]{index=9} | 120 B | Text | Trained on 48 M papers; scientific QA & equation handling |
| **SciGPT** (community forks) | 7–70 B | Text | Open fine‑tunes of GPT‑style LLMs on arXiv & PubMed |
| **PaliGemma / MedGemma** | 3 B | Vision‑Language | Multi‑image/multi‑modal pre‑training; strong transfer to biomedical VQA |

### 3.2 Life‑Sciences & Structural Biology
| Model | Modality | Key capability |
|-------|----------|----------------|
| **AlphaFold 3** :contentReference[oaicite:10]{index=10} | Structure & interactions | Proteins + RNA/DNA + ligands |
| **IntFold** :contentReference[oaicite:11]{index=11} | Controllable all‑atom structure FM | Fine‑grained control for drug design |
| **ProGen 2** | Protein language | Generative protein design at 25 B params |

### 3.3 Materials & Chemistry
| Model | Modality | Key capability |
|-------|----------|----------------|
| **GNoME** :contentReference[oaicite:12]{index=12} | Graph‑based crystal FM | Predicts stability of >2 M novel crystals |
| **Open Catalyst (FM family)** | Graph & energy grids | Catalyst binding energies & kinetics |
| **MatSci‑FM (Open MatSci Toolkit)** :contentReference[oaicite:13]{index=13} | Multi‑task graph FM | Unified materials property prediction & generative design |

### 3.4 Earth‑System & Climate Science
| Model | Range | Distinctive feature |
|-------|-------|---------------------|
| **GraphCast** :contentReference[oaicite:14]{index=14} | 10‑day global | GNN; <60 s on TPU |
| **Aurora** :contentReference[oaicite:15]{index=15} | Atmosphere & ocean | 3‑D Swin/Perceiver; subsumes air‑quality & cyclone tracks |
| **WeatherNext** :contentReference[oaicite:16]{index=16} | Now‑cast → medium‑range | Diffusion ensemble + Cloud API |
| **Prithvi WxC** :contentReference[oaicite:17]{index=17} | Weather‑Climate dual FM | Open‑sourced via Hugging Face; NASA reanalysis pre‑train |

---

## Benchmarks & Datasets

* **Protein:** PDB, AlphaFold DB  
* **Materials:** Materials Project, OQMD, **MatBench‑FM** (100 small tasks for SciFMs)  
* **Catalysis:** OC20 / **OCx24** experimental extension :contentReference[oaicite:18]{index=18}  
* **Climate:** ERA5, **ClimateBench‑FM** subset for Earth FMs  
* **Textual Science QA:** ScienceBench

---

## Software & Frameworks

| Purpose | Tool / Library |
|---------|----------------|
| FM pre‑training | **Megatron‑DeepSpeed**, **JAX / Flax** |
| Scientific graphs | **PyTorch Geometric**, **DGL‑LifeSci** |
| Physics FMs | **Modulus** (PINN→FM pipeline), **PySizing** |
| Deployment | **ONNX‑Runtime‑Inference‑Server**, **vLLM** |

---

## Conferences & Community

* **SciFM25 – Scientific Foundation Models & AI Agents for Science** :contentReference[oaicite:19]{index=19}  
* **NeurIPS & ICML Workshops:** *Foundation Models for Science* (2024‑25)  
* **NASA Open‑Science Summits** – Prithvi WxC tutorials

---

## Staying Current

* **arXiv categories:** `cs.LG`, `physics.comp-ph`, `q-bio.BM`, `stat.ML` with *foundation‑model* keyword  
* **Newsletters:** *DeepMind Science*, *SciFM Digest*  
* **Slack/Discord:** `foundation-models-science`, `ai4sciencecommunity`

---

## 📆 Progress Timeline

```mermaid
timeline
    title SciFM Survey – Weekly Progress
    2025-07-14 : 📝 Initial FM reading list
    2025-07-23 : 🔄 Scope narrowed to foundation models
    2025-07-30 : 📚 Added Aurora, IntFold, WeatherNext

---

## High‑Level Milestone Plan
Phase	Dates (2025)	Deliverable
Gap‑analysis	Jul 30 – Aug 10	Annotated FM taxonomy
Outline Freeze	Aug 18 – Aug 25	Locked FM‑centric outline
Writing Sprint	Aug 26 – Sep 26	Full draft
Internal Review	Sep 29 – Oct 15	Consolidated feedback
Submission	Oct 17	Pre‑print & journal submission

Contributing
Please open an issue or PR with:

Section (e.g., Earth‑System → Aurora)

Resource type (paper / dataset / tool)

One‑line rationale

License
Creative Commons Attribution 4.0 International (CC‑BY‑4.0)

markdown
Copy

---

### What changed?

* **Scope trimmed** to foundation models; non‑FM methodology sections (PINNs, GNNs, etc.) were removed.  
* **Timeline** now highlights eight core SciFM milestones with citations.  
* **Reading list** reorganised by domain‑specific SciFMs.  
* Added **Aurora, WeatherNext, IntFold, Prithvi WxC** and **Galactica** entries.  
* Updated **frameworks** table to emphasise FM training/deployment stacks.

Feel free to adjust naming, add your own models, or reinstate earlier sections if still relevant.
::contentReference[oaicite:20]{index=20}
