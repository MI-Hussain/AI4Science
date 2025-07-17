# AI for Science (AI4S) – Reading List & Survey Outline

> **Last updated:** 17 Jul 2025
> Curated resources and draft outline for an upcoming **ACM Computing Surveys** paper on the landscape of *Artificial Intelligence for Science*.

## Table of Contents

1. [Survey Outline](#survey-outline)
2. [Reading List](#reading-list)

   1. [Road‑maps & Overviews](#1-road-maps--big-picture-overviews)
   2. [Core Methodologies](#2-core-methodologies)
   3. [Domain Breakthroughs](#3-domain-breakthroughs)
   4. [Datasets & Benchmarks](#4-datasets--benchmarks)
   5. [Software & Frameworks](#5-software--frameworks)
   6. [Conferences & Community](#6-conferences-workshops--community)
   7. [Staying Current](#7-how-to-stay-current)
3. [Contributing](#contributing)
4. [License](#license)

---

## Survey Outline

### 1 Generic AI for Science

Key cross‑domain methods and scientific‑agent paradigms:

* Physics‑informed & knowledge‑guided learning
* Geometric & equivariant deep learning
* Neural operators & surrogate modeling
* Foundation models & LLM agents for science
* Automated experiment design & lab robotics
* Benchmarks, evaluation & interpretability

### 2 Domain‑Specific AI for Science

Application‑focused advances by discipline:

* Life sciences & structural biology
* Chemistry & materials discovery
* Earth, climate & environmental science
* Physics, astronomy & cosmology
* Energy, engineering & manufacturing
* Medicine & healthcare imaging & VQA

---

## Reading List

### 1 Road‑maps & Big‑Picture Overviews

| Year | Reference                                                         | Why it matters                           |
| ---- | ----------------------------------------------------------------- | ---------------------------------------- |
| 2025 | **“AI for Science 2025”** (Nature feature)                        | Snapshot of paradigm shift & challenges  |
| 2024 | **“A New Golden Age of Discovery”** (DeepMind white‑paper)        | Five opportunity pillars                 |
| 2023 | **“AI for Science: An Emerging Agenda”** (Berens et al.)          | Taxonomy & open questions                |
| 2024 | **Physics‑Informed Neural Networks & Extensions** (Raissi et al.) | Survey of PINNs evolution                |
| 2024 | **“From PINNs to PIKANs”** (Toscano et al.)                       | Future directions in physics‑informed ML |

### 2 Core Methodologies

#### 2.1 Physics‑Informed & Knowledge‑Guided Learning

* Raissi et al., 2019 – seminal PINNs paper (JCP 378)
* Zhao et al., 2024 – PINNs for complex fluid dynamics (Phys. Fluids)

#### 2.2 Graph Neural Networks (GNNs) for Molecules & Materials

* Defect diffusion GNN for high‑T energy materials (2024 ChemRxiv)
* Derivative‑based pre‑training of GNNs for force‑field learning (RSC Digit. Discov.)

#### 2.3 Geometric & Equivariant Deep Learning

* Bronstein et al., 2021 – Geometric Deep Learning review
* Schneideman‑Chenn et al., 2023 – E(3)‑equivariant networks

#### 2.4 Neural Operators & Surrogate Physics

* FourCastNet – global weather surrogate (Pathak et al., 2022)
* WeatherNext – DeepMind 10‑day forecast model (2025)

#### 2.5 Foundation Models for Science

* NeurIPS 2024 workshop: Foundation Models for Science
* Ramachandran et al., 2023 – Cross‑domain foundation models (AGU talk)

### 3 Domain Breakthroughs

| Area                  | Key Papers/Systems                                | Highlight                        |
| --------------------- | ------------------------------------------------- | -------------------------------- |
| Structural Biology    | AlphaFold 2 (2021); **AlphaFold 3** (Nature 2024) | Protein/RNA complex prediction   |
| Materials Science     | **GNoME** (Nature 2023)                           | 2.2 M crystal predictions        |
| Chemistry/Catalysis   | Open Catalyst 2024 & GemNet‑OC                    | Scalable catalyst discovery      |
| Climate & Weather     | FourCastNet; WeatherNext                          | Neural operators beat NWP        |
| High‑Energy Physics   | GraphNet tracking; plasma PINNs                   | Real‑time extreme‑regime control |
| Astronomy & Cosmology | SimBIG; CosmoGAN                                  | Likelihood‑free inference        |

### 4 Datasets & Benchmarks

* Materials & Chemistry: Materials Project, OQMD, OC24, QM9, OC20
* Biology: PDB, UniRef 50, AlphaFold DB, RNAcentral
* Climate: ERA5, ClimateBench
* Vision‑language: **VQA‑RAD** (radiology)
* Cross‑discipline: ScienceBench, Holobot

### 5 Software & Frameworks

| Tool                            | Notes                    |
| ------------------------------- | ------------------------ |
| DeepXDE / SciANN / Modulus      | High‑level PINNs APIs    |
| PyTorch Geometric / DGL‑LifeSci | GNN prototyping          |
| JAX MD / Jraph                  | Differentiable MD in JAX |
| ASE & Pymatgen                  | Materials workflows      |

### 6 Conferences, Workshops & Community

* **NeurIPS AI4Science** (2021‑2025) & **AI4Mat‑2024**
* **ICML Foundation Models for Science** (2024)
* **Open Conference of AI Agents for Science** (2025)
* **Nature Machine Intelligence – AI4S collection**

### 7 How to Stay Current

1. Subscribe to arXiv: `cs.LG`, `physics.comp-ph`, `q-bio.BM`, `stat.ML`, `EarthComp`
2. Newsletters: *DeepMind Science*, *NVIDIA Earth‑2*, *ML4Sci Digest*, *Matterverse*
3. Slack/Discord: `ai4sciencecommunity`, `ml-physics`
4. Podcasts: *DeepMind: The Podcast*, *ScienceML*, *Data Skeptic* (science tracks)

---

## Contributing

Contributions are welcome! Open an issue or PR with:

1. **Section** (e.g., Core Methodologies → PINNs)
2. **Resource type** (paper / dataset / tool)
3. **One‑line rationale**

Please follow the existing structure.

## License

Distributed under **CC‑BY‑4.0**.
