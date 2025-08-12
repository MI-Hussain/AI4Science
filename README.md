# AI for Science (AI4S) — Reading List & Survey Outline

**Last updated:** **12 Aug 2025**

> Curated resources and a living outline for an upcoming **Survey** on the landscape of *Artificial Intelligence for Science* — with special focus on **LLM foundation models** (text‑centric) and **Scientist Agents** for literature, planning, coding, and real‑world labs.

---

## Table of Contents

1. [Survey Outline](#survey-outline)
2. [Related Work – History & Foundational Surveys](#related-work--history--foundational-surveys)
3. [LLM & Foundation Models for Science](#llm--foundation-models-for-science)
   - [Generic (Text‑centric) LLMs](#generic-textcentric-llms)
   - [Domain‑Specific Sci‑LLMs](#domain-specific-sci-llms)
   - [Scientist‑Agent Systems](#scientist-agent-systems)
   - [Real Experimental Setups (Autonomous Labs)](#real-experimental-setups-autonomous-labs)
   - [Evaluation & Benchmarks for Science Agents](#evaluation--benchmarks-for-science-agents)
   - [Tooling / Stacks for Agentic Workflows](#tooling--stacks-for-agentic-workflows)
   - [Reproducibility & Governance](#reproducibility--governance)
4. [Domain Breakthroughs](#domain-breakthroughs)
5. [Datasets & Benchmarks](#datasets--benchmarks)
6. [Software & Frameworks](#software--frameworks)
7. [Conferences & Community](#conferences--community)
8. [Staying Current](#staying-current)
9. [📆 Progress Timeline](#-progress-timeline)
10. [📅 High‑Level Milestone Plan](#-high-level-milestone-plan)
11. [Contributing](#contributing)
12. [License](#license)

---

## Survey Outline

### Artificial Intelligence for Scientific Discovery: A Comprehensive Survey

Artificial intelligence (AI) is reshaping scientific practice — from hypothesis generation to autonomous experimentation. This survey synthesizes methods and systems across **deep learning**, **reinforcement learning**, **generative models**, **neuro‑symbolic AI**, **physics‑informed learning**, **geometric/equivariant DL**, **graph neural networks**, **neural operators**, and **large foundation models (LLMs)**. We map progress in **materials**, **chemistry/biomedicine**, **climate & earth**, and **fundamental physics**, and examine challenges (data sparsity, evaluation, reproducibility, safety). We close with trajectories in **autonomous labs**, **agentic systems**, **foundation models**, and **AI‑assisted theory building**.

#### 1. Introduction
- **1.1** What is “AI for Science”? Scope and principles
- **1.2** Historical evolution of AI in scientific research
- **1.3** Motivation and contributions of this survey

**1.1.1 Generic AI for Science**
- Physics‑informed & knowledge‑guided learning
- Geometric & equivariant deep learning
- Neural operators & surrogates
- Foundation models & LLM agents for science
- Automated experiment design & lab robotics
- Benchmarks, evaluation, trust, and interpretability

**1.1.2 Domain‑Specific AI for Science**
- Life sciences & structural biology
- Chemistry & materials discovery
- Earth, climate & environmental science
- Physics, astronomy & cosmology
- Energy, engineering & manufacturing
- Medicine & healthcare imaging/VQA

#### 2. Core AI Methodologies for Science
- **2.1** Deep learning architectures  
- **2.2** Reinforcement learning  
- **2.3** Generative models (design, simulation, data augmentation)  
- **2.4** Symbolic & neuro‑symbolic approaches  
- **2.5** Physics‑Informed Neural Networks (PINNs)  
- **2.6** Graph Neural Networks (GNNs)  

#### 3. Application Domains & Breakthroughs
- **3.1** Materials discovery  
- **3.2** Drug design & biomedicine  
- **3.3** Climate & environmental science  
- **3.4** Fundamental physics & HEP  

#### 4. Challenges & Limitations
- **4.1** Data scarcity/quality & domain shifts  
- **4.2** Interpretability, verification, & mechanistic insight  
- **4.3** Reproducibility & provenance in AI‑driven experiments  
- **4.4** Ethical, legal, biosafety/chem‑safety considerations  

#### 5. Emerging Trends
- **5.1** Interdisciplinary AI & theory‑guided ML  
- **5.2** Autonomous scientific discovery systems (closed‑loop labs)  
- **5.3** Foundation models for science (text, code, structure, multi‑modal)  
- **5.4** Quantum + AI & HPC‑scale FM training  

---

## Related Work – History & Foundational Surveys

### A. Historical Milestones

| Year | Milestone | Why it matters |
| ---- | --------- | -------------- |
| 2016 | **AlphaGo** (Nature) | First splashy combo of deep nets + RL that inspired algorithmic exploration across science. |
| 2021 | **AlphaFold 2** | Protein structure prediction jumps to near‑experimental accuracy; catalyzes modern AI4S. |
| 2023 | **GNoME** predicts **2.2M** stable crystals | Orders‑of‑magnitude acceleration in materials candidate generation. |
| 2024 | **Sakana AI – The AI Scientist** | “Idea→code→run→analyze→draft→auto‑review” end‑to‑end pipeline. |
| 2024 | **AlphaFold 3** | Extends to complexes (proteins, nucleic acids, ligands) with diffusion‑style architecture. |
| 2025 | **AI‑driven autonomous lab (Polybot) @ Argonne/UChicago** | LLM/agent loops integrated with real experimentation. |
| 2025 | **AI Co‑Scientist (Google)** | Gemini 2.0‑powered multi‑agent system for hypothesis→plan→experiment. |
| 2025 | **ASI‑Arch (“AlphaGo moment” for model‑architecture discovery)** | Autonomous end‑to‑end research in model architecture space. |

> ⚠️ *Caveat:* Some claims around “autonomy” vs. “augmentation” are evolving; many systems remain human‑in‑the‑loop.

### B. Foundational Surveys & White‑Papers

- **AI for Science 2025** (Nature feature) – landscape + policy angles.  
- **A New Golden Age of Discovery** (DeepMind, 2024) – opportunity pillars for FM‑driven science.  
- **AI for Science: An Emerging Agenda** (Berens *et al.* 2023) – taxonomy & open questions.  
- **PINNs & Extensions** (Raissi *et al.* 2024); **From PINNs to PIKANs** (2024).  
- **Geometric Deep Learning – Blueprint** (Bronstein *et al.* 2021).  
- **NASA/AGU Foundation Models for Science** (Ramachandran 2023–2024).  

<details>
<summary>🔍 Key Concepts & Principles (concise)</summary>

- **AI vs. AI4S** — AI4S applies ML/DL/statistics/control to *scientific* problems, emphasizing hypothesis‑driven workflows and experimental protocols.  
- **Augmentation → Autonomy** — Today’s systems mainly **augment** human scientists; autonomy emerges in narrow loops.  
- **Hybridization** — Physics‑informed, neuro‑symbolic, and graph‑based methods blend data + priors for extrapolation and trust.  
</details>

---

## LLM & Foundation Models for Science

### Generic (Text‑centric) LLMs

Use frontier or open‑weight LLMs for: literature QA, research planning, code gen for simulations, lab notebook analysis, experimental design checklists, and scientific writing. Track:

**Mini “Model Card” fields to capture (per model):** *context length; tool‑use/calling; function‑calling/sandboxed code; multimodality; license & usage restrictions; safety guardrails; typical strengths/limits on science tasks; eval results on science benchmarks; fine‑tuning/LoRA options; hardware footprint; cost/latency.*

### Domain‑Specific Sci‑LLMs

Bio/med: **BioMedLM**; **BioGPT**  
Science‑general: **SciGLM** (+ SciInstruct)  
Materials/Chemistry: emerging Mat/Chem‑tuned LLMs (e.g., SciDFM, ChemDFM perspectives & workshop papers)

> See links in Reading List below; include *limitations* (hallucinations, citation errors, unit handling) and *mitigations* (RAG w/ provenance, tool‑use, explicit calculators, structure parsers, simulation call‑outs).

### Scientist‑Agent Systems

**Literature‑first agents:**  
- **PaperQA2** (high‑accuracy paper RAG, supports LitQA2)  

**Domain agents (examples):**  
- **ChemCrow** (LLM + dozens of chemistry tools)  
- **SynAsk** (organic synthesis QA platform)

**End‑to‑end research agents:**  
- **AI Co‑Scientist** (Gemini‑2.0 multi‑agent; hypothesis→plan→iterate)  
- **The AI Scientist** (Sakana; idea→code→run→analyze→draft→auto‑review)  
- **ASI‑Arch** (autonomous architecture discovery)

### Real Experimental Setups (Autonomous Labs)

- **ChemOS / ChemOS 2.0** (orchestration for self‑driving labs)  
- **Mobile robotic chemist** (autonomous photocatalysis campaign)  
- **Polybot (Argonne/UChicago)** — AI‑driven autonomous materials lab  
- Reviews & community efforts on SDLs, SiLA2 instrument control, and lab safety SOPs.

### Evaluation & Benchmarks for Science Agents

**Reasoning / problem‑solving:** SciBench (physics/chem/math)  
**Literature research:** LitQA2 (retrieval, grounded summaries, contradiction checks)  
**Multi‑turn agency:** AgentBoard (long‑horizon multi‑step tasks)  
**Safety:** ChemSafetyBench (+ biosecurity eval notes)

**Report (per agent):** task success; groundedness & citation quality; reproducibility (seeds/envs); latency/cost; tool‑use coverage; safety flags.

### Tooling / Stacks for Agentic Workflows

- **Orchestration:** LangGraph; AutoGen; Semantic Kernel; CrewAI  
- **Evidence & scholarly graphs:** OpenAlex; ORKG; structured RAG stacks  
- **Execution & numerics:** unit‑aware calculators; sandboxes; JAX/NumPy/PyTorch hooks  
- **MLOps for agents:** tracing (LangSmith‑style), dataset curation, prompt/version control, eval harnesses

### Reproducibility & Governance

- Dataset/model cards; data licenses; *seeds + exact env capture* (containers/hashes); experiment logs & provenance (papers, code, data, config, tools used)  
- Human‑in‑the‑loop checkpoints for risky actions (chem/bio); red‑teaming & domain‑expert review; safety guidelines (export controls, dual‑use awareness)

---

## Reading List

### 1) Road‑maps & Big‑Picture Overviews

| Year | Reference | Why it matters |
| ---- | --------- | -------------- |
| 2025 | **“AI for Science 2025”** (*Nature* feature) | Landscape snapshot & policy challenges. |
| 2024 | **“A New Golden Age of Discovery”** (DeepMind white paper) | Opportunity pillars for FM‑driven science. |
| 2024 | **PINNs & Extensions** (Raissi *et al.*) | Comprehensive survey of physics‑informed learning. |
| 2024 | **From PINNs to PIKANs** (Toscano *et al.*) | New directions for physics‑guided ML. |
| 2023 | **“AI for Science: An Emerging Agenda”** (Berens *et al.*) | Taxonomy & open questions. |
| 2020 | **“The Automation of Science”** (King *et al.*, *Science*) | Classic manifesto for autonomous labs. |

### 2) Core Methodologies

#### 2.1 Physics‑Informed & Knowledge‑Guided Learning
- Raissi *et al.* 2019 — Seminal PINNs  
- Raissi *et al.* 2024 — PINNs & extensions survey  
- Zhao *et al.* 2024 — PINNs for fluid dynamics  
- Toscano *et al.* 2024 — From PINNs to PIKANs

#### 2.2 GNNs for Molecules & Materials
- **GNoME** — Graph networks for large‑scale materials discovery  
- Defect diffusion GNN (ChemRxiv 2024)  
- Derivative‑based GNN pre‑training (RSC Digital Discovery 2024)

#### 2.3 Geometric & Equivariant Deep Learning
- Bronstein *et al.* 2021 — Geometric DL blueprint  
- EGraFFBench 2023 — Evaluation of E(3)‑equivariant GNNs

#### 2.4 Neural Operators & Surrogate Physics
- **FourCastNet** (2022) — learned surrogates for atmospheric dynamics  
- **WeatherNext** (DeepMind 2025) — SOTA weather forecasting family

#### 2.5 Foundation Models & LLM Agents for Science (selected)
- **Transforming Science with LLMs** (Eger *et al.* 2025) – survey of tools across the research cycle  
- **Foundation models for materials discovery** (Pyzer‑Knapp *et al.* 2025) – perspective on FM classes & future directions  
- **SciGLM & SciInstruct** (2024–2025) – scientific instruction‑tuning for college‑level reasoning  
- **BioMedLM** (Stanford CRFM) & **BioGPT** (Microsoft) – biomedical Sci‑LLMs  
- **AI Co‑Scientist** (Google, 2025) – multi‑agent hypothesis→plan→experiment system  
- **The AI Scientist** (Sakana, 2024) – fully automated discovery pipeline  
- **ASI‑Arch** (2025) – autonomous model‑architecture discovery

### Scientist‑Agent Systems (LLM‑centric)

**A. Generic vs. Domain‑Specific LLMs**  
- **Generic LLMs (text‑centric):** frontier/open‑weight models via API or local hosting for literature QA, planning, coding simulations, and lab data triage.  
- **Domain‑Specific Sci‑LLMs:** biomedical (BioMedLM/BioGPT), science‑general (SciGLM/SciInstruct), materials/chemistry (emerging Chem/Mat FM/LLMs).

> **For each model, track:** context length ▪ tool‑use ▪ multimodality ▪ license ▪ strengths/limits ▪ benchmark results ▪ fine‑tuning paths ▪ cost/latency.

**B. Scientist‑Agent Papers & Code (examples)**  
- **PaperQA2**; **LitQA2** benchmark alignment  
- **ChemCrow**; **SynAsk**  
- **AI Co‑Scientist**; **AI Scientist**; **ASI‑Arch**

**C. Real Experimental Setups (Autonomous Labs)**  
- **ChemOS / ChemOS 2.0** (orchestration)  
- **Mobile robotic chemist** (Nature 2020)  
- **Argonne/UChicago Polybot** (autonomous materials discovery)

**D. Evaluation & Benchmarks for Science Agents**  
- **SciBench** ▪ **LitQA2** ▪ **AgentBoard** ▪ **ChemSafetyBench** (+ biosecurity note)  
- **Report metrics:** groundedness/citation quality ▪ reproducibility (seed/env) ▪ success rate ▪ latency & cost ▪ safety flags

**E. Tooling / Stacks**  
- **Orchestration:** LangGraph ▪ AutoGen ▪ Semantic Kernel ▪ CrewAI  
- **Evidence & data:** OpenAlex ▪ ORKG ▪ RAG stacks ▪ calculators/sandboxes  
- **AgentOps:** tracing/evals ▪ prompt/versioning ▪ dataset curation

**F. Reproducibility & Governance**  
- Dataset/model cards ▪ environment capture ▪ lab‑safety SOPs ▪ red‑teaming checklists

---

## Domain Breakthroughs

| Area | Key Papers / Systems | Highlight |
| --- | --- | --- |
| Structural Biology | **AlphaFold 3** (2024) | Joint complex prediction (proteins, nucleic acids, ligands) with diffusion‑style architecture. |
| Materials Science | **GNoME** (2023) | **2.2M** stable crystals predicted; many now synthesized via A‑Lab/partners. |
| Autonomous Labs | Stach *et al.* (2023) | Closed‑loop frameworks for robotic discovery. |
| Semiconductor Design | Google RL floor‑planning (2021) | RL cuts layout time from weeks to hours. |
| Catalysis | **OCx24** dataset (2024) | Bridges experiment+computation for CO₂RR/HER at industrially relevant conditions. |
| Climate & Weather | FourCastNet; **WeatherNext** | Neural surrogates rival/beat traditional NWP on select regimes. |
| Fundamental Physics | GraphNet tracking at LHC (2021) | Real‑time particle track finding with GNNs. |
| Astronomy & Cosmology | SimBIG (2023); CosmoGAN (2017) | Simulation‑based inference & generative LSS. |
| AI Research | **ASI‑Arch** (2025) | Autonomous architecture discovery (multi‑agent). |

---

## Datasets & Benchmarks

- **Materials & Chemistry:** Materials Project ▪ OQMD ▪ OC20 ▪ **OC22** ▪ **OCx24 (2024)**  
- **Biology:** PDB ▪ UniRef 50 ▪ AlphaFold DB ▪ RNAcentral  
- **Climate:** ERA5 Reanalysis ▪ ClimateBench  
- **Vision‑Language (medical/science):** VQA‑RAD (2018) ▪ PathVQA (2020) ▪ PMC‑VQA (2023) ▪ ScienceQA (2023) ▪ ChartQA (2022) ▪ MIMIC‑CXR (2019) ▪ IU X‑Ray (2015)  
- **Cross‑discipline leaderboards:** ScienceBench ▪ Holobot Challenge

---

## Software & Frameworks

| Tool | Link |
| ---- | ---- |
| **LangGraph** (agent graphs) | https://www.langchain.com/langgraph |
| **AutoGen** (multi‑agent conv.) | https://microsoft.github.io/autogen/ |
| **Semantic Kernel** (agent SDK) | https://github.com/microsoft/semantic-kernel |
| **CrewAI** (lean agent framework) | https://github.com/crewAIInc/crewAI |
| **PaperQA2** (paper‑centric RAG) | https://github.com/Future-House/paper-qa |
| **DeepXDE** | https://github.com/lululxvi/deepxde |
| **SciANN** | https://github.com/sciann/sciann |
| **NVIDIA Modulus** | https://github.com/NVIDIA/modulus |
| **PyTorch Geometric** | https://pytorch-geometric.readthedocs.io |
| **DGL‑LifeSci** | https://lifesci.dgl.ai/ |
| **JAX MD** | https://github.com/google/jax-md |
| **Jraph** | https://github.com/deepmind/jraph |
| **ASE** | https://wiki.fysik.dtu.dk/ase/ |
| **pymatgen** | https://pymatgen.org/ |

---

## Conferences & Community

- **NeurIPS – AI4Science workshops (2021‑2025)**  
- **NeurIPS 2024 – Foundation Models for Science (FM4Science)**  
- **ICML 2024 – Foundation Models for Science (Workshop)**  
- **Nature Machine Intelligence – AI4S collection**  
- **SCI‑FM @ ICLR 2025** – Open science for foundation models

---

## Staying Current

1. **arXiv alerts:** `cs.LG`, `cs.AI`, `cs.CL`, `stat.ML`, `physics.comp-ph`, `q-bio.BM`, `EarthComp`  
2. **Org feeds/newsletters:** DeepMind Science; NVIDIA Earth‑2; ML4Sci Digest; Matterverse  
3. **Communities:** `ai4sciencecommunity` (Slack/Discord); `ml-physics`  
4. **Podcasts:** *DeepMind: The Podcast*; *ScienceML*; *Data Skeptic* (science tracks)

---

## 📆 Progress Timeline

```mermaid
timeline
    title AI4S Survey – Weekly Progress
    2025-07-14 : 📝 Initial reading list
    2025-07-16 : 📝 Kick‑off meeting
    2025-07-18 : 📝 GitHub repo created
    2025-07-21 : 📚 Added DeepMind white‑paper & Nature feature
    2025-07-22 : ✏️ Drafted survey outline & abstract
    2025-07-23 : 🔄 2nd Meeting with Center Director
    2025-07-25 : ✏️ Incorporated agent‑trend; refined reading list
    2025-07-29 : 📝 Continuous updates
    2025-07-31 : 🔄 3rd Meeting ~
    2025-08-06 : 📝 Last update
    2025-08-07 : 🔄 4th Meeting ~

### <summary>📅 High‑Level Milestone Plan</summary>

| Phase                     | Dates (2025)    | Deliverable            |
| ------------------------- | --------------- | ---------------------- |
| Literature & Gap‑analysis | Jul 14 – Aug 14 | Annotated notes        |
| Outline Freeze            | Aug 18 – Aug 25 | Locked survey outline  |
| Writing Sprint            | Aug 26 – Sep 26 | Full draft             |
| Internal Review           | Sep 29 – Oct 15 | Feedback incorporated  |
| Submission                | Oct 17          | Pre‑print & submission |

---

## Contributing

Open an issue or pull request including:

1. **Section** (e.g., Core Methodologies → PINNs)
2. **Resource type** (paper / dataset / tool / tutorial)
3. **One‑line rationale**

## License

Creative Commons Attribution 4.0 International (CC‑BY‑4.0)

---

**Notes on key links I verified while revising this README (for your confidence):**  
- **AI Co‑Scientist (Google)** overview, agents, and examples. :contentReference[oaicite:0]{index=0}  
- **The AI Scientist** (Sakana) system description.   
- **PaperQA2** repo (literature‑first agent) and **LitQA2** benchmark. :contentReference[oaicite:2]{index=2}  
- **AgentBoard** benchmark (multi‑turn agent evaluation).   
- **OpenAlex** & **ORKG** for scholarly graphs/evidence. :contentReference[oaicite:4]{index=4}  
- **SciGLM / SciInstruct**, **BioMedLM**, **BioGPT** (domain Sci‑LLMs). :contentReference[oaicite:5]{index=5}  
- **WeatherNext** (DeepMind 2025) & **GNoME** (2.2M crystals). :contentReference[oaicite:6]{index=6}  
- **OCx24** dataset (Open Catalyst 2024). :contentReference[oaicite:7]{index=7}  
- **AlphaFold 3** paper (Nature 2024). :contentReference[oaicite:8]{index=8}  
- **ASI‑Arch** (2025). :contentReference[oaicite:9]{index=9}  
- **Argonne/UChicago Polybot** autonomous lab. :contentReference[oaicite:10]{index=10}  
- **ChemOS / ChemOS 2.0** and the **mobile robotic chemist**. :contentReference[oaicite:11]{index=11}
