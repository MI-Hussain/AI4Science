AI for Science — Reading List & Survey Outline

Curated, living resource – last updated: 17 July 2025

This repository collects foundational papers, reviews, datasets, software tools, and community links that underpin the rapidly‑growing field of AI for Science (AI4S). It doubles as the working material for an upcoming survey to be submitted to ACM Computing Surveys.

📑 Table of Contents

Proposed Survey Outline

1  Road‑maps & Big‑Picture Overviews

2  Core Methodologies

3  Domain Breakthroughs

4  Datasets & Benchmarks

5  Software & Frameworks

6  Conferences, Workshops & Community

7  How to Stay Current

Proposed Survey Outline

Section 1 – Generic AI for Science

(core cross‑domain methodologies & scientific‑agent paradigms)

Physics‑informed & knowledge‑guided learning

Geometric & equivariant deep learning

Neural operators & surrogate modeling

Foundation models & LLM agents for science

Automated experiment design & lab robotics

Cross‑domain benchmarks, evaluation & interpretability

Section 2 – Domain‑Specific AI for Science

(application‑focused advances organized by scientific discipline)

Life sciences & structural biology

Chemistry & materials discovery

Earth, climate & environmental science

Physics, astronomy & cosmology

Energy, engineering & manufacturing

Medicine & healthcare imaging & VQA

Reading List

1  Road‑maps & Big‑Picture Overviews

Year

Reference

Why it matters

2025

“AI for Science 2025” (Nature feature)

Short industry‑oriented snapshot of the paradigm shift and emerging challenges.

2024

“A New Golden Age of Discovery – Seizing the AI4S Opportunity” (DeepMind white‑paper)

Policy‑level report framing five opportunity pillars (knowledge, data, simulation, experimentation, solutions).

2023

“AI for Science: An Emerging Agenda” (Berens et al., arXiv 2303.04217)

Academic roadmap; taxonomy of AI‑science intersections & open research questions.

2024

Physics‑Informed Neural Networks & Extensions (Raissi et al.)

Compact survey of PINNs evolution, theory and software stacks.

2024

“From PINNs to PIKANs” (Toscano et al.)

Latest advances and future directions in physics‑informed ML.

2  Core Methodologies

2.1  Physics‑Informed & Knowledge‑Guided Learning

Raissi et al., 2019 – Seminal PINNs paper (JCP 378) – forward & inverse PDE solving.

Zhao et al., 2024 – PINNs for complex fluid dynamics (Phys. Fluids).

2.2  Graph Neural Networks (GNNs) for Molecules & Materials

Defect diffusion GNN for high‑T energy materials (2024 ChemRxiv preprint).

Derivative‑based pre‑training of GNNs for force‑field learning (RSC Digit. Discov.).

2.3  Geometric & Equivariant Deep Learning

Bronstein et al., 2021 – Geometric Deep Learning: Grids, Groups, Graphs… (review).

Schneideman‑Chenn et al., 2023 – E(3)‑equivariant networks for atomistic simulation.

2.4  Neural Operators & Surrogate Physics

FourCastNet – global sub‑second weather surrogate (Pathak et al., 2022).

WeatherNext – DeepMind’s 10‑day skill leader (2025).

2.5  Foundation Models for Science

NeurIPS 2024 workshop “Foundation Models for Science” – slides & proceedings.

Ramachandran et al., 2023 – AGU talk on cross‑domain foundation models.

3  Domain Breakthroughs

Area

Key Papers / Systems

Highlight

Structural Biology

AlphaFold 2 (2021); AlphaFold 3 release (Nature 2024)

Near‑ab initio prediction of protein & RNA complexes; open inference pipeline.

Materials Science

GNoME – 2.2 M crystal predictions (Nature 2023)

Orders‑of‑magnitude expansion of stable materials space.

Chemistry / Catalysis

Open Catalyst 2024 dataset & GemNet‑OC; Diffusion models for molecule generation

Open‑source benchmark and scalable catalyst discovery.

Climate & Weather

FourCastNet (NVIDIA) and WeatherNext (DeepMind)

Neural operators beating traditional NWP baselines.

High‑Energy Physics

GraphNet tracking at LHC; PINNs for plasma control

Real‑time reconstruction & control in extreme regimes.

Astronomy & Cosmology

SimBIG simulation‑based inference; CosmoGAN

Likelihood‑free inference for large‑scale structure.

4  Datasets & Benchmarks

Materials & Chemistry: Materials Project, OQMD, Open Catalyst 2024, QM9, OC20.

Biology: PDB (weekly), UniRef 50, AlphaFold DB, PDB70, RNAcentral.

Climate: ERA5 reanalysis, ClimateBench.

Vision‑language in Science: VQA‑RAD (2018) – radiology VQA (relevant to PMC‑VQA work).

Cross‑discipline leaderboards: ScienceBench, Holobot.

5  Software & Frameworks

Tool

Notes

DeepXDE / SciANN / Modulus

High‑level APIs for PINNs & PDE solving

PyTorch Geometric / DGL‑LifeSci

Rapid prototyping of GNNs on molecular graphs

JAX MD / Jraph

Differentiable molecular dynamics with JAX

ASE & Pymatgen

Programmatic materials workflows; integrates with GNoME outputs

6  Conferences, Workshops & Community

NeurIPS “AI for Science” series (2021‑2025) & AI4Mat‑2024 workshop.

ICML “Foundation Models for Science” (2024).

Open Conference of AI Agents for Science 2025 – fully AI‑authored venue.

Nature Machine Intelligence “AI for Science” collection (ongoing special issue).

7  How to Stay Current

arXiv alerts: subscribe to cs.LG, physics.comp-ph, q-bio.BM, stat.ML, and EarthComp.

Newsletters & Blogs: DeepMind Science, NVIDIA Earth‑2, ML4Sci digest, Matterverse.

Slack / Discord: ai4sciencecommunity, ml-physics.

Podcasts & YouTube: DeepMind: The Podcast, ScienceML, Data Skeptic (science tracks).

Contributing

Feel free to open issues or PRs to add papers, datasets, or tools. Please follow the existing section structure and include a brief one‑line rationale.

License

Creative Commons Attribution 4.0 International (CC‑BY‑4.0).

