# Codex Sinaiticus: AI Exegesis & Metadata Pipeline

## Overview

This repository contains computational archiving scripts designed to process, translate, and structure ancient Greek texts from the Codex Sinaiticus. Utilizing Large Language Models (LLMs) and Prompt-Driven Systems Architecture, this tool transforms unstructured, complex historical manuscripts into highly accurate exegesis based on the oldest known "word of God" through Christ (Christian perspective).

This project was developed to address a bottleneck in digital humanities: the inability of standard AI models to accurately contextualize ancient texts without severe hallucinations.

## Methodology & Architecture

Standard language models frequently fail when processing nuanced historical and theological texts. Instead of something valuable or thought provoking, they often output anachronistic or logically flawed data. This pipeline was built using rigorous adversarial testing and iterative prompt engineering to force the model into factual compliance with a large degree of accuracy. While current LLM architecture does not allow 100% mathematical precision in regard to full compliance, I believe I have successfully mitigated this to a virtually (or sometimes completely) undetectable degree. 

* **Hallucination Mitigation:** Engineered multi-layered prompt sequences to verify translations against converted historical context, effectively eliminating the model's tendency to "guess" missing linguistic links.

* **Structured Output Generation:** Forces the LLM to abandon standard conversational output in favor of strict, database-ready formatting (e.g., JSON/Markdown arrays).

* **API & Local Compute Integration:** Currently architected utilizing the Google AI API for high-level reasoning, with the system designed to scale into a fully localized, offline pipeline running on a fully local hardware environment. This ensures zero data leakage for sensitive or proprietary archival collections.

## Data Processing Example: Before & After

To demonstrate the pipeline's capability, here is an example of the unstructured input versus the generated, database-ready output.

### 1. Raw Input (Unstructured Greek / Transliteration)

[Book: Matthew] [Chapter: 6] [Verse: 10]
ελθατω η βαϲιλια ϲου γενηθητω το θε ϲου ωϲ εν ου και επι γηϲ

### 2. Pipeline Output (Structured Exegesis)

**Verse 10** **Greek Text** ελθατω η βαϲιλια ϲου γενηθητω το θε ϲου ωϲ εν ου και επι γηϲ  

**English Translation** Let come the kingdom of you; let be done the wi[ll] of you, as in heav[en], also on earth.  

**Episcopal Exegesis** The "kingdom" and the "will" are parallel forces descending from "heaven" to "earth." The petition seeks a perfect symmetry, where the earthly reality conforms entirely to the heavenly pattern. We ask for the "will" to be executed here as flawlessly as it is there.

## Current Status & Future Scope

* Phase 1 (Completed): Core extraction and exegesis pipeline for New Testament Greek texts established and verified for accuracy, as well as fully completing the book of Matthew.

* Phase 2 (In Development): Finishing the entire NT, and subsequently adapting the underlying architecture to build a searchable, locally-hosted database for Anglo-Saxon Old English texts and historical sources.

* Use Case: This architecture can be rapidly deployed to digitize, tag, and structure neglected physical archives, transforming uncatalogued collections into accessible digital repositories.

---
Developed by Ethan Jones | Texas State University  
*Assisted by LLM use*
