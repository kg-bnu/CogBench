<img width="1024" height="1024" alt="Gemini_Generated_Image_1p5o1e1p5o1e1p5o" src="https://github.com/user-attachments/assets/82f69084-3780-4f6f-b850-31016638e300" />
<h1 align="center">CogBench: Benchmarking Cognitive Alignment of Large Language Models in Educational Question Answering</h1>
<p align="center">
  <!-- <!-- Stars / Forks -->
  <a href="https://github.com/kg-bnu/CogBench">
    <img src="https://img.shields.io/github/stars/kg-bnu/CogBench?style=flat-square&logo=github&label=Stars" />
  </a>
  <a href="https://github.com/kg-bnu/CogBench/fork">
    <img src="https://img.shields.io/github/forks/kg-bnu/CogBench?style=flat-square&logo=github&label=Forks" />
  </a> 
  <!-- Dataset (huggingface -->
  <a href="https://huggingface.co/datasets/realEthanTLu/CogBench">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-orange?style=flat-square&logo=huggingface" />
  </a>
    <!-- Web -->
  <a href="https://cogbench.lutong.space/">
    <img src="https://img.shields.io/badge/Web-Project_Page-blue?style=flat-square&logo=googlechrome" />
  </a>
</p>

<br>

<p align="center">
  <strong>CogBench</strong> is a benchmark  to assess the cognitive alignment capabilities of Large Language models in educational question answering
</p>


<img width="1271" height="523" alt="framework" src="https://github.com/user-attachments/assets/5c71cfab-e078-4ad9-9bbc-b89b46ff4225" />


## 🔥 Highlights

- Benchmark: 2,100 K–12 mathematics questions, each with multiple valid, cognition-differentiated solutions
- Average 2.16 solutions per question; 3.2 curriculum knowledge components per question
- Grade coverage: Primary 40%, Middle 35%, High 25%
- 3 cognition-aware QA tasks; 3 complementary metrics (CA, KC, KD)
- Curriculum-Aware Knowledge Graph (CAKG) aligned to grade levels and solution strategies
- Evaluated 11 LLMs (open-source and proprietary) via APIs (Sept–Dec 2025)
- Key findings:
  - Large gap between standard accuracy (up to 0.942) and cognitive alignment under unconstrained QA (best CA 0.534, KC 0.604)
  - Grade-constrained prompting improves alignment (best CA 0.560, KC 0.753; KD up to 0.790)
  - Knowledge-constrained prompting often reduces alignment due to activation of higher-level parametric patterns
  - Fine-tuning (SFT + DPO) improves CA (0.47→0.63) and KC (0.54→0.68) with slight drops in ACC (0.88→0.83) and KD (0.72→0.61)
  - Automatic metrics correlate well with expert human judgments on consistency and diversity

## 🚀 Overview

CogBench is built using a Multi-solution–Alignment–Evaluation pipeline:

1) Multi-solution Generation
   - Multi-turn sampling with controlled decoding (temperature, top-k, nucleus) produces diverse, correct solution traces per question.
   - Only answers with correct final results are retained (validated against gold answers).

2) Probability Attenuation for Diversity
   - Identify anchor tokens (key concepts) from existing solutions.
   - Use semantic inference (via embedding space and Moore–Penrose pseudoinverse) to attenuate probabilities of previously used/semantically similar tokens during decoding.
   - Encourage discovery of novel, valid reasoning paths relying on different knowledge/strategies.

3) Curriculum-Aware Knowledge Graph (CAKG)
   - Extract K–12 cognition-aware math knowledge from official standards.
   - Organize as cumulative, grade-tagged subgraphs that encode procedural strategies and reasoning patterns.
   - Emphasize fine-grained, solution strategy–level knowledge beyond topic hierarchies.

4) Solution–Cognition Alignment
   - Encode solutions and CAKG triples (e.g., Qwen3 Embedding) and retrieve top-k relevant knowledge by cosine similarity.
   - Induce candidate grade levels from retrieved triples.
   - Human-in-the-loop expert validation refines solution–knowledge–grade mappings.

5) Cognition-Aware Evaluation
   - Tasks:
     - Unconstrained QA: solve without cognitive cues (baseline behavior).
     - Grade-Constrained QA: generate solutions tailored to a specified grade.
     - Knowledge-Constrained QA: solve using only provided curriculum knowledge.
   - Metrics:
     - Cognitive Accuracy (CA): correct answers that also meet the target cognitive level.
     - Knowledge Consistency (KC): adherence to grade-appropriate curriculum knowledge.
     - Knowledge Divergence (KD): differentiation of knowledge usage across grades (pairwise Jaccard distance).

## 📊 Dataset & Annotations

- Sources: 1.2K Olympiad problems (public website) + 0.9K CMMath problems
- Coverage: Primary (Grades 1–6), Middle (7–9), High (10–12)
- Per-question: at least two solutions at different cognitive levels
- Generation base model for multi-solution sampling: Qwen3-30B-A3B
- Expert alignment: education experts verify solution–knowledge–grade mapping
- Reliability: high-quality, cognition-aware labels after expert review

## 📦 Usage
Evaluation
The evaluation program is in the evaluation folder, and the metrics it uses are in the metric folder.

### Metric Explanations:
- ACC: Under each of the three prompting modes, the accuracy when the model directly answers the subject question.
- CAR: Under each of the three prompting modes, the accuracy of producing a correct answer using only knowledge whose grade tag is less than or equal to the current grade.
- KAS: (Intersection of triples used by the standard answer and the LLM) divided by (triples used by the LLM).
- PAD: For the same question, the difference between the knowledge used when the model answers from different grades.
- AS: AS = CAR + KAS + PAD; under each of the three prompting modes, this is the overall score evaluating the model’s adaptation to human cognition (grade).


### Three prompting modes:
- Direct: `response1_title_only`
- Grade-aware: `response2_title_grade`
- Knowledge-aware: `response3_title_knowledge`

### Run the evaluation scripts:
```shell
python -m evaluation.response --model_name gpt-5-nano-2025-08-07
python -m evaluation.evaluate_response --model_name gpt-5-nano-2025-08-07
python -m evaluation.find_knowledge_used --model_name gpt-5-nano-2025-08-07
python -m evaluation.calculate_metrics --model_name gpt-5-nano-2025-08-07
```



## 📬 Contact

Project Lead: ethanlu@mail.bnu.edu.cn

Dataset: https://huggingface.co/datasets/realEthanTLu/CogBench

Web Page: https://cogbench.lutong.space/


## 📄 Citation
If you use CogBench or our construction framework, please cite:
```mathematica
@article{CogBench2026,
  title={CogBench: BenchmarkingCognitiveAlignmentofLargeLanguageModels
inEducationalQuestionAnswering},
  author={Tong Lu, Zhichun Wang, Yuanhao Sun, Yaoyu Zhou, Mingrui Li,Yiming Guan, Zhiyong Bai},
  year={2026},
  journal={Findings of ACL}
}
```
