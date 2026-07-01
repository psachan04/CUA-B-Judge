# BJudge Scaling Agent

An implementation of a test-time scaling pipeline for computer-use agents, based on the architecture detailed in *Scaling Agents for Computer Use* (arXiv:2510.02250). 

The goal of this project is to fix the core issue with digital automation agents: **brittleness over long tasks**. Instead of relying on a single agent trajectory where one bad UI interaction cascades into total failure, this pipeline generates parallel candidate attempts (rollouts), distills what actually happened into a text narrative, and uses an LLM judge to pick the execution that succeeded.

## System Architecture

The pipeline splits the evaluation work into three distinct steps:
1. **Visual Augmentation Engine:** Modifies desktop screenshots dynamically by overlaying coordinate markers (red target dots) and cropping/zooming on the action center to make UI state changes obvious.
2. **Behavior Narrative Generator:** A Vision-Language Model (VLM) inspects the before/after screenshots and actions to extract a concrete, chronological text audit trail of facts.
3. **Comparative Behavior Evaluator:** A multi-choice question (MCQ) prompt stacks the competing text narratives side-by-side. A reasoning model logs its critical thinking inside `<thoughts>` tags and outputs the winning trajectory index inside `<answer>` tags.

## Core Multi-Agent Stack (OpenRouter)

To keep API costs efficient while maintaining high reasoning capabilities, this framework splits responsibilities across specialized open-weight models:
* **Orchestrator / Planning Brain:** `z-ai/glm-5.2` (Handles codebase mapping, multi-file execution analysis, and master trajectory judging).
* **Code Generator / Workers:** `deepseek/deepseek-v4-flash` (An ultra-low-cost, fast model handling repetitive local file edits and syntax corrections).
