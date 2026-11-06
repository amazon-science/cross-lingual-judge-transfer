# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Universal Criteria Set (UCS) Evaluation Module
# Three-stage pipeline: generate concepts -> generate questions per concept -> apply questions to responses

import json
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm


class UCS:
    """Universal Criteria Set - generates verification concepts & questions and evaluates responses."""
    
    def __init__(self, llm, prompts):
        """Initialize UCS.
        
        Args:
            llm: BedrockLLM instance
            prompts: Prompts dict (loaded from JSON)
        """
        self.llm = llm
        self.prompts = prompts
        self._checklist = None

        # validate required prompts
        assert "concept_generation" in self.prompts, "concept_generation required in prompts"
        assert "question_generation" in self.prompts, "question_generation required in prompts"
        assert "question_application" in self.prompts, "question_application required in prompts"
    
    # --- Concept/Question Extraction ---
    
    @staticmethod
    def _extract_concepts(output):
        """Extract concepts from <concept1>...</concept1> tags."""
        pattern = r'<concept\d+>(.*?)</concept\d+>'
        return [m.strip() for m in re.findall(pattern, output, re.IGNORECASE | re.DOTALL) if m.strip()]
    
    @staticmethod
    def _extract_universal_questions(output):
        """Extract questions from <question1>Question: ...</question1> blocks."""
        questions = []
        pattern = r'<question\d+>(.*?)</question\d+>'
        matches = re.findall(pattern, output, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            question_match = re.search(r'Question:\s*(.*?)(?:\n|$)', match, re.IGNORECASE | re.DOTALL)
            if question_match:
                questions.append(question_match.group(1).strip())
        return questions
    
    @staticmethod
    def _extract_questions_scores(output):
        """Extract question scores."""
        scores = []
        question_pattern = r'<question(\d+)>(.*?)</question\d+>'
        matches = re.findall(question_pattern, output, re.IGNORECASE | re.DOTALL)
        
        for i, (_, content) in enumerate(matches):
            score_match = re.search(r'Score:\s*(\d+)', content, re.IGNORECASE)
            score = int(score_match.group(1)) if score_match and score_match.group(1).isdigit() else None            
            scores.append(score)
        return scores
    
    # --- Checklist Cache Management ---
    
    def _load_checklist(self, path):
        """Load checklist from file if exists."""
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, 'r') as f:
                content = f.read()
                if content.strip():
                    checklist = json.loads(content)
                    print(f"Loaded checklist from {path}")
                    return checklist
        except (json.JSONDecodeError, IOError):
            pass
        return None
    
    def _save_checklist(self, path):
        """Save checklist to file."""
        with open(path, 'w') as f:
            json.dump(self._checklist, f, indent=2)
        print(f"Saved checklist to {path}")
    
    # --- Checklist Generation ---
    
    def _generate_checklist(self, cache_path=None):
        """Generate checklist. If cache_path provided, load from cache or save after generation."""
        # Try loading from cache
        cached = self._load_checklist(cache_path)
        if cached:
            self._checklist = cached
            return
        
        print(f"Generating verification concepts using {self.llm}...")
        concept_sys_prompt = self.prompts["concept_generation"]["system_prompt"]
        concept_task_prompt = self.prompts["concept_generation"]["task_prompt"]
        concept_output = self.llm.generate(concept_sys_prompt, concept_task_prompt)
        
        concepts = self._extract_concepts(concept_output)
        print(f"Generated {len(concepts)} concepts: {concepts}")
        
        # Generate questions for each concept
        print("Generating universal questions...")
        gen_sys_prompt = self.prompts["question_generation"]["system_prompt"]
        gen_task_template = self.prompts["question_generation"]["task_prompt"]
        
        self._checklist = {}
        for concept in concepts:
            task_prompt = gen_task_template.format(concept=concept)
            output = self.llm.generate(gen_sys_prompt, task_prompt)
            self._checklist[concept] = self._extract_universal_questions(output)
        
        print(f"Generated {len(concepts)} concepts with questions")
        
        # Save to cache if path provided
        if cache_path:
            self._save_checklist(cache_path)


    def _evaluate_sample(self, row):
        """Evaluate a single sample against the checklist."""
        concept_scores = {}
        
        context = row.get('context', '') 
        answer_segment = row.get('sentence', row.get('answer', ''))
        
        # Format context: list of passages
        if isinstance(context, list):
            passages = [p.get('text', str(p)) if isinstance(p, dict) else str(p) for p in context]
            context_text = "\n" + "\n\n".join(passages) + "\n"
        else:
            context_text = str(context)
        
        app_sys_prompt = self.prompts["question_application"]["system_prompt"]
        app_task_template = self.prompts["question_application"]["task_prompt"]
        
        for concept, questions in self._checklist.items():
            if not questions:
                continue
            
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
            questions_format_example = "\n" + "\n\n".join([
                f"<question{i+1}>\nQuestion: {q}\nScore: [0–10]\nReasoning: [Brief explanation citing the evidence passages]\n</question{i+1}>"
                for i, q in enumerate(questions)
            ]) + "\n"
            task_prompt = app_task_template.format(
                context=context_text,
                answer_segment=answer_segment,
                concept=concept,
                questions=questions_text,
                questions_format_example=questions_format_example
            )

            output = self.llm.generate(app_sys_prompt, task_prompt)
            concept_scores[concept] = self._extract_questions_scores(output)

        return {"concept_scores": concept_scores}
    
    def evaluate(self, df, checklist_cache_path=None, num_workers=4):
        """Evaluate dataset using UCS criteria.
        
        Generates checklist if needed, then evaluates all samples.
        
        Args:
            df: DataFrame with samples to evaluate (must have 'context' and 'sentence'/'answer' columns)
            checklist_cache_path: Optional path to cache checklist (loads if exists, saves after generation)
            num_workers: Number of parallel threads for evaluation
            
        Returns:
            DataFrame with input data and concept_scores column combined
        """
        # Generate checklist if not already done
        if self._checklist is None:
            self._generate_checklist(cache_path=checklist_cache_path)
        
        print(f"Evaluating {len(df)} samples with {num_workers} workers...")
        results = [None] * len(df)
        
        if num_workers == 1:
            for idx in tqdm(range(len(df)), desc="Evaluating"):
                results[idx] = self._evaluate_sample(df.iloc[idx])
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(self._evaluate_sample, df.iloc[idx]): idx
                    for idx in range(len(df))
                }
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        print(f"Error evaluating row {idx}: {e}")
                        results[idx] = {"concept_scores": {}, "error": str(e)}
        
        pred_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
