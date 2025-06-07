"""
ICE-DSPy Tech Writer: Iterative Consensus Ensemble for synthetic example generation
Combines DSPy's declarative programming with ICE's multi-model consensus approach
"""

import dspy
from pathlib import Path
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import Counter
import logging
import difflib
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import code analysis tools from original implementation
from original.tech_writer_from_scratch import (
    CodeAnalysisTools,
    find_all_matching_files,
    read_file,
    calculate,
    get_gitignore_spec,
    TOOLS
)

@dataclass
class ConsensusResult:
    """Results from consensus detection between models"""
    has_consensus: bool
    confidence: float
    majority_answer: str
    all_answers: List[str]
    round_number: int
    similarity_scores: List[float]

# DSPy Signatures for ICE approach
class AnalyzeCodebaseICE(dspy.Signature):
    """Analyze codebase with awareness of other models' perspectives"""
    brief = dspy.InputField(desc="Technical writing brief")
    codebase_path = dspy.InputField(desc="Path to codebase")
    previous_analyses = dspy.InputField(desc="Previous analyses from other models (empty on first round)")
    analysis = dspy.OutputField(desc="Comprehensive codebase analysis")
    key_insights = dspy.OutputField(desc="Key technical insights found")

class WriteArticleICE(dspy.Signature):
    """Write article with awareness of other models' drafts"""
    brief = dspy.InputField(desc="Original brief")
    code_analysis = dspy.InputField(desc="Consensus code analysis")
    previous_articles = dspy.InputField(desc="Previous article drafts from all models")
    round_feedback = dspy.InputField(desc="Specific areas to improve based on consensus gaps")
    article = dspy.OutputField(desc="Technical article in markdown")
    confidence_score = dspy.OutputField(desc="Self-assessed confidence (0-1)")

class RefineWithConsensus(dspy.Signature):
    """Refine article based on multi-model consensus feedback"""
    current_article = dspy.InputField(desc="Current article version")
    all_suggestions = dspy.InputField(desc="Refinement suggestions from all models")
    consensus_gaps = dspy.InputField(desc="Areas lacking consensus")
    refined_article = dspy.OutputField(desc="Refined article incorporating consensus")

class ICEOrchestrator:
    """Orchestrates Iterative Consensus Ensemble for multiple models"""
    
    def __init__(self, models: List[str], consensus_threshold: float = 0.8, max_rounds: int = 9):
        self.models = models
        self.consensus_threshold = consensus_threshold
        self.max_rounds = max_rounds
        self.lm_instances = {}
        
        # Initialize LM instances for each model
        for model in models:
            self.lm_instances[model] = dspy.LM(model, cache_turn_on=False)
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        # Simple implementation using sequence matching
        # In production, use embeddings for better semantic comparison
        matcher = difflib.SequenceMatcher(None, text1.lower(), text2.lower())
        return matcher.ratio()
    
    def detect_consensus(self, responses: List[Dict[str, Any]], round_num: int) -> ConsensusResult:
        """Detect if models have reached consensus"""
        if not responses:
            return ConsensusResult(False, 0.0, "", [], round_num, [])
        
        # Extract main content from responses
        contents = [r.get('content', '') for r in responses]
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                sim = self.calculate_semantic_similarity(contents[i], contents[j])
                similarities.append(sim)
        
        # Average similarity
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        # Find majority answer (most similar to others)
        similarity_sums = []
        for i, content in enumerate(contents):
            sim_sum = sum(self.calculate_semantic_similarity(content, other) 
                         for j, other in enumerate(contents) if i != j)
            similarity_sums.append(sim_sum)
        
        best_idx = np.argmax(similarity_sums)
        majority_answer = contents[best_idx]
        
        # Check if consensus reached
        has_consensus = avg_similarity >= self.consensus_threshold
        
        return ConsensusResult(
            has_consensus=has_consensus,
            confidence=avg_similarity,
            majority_answer=majority_answer,
            all_answers=contents,
            round_number=round_num,
            similarity_scores=similarities
        )
    
    async def run_model_async(self, model: str, module: dspy.Module, **kwargs) -> Dict[str, Any]:
        """Run a single model asynchronously"""
        try:
            # Configure DSPy to use specific model
            with dspy.settings.context(lm=self.lm_instances[model]):
                result = module(**kwargs)
                return {
                    'model': model,
                    'content': result,
                    'success': True
                }
        except Exception as e:
            logger.error(f"Error with model {model}: {e}")
            return {
                'model': model,
                'error': str(e),
                'success': False
            }
    
    async def run_ensemble_round(self, modules: Dict[str, dspy.Module], **kwargs) -> List[Dict[str, Any]]:
        """Run all models in parallel for one round"""
        tasks = []
        for model, module in modules.items():
            task = self.run_model_async(model, module, **kwargs)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return [r for r in results if r['success']]

class ICETechWriter(dspy.Module):
    """Tech Writer using Iterative Consensus Ensemble"""
    
    def __init__(self, models: List[str] = None, examples_dir: str = "./ice_training_examples"):
        super().__init__()
        
        # Default to diverse models if none specified
        if models is None:
            models = ["openai/gpt-4o-mini", "google/gemini-2.0-flash", "anthropic/claude-3.7-sonnet"]
        
        self.orchestrator = ICEOrchestrator(models)
        self.examples_dir = Path(examples_dir)
        self.examples_dir.mkdir(exist_ok=True)
        
        # Store current codebase for tools
        self.current_codebase = None
        
        # Create analysis modules for each model with ReAct + original tools
        self.analyzers = {
            model: dspy.ReAct(
                AnalyzeCodebaseICE,
                tools=[find_all_matching_files, read_file, calculate],
                max_iters=6
            )
            for model in models
        }
        
        # Create writing modules for each model with tools
        self.writers = {
            model: dspy.ReAct(
                WriteArticleICE,
                tools=[find_all_matching_files, read_file, calculate],
                max_iters=8
            )
            for model in models
        }
        
        # Keep refinement as ChainOfThought since it focuses on consensus
        # rather than exploration (no need for file system tools)
        self.refiners = {
            model: dspy.ChainOfThought(RefineWithConsensus)
            for model in models
        }
    
    def analyze_codebase_with_consensus(self, brief: str, codebase_path: str) -> Dict[str, Any]:
        """Analyze codebase using ICE consensus approach"""
        all_analyses = []
        consensus_reached = False
        
        for round_num in range(1, self.orchestrator.max_rounds + 1):
            logger.info(f"ICE Analysis Round {round_num}")
            
            # Prepare previous analyses for context
            previous_analyses = json.dumps(all_analyses) if all_analyses else "[]"
            
            # Run all models
            round_results = asyncio.run(
                self.orchestrator.run_ensemble_round(
                    self.analyzers,
                    brief=brief,
                    codebase_path=codebase_path,
                    previous_analyses=previous_analyses
                )
            )
            
            # Extract analyses
            analyses = []
            for result in round_results:
                if result['success']:
                    content = result['content']
                    analyses.append({
                        'model': result['model'],
                        'analysis': content.analysis,
                        'key_insights': content.key_insights
                    })
            
            all_analyses.extend(analyses)
            
            # Check consensus
            consensus = self.orchestrator.detect_consensus(
                [{'content': a['analysis']} for a in analyses],
                round_num
            )
            
            logger.info(f"Round {round_num} consensus: {consensus.confidence:.2f}")
            
            if consensus.has_consensus:
                consensus_reached = True
                break
        
        # Return consensus analysis
        return {
            'consensus_analysis': consensus.majority_answer if consensus_reached else analyses[-1]['analysis'],
            'all_analyses': all_analyses,
            'rounds': round_num,
            'consensus_confidence': consensus.confidence if consensus_reached else 0.0
        }
    
    def generate_article_with_ice(self, brief: str, codebase_path: str, code_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate article using ICE approach"""
        all_articles = []
        consensus_reached = False
        best_article = None
        
        for round_num in range(1, self.orchestrator.max_rounds + 1):
            logger.info(f"ICE Writing Round {round_num}")
            
            # Identify areas needing improvement
            round_feedback = ""
            if all_articles:
                # Analyze gaps from previous round
                unique_points = self._identify_unique_points(all_articles)
                round_feedback = f"Focus on these gaps: {json.dumps(unique_points)}"
            
            # Prepare previous articles
            previous_articles = json.dumps([{
                'model': a['model'],
                'article': a['article'][:1000]  # First 1000 chars for context
            } for a in all_articles]) if all_articles else "[]"
            
            # Run all models
            round_results = asyncio.run(
                self.orchestrator.run_ensemble_round(
                    self.writers,
                    brief=brief,
                    code_analysis=json.dumps(code_analysis),
                    previous_articles=previous_articles,
                    round_feedback=round_feedback
                )
            )
            
            # Extract articles
            articles = []
            for result in round_results:
                if result['success']:
                    content = result['content']
                    articles.append({
                        'model': result['model'],
                        'article': content.article,
                        'confidence': float(content.confidence_score)
                    })
            
            all_articles.extend(articles)
            
            # Check consensus
            consensus = self.orchestrator.detect_consensus(
                [{'content': a['article']} for a in articles],
                round_num
            )
            
            logger.info(f"Round {round_num} article consensus: {consensus.confidence:.2f}")
            
            if consensus.has_consensus:
                consensus_reached = True
                best_article = consensus.majority_answer
                break
        
        # If no consensus, use weighted average of confidence scores
        if not consensus_reached and articles:
            best_idx = np.argmax([a['confidence'] for a in articles])
            best_article = articles[best_idx]['article']
        
        return {
            'article': best_article,
            'all_versions': all_articles,
            'rounds': round_num,
            'consensus_reached': consensus_reached,
            'final_confidence': consensus.confidence if consensus_reached else 0.0
        }
    
    def _identify_unique_points(self, articles: List[Dict[str, Any]]) -> List[str]:
        """Identify unique points across articles that need consensus"""
        # Simple implementation - in production use NLP to extract key points
        unique_points = []
        
        # Find sections that vary significantly
        for i, article in enumerate(articles):
            for j, other in enumerate(articles[i+1:], i+1):
                if self.orchestrator.calculate_semantic_similarity(
                    article['article'], other['article']
                ) < 0.7:
                    unique_points.append(f"Reconcile differences between {article['model']} and {other['model']}")
        
        return unique_points[:5]  # Top 5 gaps
    
    def refine_with_ensemble_feedback(self, current_article: str, feedback: str) -> Dict[str, Any]:
        """Refine article using ensemble consensus on improvements"""
        # Get suggestions from all models
        suggestions = []
        
        for model in self.orchestrator.models:
            with dspy.settings.context(lm=self.orchestrator.lm_instances[model]):
                # Each model suggests improvements
                suggestion = f"Model {model} suggests: {feedback}"
                suggestions.append({
                    'model': model,
                    'suggestion': suggestion
                })
        
        # Find consensus gaps
        consensus_gaps = ["Ensure technical accuracy", "Add more examples", "Improve clarity"]
        
        # Run refinement with all models
        refined_results = asyncio.run(
            self.orchestrator.run_ensemble_round(
                self.refiners,
                current_article=current_article,
                all_suggestions=json.dumps(suggestions),
                consensus_gaps=json.dumps(consensus_gaps)
            )
        )
        
        # Extract best refinement
        refinements = []
        for result in refined_results:
            if result['success']:
                refinements.append({
                    'model': result['model'],
                    'article': result['content'].refined_article
                })
        
        # Find consensus on refinements
        if refinements:
            consensus = self.orchestrator.detect_consensus(
                [{'content': r['article']} for r in refinements],
                1
            )
            
            return {
                'refined_article': consensus.majority_answer,
                'consensus_confidence': consensus.confidence,
                'all_refinements': refinements
            }
        
        return {'refined_article': current_article, 'consensus_confidence': 0.0}
    
    def forward(self, brief: str, codebase_path: str) -> Dict[str, Any]:
        """Main entry point - generates article using ICE approach"""
        logger.info(f"Starting ICE Tech Writer for: {codebase_path}")
        
        # Phase 1: Consensus-based code analysis
        code_analysis = self.analyze_codebase_with_consensus(brief, codebase_path)
        
        # Phase 2: Generate article with iterative consensus
        article_result = self.generate_article_with_ice(brief, codebase_path, code_analysis)
        
        # Create session for tracking
        session_id = hashlib.md5(f"{brief}{datetime.now()}".encode()).hexdigest()[:8]
        
        return {
            'session_id': session_id,
            'article': article_result['article'],
            'code_analysis': code_analysis['consensus_analysis'],
            'metadata': {
                'analysis_rounds': code_analysis['rounds'],
                'writing_rounds': article_result['rounds'],
                'consensus_reached': article_result['consensus_reached'],
                'final_confidence': article_result['final_confidence'],
                'models_used': self.orchestrator.models
            },
            'all_versions': article_result['all_versions']
        }
    
    def save_ice_example(self, session_data: Dict[str, Any], quality_score: float) -> str:
        """Save ICE-generated example with metadata"""
        example_dir = self.examples_dir / f"ice_example_{session_data['session_id']}"
        example_dir.mkdir(exist_ok=True)
        
        # Save all components
        with open(example_dir / "session_data.json", 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Save quality metrics
        metrics = {
            'quality_score': quality_score,
            'consensus_confidence': session_data['metadata']['final_confidence'],
            'total_rounds': session_data['metadata']['analysis_rounds'] + session_data['metadata']['writing_rounds'],
            'models': session_data['metadata']['models_used'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(example_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved ICE example to {example_dir}")
        return str(example_dir)

# Synthetic Example Generator using ICE
class ICESyntheticExampleGenerator:
    """Generate high-quality synthetic examples using ICE consensus"""
    
    def __init__(self, writer: ICETechWriter):
        self.writer = writer
        self.quality_threshold = 0.85
    
    def generate_example_batch(self, briefs: List[str], codebase_paths: List[str]) -> List[Dict[str, Any]]:
        """Generate a batch of examples with quality filtering"""
        examples = []
        
        for brief, codebase_path in zip(briefs, codebase_paths):
            try:
                # Generate with ICE
                result = self.writer(brief, codebase_path)
                
                # Quality check based on consensus confidence
                quality_score = result['metadata']['final_confidence']
                
                if quality_score >= self.quality_threshold:
                    examples.append({
                        'brief': brief,
                        'codebase_path': codebase_path,
                        'article': result['article'],
                        'quality_score': quality_score,
                        'metadata': result['metadata']
                    })
                    
                    # Save high-quality examples
                    self.writer.save_ice_example(result, quality_score)
                else:
                    logger.warning(f"Example below quality threshold: {quality_score}")
                    
            except Exception as e:
                logger.error(f"Error generating example: {e}")
        
        return examples
    
    def bootstrap_from_seed_examples(self, seed_examples: List[Dict], num_synthetic: int = 10) -> List[Dict]:
        """Bootstrap new examples from high-quality seeds"""
        synthetic_examples = []
        
        for i in range(num_synthetic):
            # Vary seed examples
            seed = seed_examples[i % len(seed_examples)]
            
            # Create variations
            brief_variation = self._create_brief_variation(seed['brief'])
            
            # Generate new example
            result = self.writer(brief_variation, seed['codebase_path'])
            
            if result['metadata']['final_confidence'] >= self.quality_threshold:
                synthetic_examples.append({
                    'brief': brief_variation,
                    'codebase_path': seed['codebase_path'],
                    'article': result['article'],
                    'quality_score': result['metadata']['final_confidence'],
                    'seed_id': seed.get('id', 'unknown')
                })
        
        return synthetic_examples
    
    def _create_brief_variation(self, original_brief: str) -> str:
        """Create variation of brief for diversity"""
        variations = [
            f"Write a comprehensive guide about {original_brief}",
            f"Create a tutorial explaining {original_brief}",
            f"Develop technical documentation for {original_brief}",
            f"Write an in-depth analysis of {original_brief}"
        ]
        
        import random
        return random.choice(variations)

# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    models = [
        "openai/gpt-4.1-mini",
        "anthropic/claude-3.7", 
        "google/gemini-2.0-flash"
    ]
    
    # Initialize ICE Tech Writer
    ice_writer = ICETechWriter(models=models)
    
    # Test with a sample brief
    test_brief = "Write a technical guide about the caching implementation in this codebase"
    test_codebase = "../LandscapeHub"
    
    # Generate article
    result = ice_writer(test_brief, test_codebase)
    
    print(f"Generated article with {result['metadata']['writing_rounds']} rounds")
    print(f"Consensus confidence: {result['metadata']['final_confidence']:.2f}")
    print(f"Article preview: {result['article'][:500]}...")