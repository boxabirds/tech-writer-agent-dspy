"""
Evaluation metrics and quality assessment for ICE-generated examples
"""

import dspy
from typing import Dict, List, Any, Tuple
import numpy as np
from dataclasses import dataclass
import json
from pathlib import Path
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for ICE-generated articles"""
    consensus_confidence: float
    semantic_coherence: float
    technical_accuracy: float
    code_coverage: float
    example_diversity: float
    readability_score: float
    completeness_score: float
    overall_quality: float

class TechnicalAccuracyValidator(dspy.Signature):
    """Validate technical accuracy of generated content"""
    article = dspy.InputField(desc="Generated technical article")
    codebase_analysis = dspy.InputField(desc="Ground truth code analysis")
    accuracy_score = dspy.OutputField(desc="Technical accuracy score (0-1)")
    errors_found = dspy.OutputField(desc="List of technical errors or inaccuracies")

class CompletenessChecker(dspy.Signature):
    """Check if article addresses all aspects of the brief"""
    brief = dspy.InputField(desc="Original brief")
    article = dspy.InputField(desc="Generated article")
    completeness_score = dspy.OutputField(desc="Completeness score (0-1)")
    missing_aspects = dspy.OutputField(desc="Aspects from brief not covered")

class ICEQualityEvaluator:
    """Comprehensive evaluator for ICE-generated technical articles"""
    
    def __init__(self):
        self.accuracy_validator = dspy.ChainOfThought(TechnicalAccuracyValidator)
        self.completeness_checker = dspy.ChainOfThought(CompletenessChecker)
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    
    def evaluate_example(self, example: Dict[str, Any]) -> QualityMetrics:
        """Evaluate a single ICE-generated example"""
        article = example.get('article', '')
        metadata = example.get('metadata', {})
        brief = example.get('brief', '')
        code_analysis = example.get('code_analysis', '')
        
        # 1. Consensus confidence (from ICE)
        consensus_confidence = metadata.get('final_confidence', 0.0)
        
        # 2. Semantic coherence
        semantic_coherence = self._calculate_semantic_coherence(article)
        
        # 3. Technical accuracy
        technical_accuracy = self._validate_technical_accuracy(article, code_analysis)
        
        # 4. Code coverage
        code_coverage = self._calculate_code_coverage(article, code_analysis)
        
        # 5. Example diversity
        example_diversity = self._calculate_example_diversity(article)
        
        # 6. Readability
        readability_score = self._calculate_readability(article)
        
        # 7. Completeness
        completeness_score = self._check_completeness(brief, article)
        
        # 8. Overall quality (weighted average)
        overall_quality = self._calculate_overall_quality({
            'consensus': consensus_confidence,
            'coherence': semantic_coherence,
            'accuracy': technical_accuracy,
            'coverage': code_coverage,
            'diversity': example_diversity,
            'readability': readability_score,
            'completeness': completeness_score
        })
        
        return QualityMetrics(
            consensus_confidence=consensus_confidence,
            semantic_coherence=semantic_coherence,
            technical_accuracy=technical_accuracy,
            code_coverage=code_coverage,
            example_diversity=example_diversity,
            readability_score=readability_score,
            completeness_score=completeness_score,
            overall_quality=overall_quality
        )
    
    def _calculate_semantic_coherence(self, article: str) -> float:
        """Calculate semantic coherence of the article"""
        # Split into paragraphs
        paragraphs = [p.strip() for p in article.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            return 1.0
        
        try:
            # Vectorize paragraphs
            vectors = self.tfidf.fit_transform(paragraphs)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(paragraphs) - 1):
                sim = cosine_similarity(vectors[i], vectors[i+1])[0][0]
                similarities.append(sim)
            
            # Average similarity as coherence score
            return np.mean(similarities) if similarities else 0.0
        except:
            return 0.5  # Default if vectorization fails
    
    def _validate_technical_accuracy(self, article: str, code_analysis: str) -> float:
        """Validate technical accuracy using LLM"""
        try:
            result = self.accuracy_validator(
                article=article,
                codebase_analysis=code_analysis
            )
            return float(result.accuracy_score)
        except:
            # Fallback: check for code-related keywords
            code_terms = re.findall(r'\b(function|class|method|variable|import|module)\b', article.lower())
            return min(len(code_terms) / 50.0, 1.0)  # Expect at least 50 technical terms
    
    def _calculate_code_coverage(self, article: str, code_analysis: str) -> float:
        """Calculate how well the article covers the codebase"""
        # Extract key components from code analysis
        components = re.findall(r'(class|function|module)\s+(\w+)', code_analysis.lower())
        
        if not components:
            return 0.5
        
        # Check how many are mentioned in article
        mentioned = 0
        article_lower = article.lower()
        
        for _, name in components:
            if name in article_lower:
                mentioned += 1
        
        return mentioned / len(components) if components else 0.0
    
    def _calculate_example_diversity(self, article: str) -> float:
        """Calculate diversity of code examples"""
        # Extract code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', article)
        
        if not code_blocks:
            return 0.0
        
        # Analyze diversity
        languages = set()
        patterns = set()
        
        for block in code_blocks:
            # Detect language
            lang_match = re.match(r'```(\w+)', block)
            if lang_match:
                languages.add(lang_match.group(1))
            
            # Extract patterns (simplified)
            if 'class' in block:
                patterns.add('class_definition')
            if 'def' in block or 'function' in block:
                patterns.add('function_definition')
            if 'import' in block or 'require' in block:
                patterns.add('imports')
            if 'for' in block or 'while' in block:
                patterns.add('loops')
        
        # Score based on variety
        diversity_score = (len(languages) * 0.3 + len(patterns) * 0.7) / 5.0
        return min(diversity_score, 1.0)
    
    def _calculate_readability(self, article: str) -> float:
        """Calculate readability score"""
        # Simple readability metrics
        sentences = article.split('.')
        words = article.split()
        
        if not sentences or not words:
            return 0.0
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Complex word ratio (words > 6 chars)
        complex_words = [w for w in words if len(w) > 6]
        complex_ratio = len(complex_words) / len(words)
        
        # Scoring (prefer medium sentence length, moderate complexity)
        if 15 <= avg_sentence_length <= 25:
            length_score = 1.0
        else:
            length_score = 0.5
        
        if 0.2 <= complex_ratio <= 0.4:
            complexity_score = 1.0
        else:
            complexity_score = 0.5
        
        return (length_score + complexity_score) / 2
    
    def _check_completeness(self, brief: str, article: str) -> float:
        """Check if article addresses all aspects of brief"""
        try:
            result = self.completeness_checker(
                brief=brief,
                article=article
            )
            return float(result.completeness_score)
        except:
            # Fallback: keyword matching
            brief_words = set(brief.lower().split())
            article_words = set(article.lower().split())
            
            overlap = len(brief_words & article_words)
            return min(overlap / len(brief_words), 1.0) if brief_words else 0.5
    
    def _calculate_overall_quality(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            'consensus': 0.20,
            'coherence': 0.15,
            'accuracy': 0.25,
            'coverage': 0.15,
            'diversity': 0.10,
            'readability': 0.10,
            'completeness': 0.05
        }
        
        weighted_sum = sum(scores[key] * weights[key] for key in weights)
        return weighted_sum

class ICEExampleRanker:
    """Rank and filter ICE-generated examples by quality"""
    
    def __init__(self, evaluator: ICEQualityEvaluator):
        self.evaluator = evaluator
    
    def rank_examples(self, examples: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], QualityMetrics]]:
        """Rank examples by overall quality"""
        evaluated = []
        
        for example in examples:
            metrics = self.evaluator.evaluate_example(example)
            evaluated.append((example, metrics))
        
        # Sort by overall quality
        evaluated.sort(key=lambda x: x[1].overall_quality, reverse=True)
        
        return evaluated
    
    def filter_high_quality(self, examples: List[Dict[str, Any]], threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Filter examples above quality threshold"""
        ranked = self.rank_examples(examples)
        
        filtered = []
        for example, metrics in ranked:
            if metrics.overall_quality >= threshold:
                # Add metrics to example
                example['quality_metrics'] = metrics
                filtered.append(example)
        
        return filtered
    
    def get_quality_distribution(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quality distribution of examples"""
        ranked = self.rank_examples(examples)
        
        if not ranked:
            return {}
        
        qualities = [metrics.overall_quality for _, metrics in ranked]
        
        return {
            'mean': np.mean(qualities),
            'std': np.std(qualities),
            'min': np.min(qualities),
            'max': np.max(qualities),
            'quartiles': {
                'q1': np.percentile(qualities, 25),
                'median': np.percentile(qualities, 50),
                'q3': np.percentile(qualities, 75)
            },
            'high_quality_ratio': len([q for q in qualities if q >= 0.8]) / len(qualities)
        }

class ICEBenchmark:
    """Benchmark ICE against baseline approaches"""
    
    def __init__(self):
        self.evaluator = ICEQualityEvaluator()
    
    def compare_approaches(self, ice_examples: List[Dict], baseline_examples: List[Dict]) -> Dict[str, Any]:
        """Compare ICE examples with baseline"""
        # Evaluate ICE examples
        ice_metrics = []
        for example in ice_examples:
            metrics = self.evaluator.evaluate_example(example)
            ice_metrics.append(metrics)
        
        # Evaluate baseline examples
        baseline_metrics = []
        for example in baseline_examples:
            metrics = self.evaluator.evaluate_example(example)
            baseline_metrics.append(metrics)
        
        # Calculate averages
        ice_avg = self._average_metrics(ice_metrics)
        baseline_avg = self._average_metrics(baseline_metrics)
        
        # Calculate improvements
        improvements = {}
        for field in ['semantic_coherence', 'technical_accuracy', 'overall_quality']:
            ice_val = getattr(ice_avg, field)
            baseline_val = getattr(baseline_avg, field)
            improvements[field] = (ice_val - baseline_val) / baseline_val if baseline_val > 0 else 0
        
        return {
            'ice_metrics': ice_avg,
            'baseline_metrics': baseline_avg,
            'improvements': improvements,
            'ice_wins': sum(1 for k, v in improvements.items() if v > 0),
            'significant_improvement': improvements['overall_quality'] > 0.1
        }
    
    def _average_metrics(self, metrics_list: List[QualityMetrics]) -> QualityMetrics:
        """Calculate average metrics"""
        if not metrics_list:
            return QualityMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        avg_dict = {}
        for field in QualityMetrics.__dataclass_fields__:
            values = [getattr(m, field) for m in metrics_list]
            avg_dict[field] = np.mean(values)
        
        return QualityMetrics(**avg_dict)

# Utility functions
def save_evaluation_report(examples: List[Dict], output_path: str):
    """Save detailed evaluation report"""
    evaluator = ICEQualityEvaluator()
    ranker = ICEExampleRanker(evaluator)
    
    # Rank examples
    ranked = ranker.rank_examples(examples)
    
    # Create report
    report = {
        'total_examples': len(examples),
        'evaluation_timestamp': str(Path.ctime(Path())),
        'quality_distribution': ranker.get_quality_distribution(examples),
        'top_examples': [],
        'metrics_breakdown': {}
    }
    
    # Add top 10 examples
    for example, metrics in ranked[:10]:
        report['top_examples'].append({
            'brief': example.get('brief', '')[:100] + '...',
            'metrics': metrics.__dict__
        })
    
    # Metrics breakdown
    for metric_name in QualityMetrics.__dataclass_fields__:
        values = [getattr(m, metric_name) for _, m in ranked]
        report['metrics_breakdown'][metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Evaluation report saved to {output_path}")
    
    return report

if __name__ == "__main__":
    # Example usage
    evaluator = ICEQualityEvaluator()
    
    # Test with sample example
    test_example = {
        'brief': "Write about the authentication system",
        'article': "# Authentication System\n\nThe authentication system uses JWT tokens...",
        'code_analysis': "Found auth module with login() and logout() functions",
        'metadata': {
            'final_confidence': 0.85,
            'consensus_reached': True
        }
    }
    
    metrics = evaluator.evaluate_example(test_example)
    print(f"Quality Metrics: {metrics}")
    print(f"Overall Quality: {metrics.overall_quality:.2%}")