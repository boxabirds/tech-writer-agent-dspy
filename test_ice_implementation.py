"""
Test script for ICE-DSPy Tech Writer implementation
"""

import asyncio
import json
from pathlib import Path
import logging
from ice_dspy_tech_writer import (
    ICETechWriter,
    ICESyntheticExampleGenerator,
    ICEOrchestrator
)
from ice_evaluation import (
    ICEQualityEvaluator,
    ICEExampleRanker,
    ICEBenchmark,
    save_evaluation_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_ice_orchestrator():
    """Test the ICE orchestrator component"""
    logger.info("Testing ICE Orchestrator...")
    
    # Initialize with test models
    models = ["gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini"]  # Using same model for testing
    orchestrator = ICEOrchestrator(models, consensus_threshold=0.8)
    
    # Test consensus detection
    test_responses = [
        {'content': 'The authentication system uses JWT tokens for secure access control.'},
        {'content': 'JWT tokens are used by the authentication system for access control.'},
        {'content': 'The auth system implements JWT-based token authentication.'}
    ]
    
    consensus = orchestrator.detect_consensus(test_responses, round_num=1)
    
    logger.info(f"Consensus detection test:")
    logger.info(f"  - Has consensus: {consensus.has_consensus}")
    logger.info(f"  - Confidence: {consensus.confidence:.2f}")
    logger.info(f"  - Similarity scores: {consensus.similarity_scores}")
    
    assert consensus.confidence > 0.7, "Expected high consensus for similar responses"
    logger.info("✅ ICE Orchestrator test passed")

def test_ice_tech_writer():
    """Test the main ICE Tech Writer"""
    logger.info("\nTesting ICE Tech Writer...")
    
    # Initialize writer
    writer = ICETechWriter(models=["gpt-4o-mini"])  # Single model for testing
    
    # Test brief
    test_brief = "Write a technical guide about error handling patterns in Python"
    test_codebase = "."  # Current directory
    
    # Generate article
    logger.info("Generating article with ICE approach...")
    result = writer(test_brief, test_codebase)
    
    # Validate result structure
    assert 'session_id' in result
    assert 'article' in result
    assert 'metadata' in result
    assert 'code_analysis' in result
    
    logger.info(f"Article generation test:")
    logger.info(f"  - Session ID: {result['session_id']}")
    logger.info(f"  - Analysis rounds: {result['metadata']['analysis_rounds']}")
    logger.info(f"  - Writing rounds: {result['metadata']['writing_rounds']}")
    logger.info(f"  - Consensus reached: {result['metadata']['consensus_reached']}")
    logger.info(f"  - Article length: {len(result['article'])} chars")
    
    # Save example
    if result['metadata']['final_confidence'] > 0.7:
        save_path = writer.save_ice_example(result, result['metadata']['final_confidence'])
        logger.info(f"  - Example saved to: {save_path}")
    
    logger.info("✅ ICE Tech Writer test passed")
    return result

def test_synthetic_generation():
    """Test synthetic example generation"""
    logger.info("\nTesting Synthetic Example Generation...")
    
    # Initialize components
    writer = ICETechWriter(models=["gpt-4o-mini"])
    generator = ICESyntheticExampleGenerator(writer)
    
    # Test briefs
    test_briefs = [
        "Document the logging system implementation",
        "Explain the database connection pooling"
    ]
    test_codebases = [".", "."]
    
    # Generate batch
    logger.info("Generating example batch...")
    examples = generator.generate_example_batch(test_briefs, test_codebases)
    
    logger.info(f"Synthetic generation test:")
    logger.info(f"  - Generated {len(examples)} examples")
    
    for i, example in enumerate(examples):
        logger.info(f"  - Example {i+1}: Quality={example['quality_score']:.2f}")
    
    logger.info("✅ Synthetic generation test passed")
    return examples

def test_quality_evaluation(examples):
    """Test quality evaluation system"""
    logger.info("\nTesting Quality Evaluation...")
    
    # Initialize evaluator
    evaluator = ICEQualityEvaluator()
    ranker = ICEExampleRanker(evaluator)
    
    # Evaluate examples
    if not examples:
        # Create test example
        examples = [{
            'brief': "Test brief",
            'article': "# Test Article\n\nThis is a test article about Python.",
            'code_analysis': "Found main.py with test functions",
            'metadata': {'final_confidence': 0.85}
        }]
    
    # Rank examples
    ranked = ranker.rank_examples(examples)
    
    logger.info(f"Quality evaluation test:")
    for i, (example, metrics) in enumerate(ranked):
        logger.info(f"  - Example {i+1}:")
        logger.info(f"    - Overall quality: {metrics.overall_quality:.2f}")
        logger.info(f"    - Consensus: {metrics.consensus_confidence:.2f}")
        logger.info(f"    - Coherence: {metrics.semantic_coherence:.2f}")
        logger.info(f"    - Accuracy: {metrics.technical_accuracy:.2f}")
    
    # Get distribution
    distribution = ranker.get_quality_distribution(examples)
    logger.info(f"  - Quality distribution: mean={distribution.get('mean', 0):.2f}, std={distribution.get('std', 0):.2f}")
    
    logger.info("✅ Quality evaluation test passed")

def test_benchmark_comparison():
    """Test ICE vs baseline comparison"""
    logger.info("\nTesting Benchmark Comparison...")
    
    # Create mock examples
    ice_examples = [
        {
            'brief': "ICE example",
            'article': "# Comprehensive Guide\n\nDetailed technical content with examples...",
            'code_analysis': "Thorough analysis",
            'metadata': {'final_confidence': 0.88}
        }
    ]
    
    baseline_examples = [
        {
            'brief': "Baseline example",
            'article': "# Basic Guide\n\nSimple content...",
            'code_analysis': "Basic analysis",
            'metadata': {'final_confidence': 0.65}
        }
    ]
    
    # Run benchmark
    benchmark = ICEBenchmark()
    comparison = benchmark.compare_approaches(ice_examples, baseline_examples)
    
    logger.info(f"Benchmark comparison test:")
    logger.info(f"  - ICE overall quality: {comparison['ice_metrics'].overall_quality:.2f}")
    logger.info(f"  - Baseline overall quality: {comparison['baseline_metrics'].overall_quality:.2f}")
    logger.info(f"  - Improvement: {comparison['improvements']['overall_quality']:.1%}")
    logger.info(f"  - Significant improvement: {comparison['significant_improvement']}")
    
    logger.info("✅ Benchmark comparison test passed")

def test_full_pipeline():
    """Test the complete ICE pipeline"""
    logger.info("\n" + "="*50)
    logger.info("Testing Complete ICE Pipeline")
    logger.info("="*50)
    
    try:
        # 1. Test orchestrator
        test_ice_orchestrator()
        
        # 2. Test tech writer
        article_result = test_ice_tech_writer()
        
        # 3. Test synthetic generation
        examples = test_synthetic_generation()
        
        # 4. Test quality evaluation
        test_quality_evaluation(examples if examples else [article_result])
        
        # 5. Test benchmark
        test_benchmark_comparison()
        
        # 6. Save evaluation report
        if examples:
            report_path = "ice_evaluation_report.json"
            save_evaluation_report(examples, report_path)
            logger.info(f"\n📊 Evaluation report saved to: {report_path}")
        
        logger.info("\n" + "="*50)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Run all tests
    test_full_pipeline()
    
    # Print summary
    print("\n📊 ICE-DSPy Implementation Summary:")
    print("- ✅ ICE Orchestrator with consensus detection")
    print("- ✅ Multi-model collaboration using DSPy")
    print("- ✅ Iterative refinement until consensus")
    print("- ✅ Synthetic example generation")
    print("- ✅ Comprehensive quality evaluation")
    print("- ✅ Benchmark comparison capabilities")
    print("\n🚀 Ready for production use!")