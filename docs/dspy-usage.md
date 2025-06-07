# DSPy Usage Guide: Training Examples and Model Optimization

## Overview

This guide explains how to effectively use DSPy's training capabilities in the tech writer agent, including best practices for creating training examples and understanding the trade-offs between using DSPy with and without optimization.

## The Training Process

### How Training Examples Work

When you mark an article as "perfect" in the UI, the system saves:

```
training_examples/
└── example_abc123/
    ├── brief.txt           # Original user brief
    ├── final_article.md    # The perfected article
    ├── iterations.json     # All refinement steps
    ├── codebase_path.txt   # Which codebase was analyzed
    └── metadata.json       # Session metadata
```

These examples are used to optimize the DSPy modules through few-shot learning, teaching the model what makes a good technical article.

### Training Implementation (When Added)

```python
# Load examples
examples = [
    dspy.Example(
        brief=brief,
        article=article
    ).with_inputs('brief')
    for brief, article in load_saved_examples()
]

# Optimize with BootstrapFewShot
optimizer = BootstrapFewShotWithRandomSearch(
    metric=article_quality_metric,
    max_bootstrapped_demos=3,
    num_candidate_programs=10
)

# Compile optimized model
optimized_writer = optimizer.compile(writer, trainset=examples)
```

## Training Data Requirements

### Minimum Viable Dataset

- **Absolute minimum**: 3-5 examples (barely functional)
- **Recommended minimum**: 10-15 examples (noticeable improvement)
- **Ideal**: 20-30 diverse examples (robust performance)

### The Diversity Problem

Training with examples from a single brief creates severe limitations:

| Issue | Impact | Example |
|-------|--------|---------|
| **Overfitting** | Model only writes one type of article well | Trains on Redis tutorials → fails at API docs |
| **Style Lock-in** | Cannot adapt tone/structure | Always writes beginner tutorials even for advanced topics |
| **Vocabulary Limits** | Restricted technical terminology | Only knows caching terms, struggles with auth/database topics |
| **Structure Rigidity** | Single article format | Always creates step-by-step guides, can't write announcements |

### Recommended Training Diversity

For a well-rounded model, include examples across these dimensions:

1. **Article Types** (aim for 3-4 of each):
   - Step-by-step tutorials
   - Feature announcements  
   - Technical deep-dives
   - API documentation
   - Architecture overviews
   - Migration guides

2. **Technical Levels**:
   - Beginner-friendly explanations
   - Intermediate implementations
   - Advanced optimizations
   - Expert-level internals

3. **Codebase Types**:
   - Web applications
   - CLI tools
   - Libraries/frameworks
   - Microservices
   - Data pipelines

4. **Content Characteristics**:
   - Short updates (500-1000 words)
   - Medium guides (1500-2500 words)
   - Comprehensive docs (3000+ words)
   - Code-heavy vs. explanation-heavy
   - Visual/diagram-heavy vs. text-only

## The Effort Reality Check

### Creating Quality Training Data

Each training example requires:

1. **Finding/Creating a Good Brief** (5-10 min)
   - Must be realistic and clear
   - Should represent actual use cases

2. **Generating Initial Article** (2-5 min)
   - Running the agent
   - Waiting for generation

3. **Refinement Iterations** (15-30 min)
   - Multiple rounds of feedback
   - Testing different improvements
   - Ensuring high quality

4. **Quality Validation** (5-10 min)
   - Verifying accuracy
   - Checking completeness
   - Ensuring it's truly "perfect"

**Total per example**: 30-60 minutes
**For 20 examples**: 10-20 hours of work

### DSPy Without Training

You can use DSPy effectively without training:

```python
# Zero-shot usage (no examples)
writer = dspy.ChainOfThought(WriteArticle)
article = writer(brief=user_brief, ...)

# Few-shot with manual examples (no optimization)
writer = dspy.ChainOfThought(WriteArticle)
writer.demos = [
    dspy.Example(brief="...", article="..."),
    dspy.Example(brief="...", article="...")
]
```

**Advantages**:
- Immediate usage
- No training data needed
- Still benefits from DSPy's structured approach
- Good enough for many use cases

**Disadvantages**:
- Less consistent quality
- May need more refinement iterations
- Doesn't learn your specific preferences

## Practical Recommendations

### Start Without Training

1. Use the system in zero-shot mode initially
2. Collect "perfect" examples during normal usage
3. Only invest in training once you have natural diversity

### Progressive Enhancement

```
Phase 1: Zero-shot usage → Get familiar with the system
Phase 2: Collect 5-10 examples → See if patterns emerge  
Phase 3: Train with initial set → Evaluate improvement
Phase 4: Expand to 20+ examples → Only if Phase 3 shows promise
```

### When Training is Worth It

Training makes sense when:
- You have recurring similar briefs
- Quality consistency is critical
- You can invest the upfront time
- You have diverse use cases to train on

Training may not be worth it when:
- You have highly varied, one-off requests
- The zero-shot quality is already acceptable
- You don't have time for data collection
- Your use cases are too diverse to capture

### Alternative Approaches

Instead of full training, consider:

1. **Prompt Engineering**: Refine the base prompts in your signatures
2. **Better Briefs**: Create brief templates for common scenarios
3. **Selective Examples**: Hand-pick 2-3 excellent examples for few-shot
4. **Hybrid Approach**: Train only the most critical modules

## Conclusion

While DSPy's training capabilities are powerful, they require significant investment in creating diverse, high-quality training data. For many use cases, using DSPy without training—leveraging its structured approach to prompting and chaining—provides excellent results with minimal setup time.

The decision to invest in training should be based on your specific needs, available time, and the diversity of your use cases. Remember: an unoptimized DSPy system is still more structured and maintainable than ad-hoc prompting, and you can always add training later as you naturally accumulate good examples.