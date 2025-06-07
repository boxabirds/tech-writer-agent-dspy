"""
Streamlit app for ICE-based Tech Writer with synthetic example generation
"""

import streamlit as st
import dspy
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ice_dspy_tech_writer import (
    ICETechWriter, 
    ICESyntheticExampleGenerator,
    ConsensusResult
)
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="ICE Tech Writer - Consensus-Based Article Generation",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.ice-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.consensus-meter {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
.model-card {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}
.quality-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 500;
}
.quality-high { background-color: #10b981; color: white; }
.quality-medium { background-color: #f59e0b; color: white; }
.quality-low { background-color: #ef4444; color: white; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ice_writer' not in st.session_state:
    # Configure models
    models = st.session_state.get('selected_models', [
        "gpt-4o-mini",
        "claude-3-haiku-20240307",
        "gemini-2.0-flash"
    ])
    st.session_state.ice_writer = ICETechWriter(models=models)
    st.session_state.generator = ICESyntheticExampleGenerator(st.session_state.ice_writer)

if 'current_result' not in st.session_state:
    st.session_state.current_result = None

if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []

# Sidebar
with st.sidebar:
    st.title("🎛️ ICE Configuration")
    
    # Model selection
    st.markdown("### 🤖 Ensemble Models")
    available_models = [
        "gpt-4o-mini",
        "gpt-4o",
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
        "gemini-2.0-flash",
        "gemini-2.5-pro-preview-03-25"
    ]
    
    selected_models = st.multiselect(
        "Select 3+ diverse models",
        available_models,
        default=["gpt-4o-mini", "claude-3-haiku-20240307", "gemini-2.0-flash"],
        help="ICE requires at least 3 diverse models for effective consensus"
    )
    
    if len(selected_models) < 3:
        st.warning("⚠️ ICE works best with 3+ models")
    
    # Consensus parameters
    st.markdown("### 🎯 Consensus Settings")
    consensus_threshold = st.slider(
        "Consensus Threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.8,
        step=0.05,
        help="Similarity threshold for consensus detection"
    )
    
    max_rounds = st.slider(
        "Max Iteration Rounds",
        min_value=2,
        max_value=9,
        value=5,
        help="Maximum rounds before forcing consensus"
    )
    
    # Apply configuration
    if st.button("🔄 Update Configuration"):
        st.session_state.ice_writer = ICETechWriter(
            models=selected_models,
            consensus_threshold=consensus_threshold,
            max_rounds=max_rounds
        )
        st.success("✅ Configuration updated!")
    
    st.divider()
    
    # Statistics
    st.markdown("### 📊 Generation Stats")
    if st.session_state.generation_history:
        df = pd.DataFrame(st.session_state.generation_history)
        avg_rounds = df['total_rounds'].mean()
        avg_confidence = df['final_confidence'].mean()
        consensus_rate = (df['consensus_reached'].sum() / len(df)) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Rounds", f"{avg_rounds:.1f}")
            st.metric("Consensus Rate", f"{consensus_rate:.0f}%")
        with col2:
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
            st.metric("Total Examples", len(df))

# Main content
st.markdown('<div class="ice-header"><h1>🧊 ICE Tech Writer</h1><p>Iterative Consensus Ensemble for High-Quality Technical Articles</p></div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Generate Article", "🔄 Synthetic Examples", "📊 Consensus Analytics", "📚 Example Library"])

with tab1:
    # Article generation interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Article Brief")
        
        # Codebase selection
        codebase_path = st.text_input(
            "Codebase Path",
            placeholder="/path/to/repository",
            help="Path to the codebase to analyze"
        )
        
        # Brief input
        brief = st.text_area(
            "Technical Writing Brief",
            placeholder="Write a comprehensive guide about the authentication system implementation...",
            height=120
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            include_code_examples = st.checkbox("Include code examples", value=True)
            target_length = st.select_slider(
                "Target article length",
                options=["Short", "Medium", "Long", "Comprehensive"],
                value="Medium"
            )
    
    with col2:
        st.markdown("### 🎯 ICE Process")
        if st.session_state.current_result:
            result = st.session_state.current_result
            
            # Consensus meter
            st.markdown('<div class="consensus-meter">', unsafe_allow_html=True)
            confidence = result['metadata']['final_confidence']
            
            # Visual confidence meter
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                title={'text': "Consensus Confidence"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Process stats
            st.metric("Analysis Rounds", result['metadata']['analysis_rounds'])
            st.metric("Writing Rounds", result['metadata']['writing_rounds'])
            st.metric("Models Used", len(result['metadata']['models_used']))
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate button
    if st.button("🚀 Generate with ICE", type="primary", disabled=not (brief and codebase_path)):
        with st.spinner("🧊 Running Iterative Consensus Ensemble..."):
            try:
                # Show progress
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                # Mock progress updates (in real implementation, get from ICE)
                status_text.text("🔍 Phase 1: Analyzing codebase across models...")
                progress_bar.progress(0.25)
                
                # Generate article
                result = st.session_state.ice_writer(brief, codebase_path)
                
                status_text.text("✍️ Phase 2: Generating article with consensus...")
                progress_bar.progress(0.75)
                
                # Store result
                st.session_state.current_result = result
                
                # Add to history
                st.session_state.generation_history.append({
                    'timestamp': datetime.now(),
                    'brief': brief[:50] + "...",
                    'total_rounds': result['metadata']['analysis_rounds'] + result['metadata']['writing_rounds'],
                    'final_confidence': result['metadata']['final_confidence'],
                    'consensus_reached': result['metadata']['consensus_reached']
                })
                
                status_text.text("✅ Generation complete!")
                progress_bar.progress(1.0)
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display result
    if st.session_state.current_result:
        st.divider()
        
        result = st.session_state.current_result
        
        # Quality badge
        confidence = result['metadata']['final_confidence']
        if confidence >= 0.85:
            badge_class = "quality-high"
            badge_text = "High Quality"
        elif confidence >= 0.7:
            badge_class = "quality-medium"
            badge_text = "Medium Quality"
        else:
            badge_class = "quality-low"
            badge_text = "Low Quality"
        
        st.markdown(f'<span class="quality-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)
        
        # Article display
        st.markdown("### 📄 Generated Article")
        st.markdown(result['article'])
        
        # Model contributions
        with st.expander("🤖 Model Contributions"):
            for version in result.get('all_versions', []):
                st.markdown(f'<div class="model-card">', unsafe_allow_html=True)
                st.markdown(f"**{version['model']}** - Confidence: {version.get('confidence', 0):.2%}")
                st.text(version['article'][:200] + "...")
                st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    # Synthetic example generation
    st.markdown("### 🔄 Generate Synthetic Training Examples")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Batch Generation")
        
        # Input briefs
        num_examples = st.number_input("Number of examples", min_value=1, max_value=20, value=5)
        
        briefs_input = st.text_area(
            "Enter briefs (one per line)",
            height=150,
            placeholder="Write about authentication system\nExplain the caching layer\nDocument the API endpoints"
        )
        
        # Parse briefs
        briefs = [b.strip() for b in briefs_input.split('\n') if b.strip()]
        
        if st.button("🏭 Generate Batch", disabled=not (briefs and codebase_path)):
            with st.spinner(f"Generating {len(briefs)} examples..."):
                # Use same codebase for all
                codebase_paths = [codebase_path] * len(briefs)
                
                examples = st.session_state.generator.generate_example_batch(briefs, codebase_paths)
                
                st.success(f"✅ Generated {len(examples)} high-quality examples")
                
                # Display results
                for i, example in enumerate(examples):
                    with st.expander(f"Example {i+1}: {example['brief'][:50]}..."):
                        st.metric("Quality Score", f"{example['quality_score']:.2%}")
                        st.markdown(example['article'][:500] + "...")
    
    with col2:
        st.markdown("#### Bootstrap from Seeds")
        
        # Load seed examples
        seed_path = st.text_input("Seed examples directory", value="./training_examples")
        
        if Path(seed_path).exists():
            seed_files = list(Path(seed_path).glob("*/metadata.json"))
            st.info(f"Found {len(seed_files)} seed examples")
            
            num_synthetic = st.slider("Synthetic examples to generate", 1, 50, 10)
            
            if st.button("🌱 Bootstrap Examples"):
                # Load seeds
                seeds = []
                for seed_file in seed_files[:5]:  # Use first 5 seeds
                    with open(seed_file) as f:
                        metadata = json.load(f)
                        seeds.append({
                            'brief': metadata['brief'],
                            'codebase_path': metadata['codebase_path']
                        })
                
                # Generate synthetic examples
                with st.spinner(f"Bootstrapping {num_synthetic} examples..."):
                    synthetic = st.session_state.generator.bootstrap_from_seed_examples(
                        seeds, num_synthetic
                    )
                    
                    st.success(f"✅ Generated {len(synthetic)} synthetic examples")
                    
                    # Quality distribution
                    qualities = [ex['quality_score'] for ex in synthetic]
                    fig = px.histogram(
                        x=qualities,
                        nbins=20,
                        title="Quality Score Distribution",
                        labels={'x': 'Quality Score', 'y': 'Count'}
                    )
                    st.plotly_chart(fig)

with tab3:
    # Consensus analytics
    st.markdown("### 📊 Consensus Analytics")
    
    if st.session_state.generation_history:
        df = pd.DataFrame(st.session_state.generation_history)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Rounds distribution
            fig = px.histogram(
                df,
                x='total_rounds',
                title="Distribution of ICE Rounds",
                labels={'total_rounds': 'Total Rounds', 'count': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence over time
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            fig = px.line(
                df,
                x='timestamp',
                y='final_confidence',
                title="Consensus Confidence Over Time",
                labels={'final_confidence': 'Confidence', 'timestamp': 'Time'}
            )
            fig.update_yaxis(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Consensus achievement rate
            consensus_data = df.groupby('consensus_reached').size()
            fig = px.pie(
                values=consensus_data.values,
                names=['No Consensus', 'Consensus Reached'],
                title="Consensus Achievement Rate"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Model performance (mock data)
            model_performance = pd.DataFrame({
                'Model': ['GPT-4o-mini', 'Claude Haiku', 'Gemini Flash'],
                'Avg Confidence': [0.82, 0.78, 0.85],
                'Contributions': [35, 30, 35]
            })
            
            fig = px.bar(
                model_performance,
                x='Model',
                y='Avg Confidence',
                title="Model Performance Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Generate some articles to see analytics")

with tab4:
    # Example library
    st.markdown("### 📚 ICE Example Library")
    
    examples_dir = Path("./ice_training_examples")
    if examples_dir.exists():
        example_dirs = [d for d in examples_dir.iterdir() if d.is_dir()]
        
        if example_dirs:
            # Filter and search
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                search_query = st.text_input("🔍 Search examples", placeholder="authentication, caching, API...")
            with col2:
                quality_filter = st.select_slider(
                    "Min quality",
                    options=[0.6, 0.7, 0.8, 0.85, 0.9],
                    value=0.8
                )
            with col3:
                sort_by = st.selectbox("Sort by", ["Date", "Quality", "Rounds"])
            
            # Load and display examples
            examples = []
            for example_dir in example_dirs:
                try:
                    with open(example_dir / "metrics.json") as f:
                        metrics = json.load(f)
                    with open(example_dir / "session_data.json") as f:
                        session = json.load(f)
                    
                    if metrics['quality_score'] >= quality_filter:
                        examples.append({
                            'dir': example_dir,
                            'brief': session['brief'],
                            'quality': metrics['quality_score'],
                            'rounds': metrics['total_rounds'],
                            'timestamp': metrics['timestamp']
                        })
                except:
                    continue
            
            # Sort examples
            if sort_by == "Date":
                examples.sort(key=lambda x: x['timestamp'], reverse=True)
            elif sort_by == "Quality":
                examples.sort(key=lambda x: x['quality'], reverse=True)
            else:
                examples.sort(key=lambda x: x['rounds'])
            
            # Display examples
            for example in examples[:20]:  # Show top 20
                with st.expander(f"📄 {example['brief'][:60]}..."):
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        st.metric("Quality Score", f"{example['quality']:.2%}")
                    with col2:
                        st.metric("ICE Rounds", example['rounds'])
                    with col3:
                        st.caption(f"Generated: {example['timestamp'][:10]}")
                    
                    # Load and show article preview
                    try:
                        with open(example['dir'] / "session_data.json") as f:
                            data = json.load(f)
                        st.markdown("**Article Preview:**")
                        st.text(data['article'][:500] + "...")
                        
                        if st.button(f"📥 Export", key=f"export_{example['dir'].name}"):
                            st.download_button(
                                "Download Article",
                                data['article'],
                                file_name=f"{example['dir'].name}.md",
                                mime="text/markdown"
                            )
                    except:
                        st.error("Error loading article")
        else:
            st.info("No examples found. Generate some articles first!")
    else:
        st.info("Example library not initialized yet.")