import streamlit as st
import dspy
from pathlib import Path
import json
from datetime import datetime
import hashlib
from typing import Dict, List, Optional, Tuple
import pandas as pd
import plotly.express as px
import os
import shutil
import pathspec
from pathspec.patterns import GitWildMatchPattern
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

try:
    from binaryornot.check import is_binary
except ImportError:
    logger.warning("binaryornot not installed, binary detection disabled")
    # Fallback if binaryornot not installed
    def is_binary(filepath):
        return False

# Configure page
st.set_page_config(
    page_title="AI Tech Writer - Code Analyzer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure DSPy (only once)
if 'dspy_configured' not in st.session_state:
    # Note: DSPy caches responses by default. To disable caching:
    # import dspy.cache_utils
    # dspy.cache_utils.clear_cache()
    # or set cache_turn_on=False in the LM constructor
    dspy.settings.configure(lm=dspy.LM("openai/gpt-4.1-mini"))
    st.session_state.dspy_configured = True

# DSPy Signatures
class InterpretBrief(dspy.Signature):
    """Extract structured requirements from a plain English brief."""
    brief = dspy.InputField(desc="Plain English brief from non-technical user")
    codebase_path = dspy.InputField(desc="Path to the codebase to analyze")
    tech_spec = dspy.OutputField(desc="Technical requirements and key points to cover")
    target_audience = dspy.OutputField(desc="Inferred target audience")
    article_type = dspy.OutputField(desc="Type: tutorial/explanation/announcement/guide")

class WriteArticle(dspy.Signature):
    """Write a technical article based on code analysis and requirements."""
    brief = dspy.InputField(desc="Original brief")
    tech_spec = dspy.InputField(desc="Technical requirements")
    code_analysis = dspy.InputField(desc="Analysis of the codebase")
    audience = dspy.InputField(desc="Target audience")
    article = dspy.OutputField(desc="Complete technical article in markdown")

class RefineArticle(dspy.Signature):
    """Refine article based on user feedback."""
    article = dspy.InputField(desc="Current article version")
    feedback = dspy.InputField(desc="User feedback on what to change")
    original_brief = dspy.InputField(desc="Original brief for context")
    code_analysis = dspy.InputField(desc="Code analysis for reference")
    refined_article = dspy.OutputField(desc="Improved article incorporating feedback")

# Code Analysis Tools
class CodeAnalysisTools:
    """Tools for analyzing codebases - similar to your original example."""
    
    @staticmethod
    def get_gitignore_spec(directory: str) -> pathspec.PathSpec:
        """
        Get a PathSpec object from .gitignore in the specified directory.
        
        Args:
            directory: The directory containing .gitignore
            
        Returns:
            A PathSpec object for matching against .gitignore patterns
        """
        ignore_patterns = []
        
        # Always ignore these patterns
        default_patterns = [
            '.git/',
            '__pycache__/',
            '*.pyc',
            'node_modules/',
            '.env',
            '.venv/',
            'venv/',
            '.DS_Store'
        ]
        ignore_patterns.extend(default_patterns)
        
        # Try to read .gitignore file
        gitignore_path = Path(directory) / ".gitignore"
        if gitignore_path.exists():
            try:
                with open(gitignore_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        # Skip empty lines and comments
                        if line and not line.startswith("#"):
                            ignore_patterns.append(line)
            except (IOError, UnicodeDecodeError):
                pass
        
        # Create pathspec matcher
        return pathspec.PathSpec.from_lines(
            GitWildMatchPattern, ignore_patterns
        )
    
    @staticmethod
    def read_file(filepath: str) -> str:
        """Read a file from the filesystem."""
        try:
            path = Path(filepath)
            if not path.exists():
                return f"File not found: {filepath}"
            
            if is_binary(str(path)):
                return f"Cannot read binary file: {filepath}"
            
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            return f"Cannot decode file as UTF-8: {filepath}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    @staticmethod
    def list_files(directory: str, pattern: str = "*", respect_gitignore: bool = True) -> List[str]:
        """List files in a directory matching a pattern while respecting .gitignore."""
        try:
            path = Path(directory)
            if not path.exists() or not path.is_dir():
                return []
            
            # Get gitignore spec if needed
            spec = CodeAnalysisTools.get_gitignore_spec(str(path)) if respect_gitignore else None
            
            result = []
            files = list(path.rglob(pattern))
            
            for f in files:
                if f.is_file():
                    # Skip hidden files
                    if any(part.startswith('.') for part in f.parts[len(path.parts):]):
                        continue
                    
                    # Check gitignore
                    if respect_gitignore and spec:
                        rel_path = f.relative_to(path)
                        if spec.match_file(str(rel_path)):
                            continue
                    
                    result.append(str(f.relative_to(path)))
            
            return result
        except Exception as e:
            return [f"Error listing files: {str(e)}"]
    
    @staticmethod
    def analyze_structure(directory: str) -> Dict:
        """Analyze the structure of a codebase."""
        structure = {
            'total_files': 0,
            'file_types': {},
            'directories': [],
            'key_files': []
        }
        
        try:
            path = Path(directory)
            if not path.exists():
                return {'error': f'Directory not found: {directory}'}
            
            # Get gitignore spec
            spec = CodeAnalysisTools.get_gitignore_spec(str(path))
            
            # Count files by type
            for file in path.rglob('*'):
                if file.is_file():
                    # Skip gitignored files
                    rel_path = file.relative_to(path)
                    if spec.match_file(str(rel_path)):
                        continue
                    
                    structure['total_files'] += 1
                    ext = file.suffix.lower()
                    structure['file_types'][ext] = structure['file_types'].get(ext, 0) + 1
                    
                    # Identify key files
                    if file.name in ['README.md', 'setup.py', 'package.json', 'requirements.txt', 'Cargo.toml', 'pyproject.toml']:
                        structure['key_files'].append(str(file.relative_to(path)))
            
            # Get directory structure
            for subdir in path.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    structure['directories'].append(subdir.name)
            
            return structure
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def extract_functions(filepath: str) -> List[str]:
        """Extract function/method names from a code file."""
        try:
            content = CodeAnalysisTools.read_file(filepath)
            functions = []
            
            # Simple extraction - could be made more sophisticated
            lines = content.split('\n')
            for line in lines:
                if 'def ' in line or 'function ' in line or 'class ' in line:
                    # Extract function/class name
                    if 'def ' in line:
                        name = line.split('def ')[1].split('(')[0].strip()
                        functions.append(f"Function: {name}")
                    elif 'class ' in line:
                        name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
                        functions.append(f"Class: {name}")
                    elif 'function ' in line and '(' in line:
                        name = line.split('function ')[1].split('(')[0].strip()
                        functions.append(f"Function: {name}")
            
            return functions
        except Exception as e:
            return [f"Error extracting functions: {str(e)}"]
    
    @staticmethod
    def find_main_entry(directory: str) -> Optional[str]:
        """Find the main entry point of a project."""
        common_entries = [
            'main.py', 'app.py', 'index.js', 'main.js', 'index.ts', 
            'main.rs', 'main.go', 'Main.java', 'Program.cs'
        ]
        
        path = Path(directory)
        for entry in common_entries:
            entry_path = path / entry
            if entry_path.exists():
                return str(entry_path.relative_to(path))
        
        # Check for executable scripts
        for file in path.glob('*.py'):
            content = CodeAnalysisTools.read_file(str(file))
            if '__main__' in content:
                return str(file.relative_to(path))
        
        return None

# ReAct Tech Writer with Code Analysis
class ReactCodeAwareTechWriter(dspy.Module):
    def __init__(self, examples_dir: str = "./training_examples"):
        super().__init__()
        self.examples_dir = Path(examples_dir)
        self.examples_dir.mkdir(exist_ok=True)
        
        # Core modules
        self.brief_interpreter = dspy.ChainOfThought(InterpretBrief)
        
        # ReAct module with code analysis tools
        self.writer = dspy.ReAct(
            WriteArticle,
            tools=[
                self.analyze_codebase,
                self.read_source_file,
                self.check_documentation,
                self.find_examples,
                self.analyze_dependencies
            ],
            max_iters=8
        )
        
        # Create refiner with verbose mode for debugging
        self.refiner = dspy.ReAct(
            RefineArticle,
            tools=[
                self.read_source_file,
                self.find_examples,
                self.check_specific_implementation
            ],
            max_iters=5
        )
        
        # Enable verbose mode for debugging
        import os
        if os.getenv('DSPY_DEBUG'):
            logger.info("DSPy debug mode enabled")
            dspy.settings.configure(experimental=True)
        
        # Track current session
        self.current_session = None
        self.current_codebase = None
    
    # Tool methods for ReAct
    def analyze_codebase(self, directory: str) -> str:
        """Analyze the overall structure of the codebase."""
        structure = CodeAnalysisTools.analyze_structure(directory)
        
        result = f"Codebase Analysis for {directory}:\n"
        result += f"- Total files: {structure.get('total_files', 0)}\n"
        result += f"- File types: {structure.get('file_types', {})}\n"
        result += f"- Main directories: {', '.join(structure.get('directories', []))}\n"
        result += f"- Key files: {', '.join(structure.get('key_files', []))}\n"
        
        # Find main entry
        main_entry = CodeAnalysisTools.find_main_entry(directory)
        if main_entry:
            result += f"- Main entry point: {main_entry}\n"
        
        return result
    
    def read_source_file(self, filepath: str) -> str:
        """Read a specific source file."""
        logger.debug(f"read_source_file called with: {filepath}")
        logger.debug(f"current_codebase: {self.current_codebase}")
        if self.current_codebase:
            full_path = Path(self.current_codebase) / filepath
            return CodeAnalysisTools.read_file(str(full_path))
        return "No codebase path set"
    
    def check_documentation(self, directory: str) -> str:
        """Check for existing documentation."""
        doc_files = []
        path = Path(directory)
        
        # Look for documentation files
        for pattern in ['README*', 'CONTRIBUTING*', 'docs/*', '*.md']:
            doc_files.extend(CodeAnalysisTools.list_files(directory, pattern))
        
        if doc_files:
            result = f"Found documentation files:\n"
            for doc in doc_files[:10]:  # Limit to first 10
                result += f"- {doc}\n"
            
            # Read README if exists
            readme_path = path / "README.md"
            if readme_path.exists():
                content = CodeAnalysisTools.read_file(str(readme_path))
                result += f"\nREADME.md preview:\n{content[:500]}...\n"
        else:
            result = "No documentation files found"
        
        return result
    
    def find_examples(self, directory: str) -> str:
        """Find example code in the repository."""
        example_files = []
        
        # Look for example files
        for pattern in ['examples/**/*', 'example/**/*', 'samples/**/*', 'demo/**/*', '*example*', '*demo*']:
            files = CodeAnalysisTools.list_files(directory, pattern)
            example_files.extend([f for f in files if not f.endswith('.pyc')])
        
        if example_files:
            result = f"Found example files:\n"
            for ex in example_files[:10]:
                result += f"- {ex}\n"
                
            # Read first example
            if example_files:
                first_example = Path(directory) / example_files[0]
                if first_example.suffix in ['.py', '.js', '.ts', '.java', '.go', '.rs']:
                    content = CodeAnalysisTools.read_file(str(first_example))
                    result += f"\nExample from {example_files[0]}:\n```\n{content[:500]}...\n```\n"
        else:
            result = "No example files found"
        
        return result
    
    def analyze_dependencies(self, directory: str) -> str:
        """Analyze project dependencies."""
        path = Path(directory)
        result = "Dependencies:\n"
        
        # Python
        requirements = path / "requirements.txt"
        if requirements.exists():
            content = CodeAnalysisTools.read_file(str(requirements))
            result += f"\nPython (requirements.txt):\n{content[:300]}\n"
        
        # Node.js
        package_json = path / "package.json"
        if package_json.exists():
            content = CodeAnalysisTools.read_file(str(package_json))
            try:
                import json
                data = json.loads(content)
                deps = data.get('dependencies', {})
                result += f"\nNode.js dependencies: {', '.join(list(deps.keys())[:10])}\n"
            except:
                pass
        
        # Other dependency files
        for dep_file in ['Cargo.toml', 'go.mod', 'pom.xml', 'build.gradle']:
            dep_path = path / dep_file
            if dep_path.exists():
                result += f"\nFound {dep_file}\n"
        
        return result
    
    def check_specific_implementation(self, query: str) -> str:
        """Check for specific implementation details."""
        if not self.current_codebase:
            return "No codebase set"
        
        # Search for files containing the query
        results = []
        path = Path(self.current_codebase)
        spec = CodeAnalysisTools.get_gitignore_spec(str(path))
        
        # Define common code file extensions
        code_extensions = ['.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c', '.cs', '.rb', '.php']
        
        # Search through code files
        for ext in code_extensions:
            for file in path.rglob(f'*{ext}'):
                # Skip gitignored files
                rel_path = file.relative_to(path)
                if spec.match_file(str(rel_path)):
                    continue
                
                # Skip binary files
                if is_binary(str(file)):
                    continue
                
                try:
                    content = file.read_text()
                    if query.lower() in content.lower():
                        # Find the line containing the query
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if query.lower() in line.lower():
                                results.append(f"{rel_path}:{i+1}: {line.strip()}")
                                if len(results) >= 20:  # Limit results
                                    break
                except:
                    continue
                
                if len(results) >= 20:
                    break
        
        if results:
            return f"Found '{query}' in:\n" + "\n".join(results[:10])
        return f"No occurrences of '{query}' found"
    
    def forward(self, brief: str, codebase_path: str) -> Dict:
        """Generate initial article from brief and codebase."""
        logger.info(f"Starting article generation for codebase: {codebase_path}")
        logger.debug(f"Brief: {brief[:200]}...")
        
        # Set current codebase
        self.current_codebase = codebase_path
        
        # Create new session
        session_id = hashlib.md5(f"{brief}{datetime.now()}".encode()).hexdigest()[:8]
        self.current_session = {
            'id': session_id,
            'brief': brief,
            'codebase_path': codebase_path,
            'iterations': [],
            'start_time': datetime.now().isoformat()
        }
        
        # Interpret brief with codebase context
        interpretation = self.brief_interpreter(
            brief=brief,
            codebase_path=codebase_path
        )
        
        # Initial code analysis
        code_analysis = self.analyze_codebase(codebase_path)
        
        # Generate article using ReAct
        # The ReAct agent will use the tools to explore the codebase as needed
        article = self.writer(
            brief=brief,
            tech_spec=interpretation.tech_spec,
            code_analysis=code_analysis,
            audience=interpretation.target_audience
        )
        
        # Store first iteration
        self.current_session['iterations'].append({
            'version': 1,
            'article': article.article,
            'interpretation': {
                'tech_spec': interpretation.tech_spec,
                'target_audience': interpretation.target_audience,
                'article_type': interpretation.article_type
            },
            'code_analysis': code_analysis,
            'feedback': None
        })
        
        return {
            'session_id': session_id,
            'article': article.article,
            'interpretation': interpretation,
            'code_analysis': code_analysis,
            'version': 1
        }
    
    def refine_with_feedback(self, feedback: str) -> Dict:
        """Refine article based on user feedback."""
        logger.info(f"Starting refinement with feedback: {feedback[:100]}...")
        
        if not self.current_session:
            raise ValueError("No active session. Call forward() first.")
        
        # Ensure current_codebase is set
        if not self.current_codebase and 'codebase_path' in self.current_session:
            self.current_codebase = self.current_session['codebase_path']
            logger.info(f"Restored current_codebase: {self.current_codebase}")
        
        # Get current state
        current_article = self.current_session['iterations'][-1]['article']
        code_analysis = self.current_session['iterations'][-1]['code_analysis']
        
        logger.debug(f"Current article length: {len(current_article)}")
        logger.debug(f"Code analysis available: {bool(code_analysis)}")
        logger.debug(f"Current codebase: {self.current_codebase}")
        
        try:
            # Refine using ReAct with tools
            logger.info("Calling DSPy refiner...")
            
            # Log the actual values being passed
            logger.debug("Refiner inputs:")
            logger.debug(f"  - article length: {len(current_article)}")
            logger.debug(f"  - feedback: {feedback}")
            logger.debug(f"  - original_brief: {self.current_session['brief'][:100]}...")
            logger.debug(f"  - code_analysis length: {len(code_analysis) if code_analysis else 0}")
            
            # Add timestamp to prevent caching issues
            import time
            timestamp = str(time.time())
            
            refined = self.refiner(
                article=current_article,
                feedback=feedback,
                original_brief=self.current_session['brief'],
                code_analysis=code_analysis + f"\n[Refinement timestamp: {timestamp}]"
            )
            
            logger.info("Refinement completed successfully")
            
            # Check if refined has the expected structure
            if hasattr(refined, 'refined_article'):
                logger.debug(f"Refined article length: {len(refined.refined_article)}")
                logger.debug(f"Article changed: {refined.refined_article != current_article}")
                
                # Detailed diff logging
                if refined.refined_article == current_article:
                    logger.warning("WARNING: Article did not change after refinement!")
                    logger.warning("This suggests the feedback was not processed.")
                else:
                    # Log a simple diff summary
                    import difflib
                    diff = difflib.unified_diff(
                        current_article.splitlines(keepends=True),
                        refined.refined_article.splitlines(keepends=True),
                        fromfile='current',
                        tofile='refined',
                        n=3
                    )
                    diff_lines = list(diff)
                    if diff_lines:
                        logger.info(f"Article changed - {len(diff_lines)} diff lines")
                        # Log first few diff lines for debugging
                        logger.debug("First 10 diff lines:")
                        for line in diff_lines[:10]:
                            logger.debug(line.rstrip())
            else:
                logger.error(f"Unexpected refined structure: {type(refined)}")
                logger.error(f"Refined attributes: {dir(refined)}")
        except Exception as e:
            logger.error(f"Error during refinement: {str(e)}", exc_info=True)
            raise
        
        # Store iteration
        version = len(self.current_session['iterations']) + 1
        self.current_session['iterations'].append({
            'version': version,
            'article': refined.refined_article,
            'code_analysis': code_analysis,
            'feedback': feedback
        })
        
        return {
            'session_id': self.current_session['id'],
            'article': refined.refined_article,
            'version': version,
            'code_analysis': code_analysis,
            'interpretation': self.current_session['iterations'][0]['interpretation']
        }
    
    def save_perfect_example(self) -> str:
        """Save the current session as a perfect training example."""
        if not self.current_session:
            raise ValueError("No active session to save.")
        
        # Create example directory
        example_dir = self.examples_dir / f"example_{self.current_session['id']}"
        example_dir.mkdir(exist_ok=True)
        
        # Save all components
        with open(example_dir / "brief.txt", 'w') as f:
            f.write(self.current_session['brief'])
        
        with open(example_dir / "codebase_path.txt", 'w') as f:
            f.write(self.current_session['codebase_path'])
        
        final_article = self.current_session['iterations'][-1]['article']
        with open(example_dir / "final_article.md", 'w') as f:
            f.write(final_article)
        
        with open(example_dir / "iterations.json", 'w') as f:
            json.dump(self.current_session['iterations'], f, indent=2)
        
        metadata = {
            'session_id': self.current_session['id'],
            'brief': self.current_session['brief'],
            'codebase_path': self.current_session['codebase_path'],
            'total_iterations': len(self.current_session['iterations']),
            'start_time': self.current_session['start_time'],
            'end_time': datetime.now().isoformat(),
            'final_interpretation': self.current_session['iterations'][0]['interpretation']
        }
        
        with open(example_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Clear session
        self.current_session = None
        self.current_codebase = None
        
        return str(example_dir)
    
    def get_example_stats(self) -> Dict:
        """Get statistics about saved examples."""
        stats = {
            'total_examples': 0,
            'total_iterations': 0,
            'briefs_by_type': {},
            'iterations_distribution': []
        }
        
        if not self.examples_dir.exists():
            return stats
        
        for example_dir in self.examples_dir.iterdir():
            if example_dir.is_dir() and example_dir.name.startswith("example_"):
                stats['total_examples'] += 1
                
                # Load metadata
                metadata_path = example_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                        
                    # Count iterations
                    iterations = metadata.get('total_iterations', 0)
                    stats['total_iterations'] += iterations
                    stats['iterations_distribution'].append(iterations)
                    
                    # Count by type
                    article_type = metadata.get('final_interpretation', {}).get('article_type', 'unknown')
                    stats['briefs_by_type'][article_type] = stats['briefs_by_type'].get(article_type, 0) + 1
        
        return stats

# Streamlit UI
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.code-analysis-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'writer' not in st.session_state:
    st.session_state.writer = ReactCodeAwareTechWriter()

if 'current_article' not in st.session_state:
    st.session_state.current_article = None

# Sidebar
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    # Codebase selection
    st.markdown("### 📁 Select Codebase")
    
    # Simple path input
    codebase_path = st.text_input(
        "Repository path",
        value=st.session_state.get('codebase_path', ''),
        placeholder="/path/to/your/repository",
        help="Enter the full path to your repository/codebase"
    )
    
    # Validate and set path
    if codebase_path:
        path = Path(codebase_path)
        if path.exists() and path.is_dir():
            st.session_state.codebase_path = codebase_path
            st.success(f"✅ Found: {path.name}")
            
            # Show some info about the directory
            try:
                spec = CodeAnalysisTools.get_gitignore_spec(str(path))
                file_count = 0
                for f in path.rglob('*'):
                    if f.is_file():
                        rel_path = f.relative_to(path)
                        if not spec.match_file(str(rel_path)):
                            file_count += 1
                st.caption(f"📊 {file_count} files in repository (excluding gitignored)")
            except:
                pass
        else:
            st.error("❌ Directory not found")
    
    # Quick access to recent paths
    if 'recent_paths' not in st.session_state:
        st.session_state.recent_paths = []
    
    if st.session_state.recent_paths:
        st.caption("Recent repositories:")
        for recent_path in st.session_state.recent_paths[-3:]:
            if st.button(f"📁 {Path(recent_path).name}", key=f"recent_{recent_path}"):
                st.session_state.codebase_path = recent_path
                st.rerun()
    
    # Add current path to recent if valid
    if codebase_path and Path(codebase_path).exists() and codebase_path not in st.session_state.recent_paths:
        st.session_state.recent_paths.append(codebase_path)
    
    st.divider()
    
    # Stats
    stats = st.session_state.writer.get_example_stats() if hasattr(st.session_state.writer, 'get_example_stats') else {'total_examples': 0}
    st.metric("Training Examples", stats.get('total_examples', 0))

# Main interface
st.markdown('<h1 class="main-header">✍️ Code-Aware Tech Writer</h1>', unsafe_allow_html=True)

# Brief input section
if st.session_state.current_article is None:
    st.markdown("### 📝 Article Brief")
    
    # Option to load brief from file
    col1, col2 = st.columns([3, 1])
    
    with col1:
        brief_source = st.radio(
            "Brief source",
            ["✏️ Write brief", "📄 Load from file"],
            horizontal=True
        )
    
    brief = ""
    
    if brief_source == "✏️ Write brief":
        brief = st.text_area(
            "Enter your brief",
            placeholder="Write a technical guide about the caching implementation in this codebase. Focus on the Redis integration and performance optimizations.",
            height=120
        )
    else:
        uploaded_file = st.file_uploader("Choose brief file", type=['txt', 'md'])
        if uploaded_file:
            brief = uploaded_file.read().decode('utf-8')
            st.text_area("Brief preview", brief, height=120, disabled=True)
    
    # Generate button
    if st.button("🚀 Generate Article", type="primary", 
                disabled=not brief or not st.session_state.get('codebase_path')):
        with st.spinner("Analyzing codebase and generating article..."):
            try:
                result = st.session_state.writer(
                    brief=brief,
                    codebase_path=st.session_state.codebase_path
                )
                st.session_state.current_article = result
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    # Display current article
    result = st.session_state.current_article
    
    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"### 📄 Article v{result['version']}")
    with col3:
        if st.button("❌ Start Over"):
            st.session_state.current_article = None
            st.rerun()
    
    # Show code analysis
    with st.expander("🔍 Code Analysis", expanded=False):
        st.markdown(f"**Repository:** {st.session_state.codebase_path}")
        st.code(result.get('code_analysis', 'No analysis available'), language='text')
    
    # Article display
    st.markdown("### 📖 Generated Article")
    st.markdown(result['article'])
    
    # Refinement section
    st.markdown("### 🔄 Refinement")
    
    feedback = st.text_area(
        "What changes would you like?",
        placeholder="Add more code examples from the utils module...",
        height=100
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔄 Refine", type="primary", disabled=not feedback):
            logger.info(f"Refine button clicked with feedback: {feedback}")
            with st.spinner("Refining article..."):
                try:
                    logger.info("Calling refine_with_feedback...")
                    refined = st.session_state.writer.refine_with_feedback(feedback)
                    logger.info(f"Refinement returned version {refined.get('version', 'unknown')}")
                    
                    # Check if article actually changed
                    if refined['article'] == result['article']:
                        st.warning("⚠️ The article did not change. The feedback may not have been processed correctly.")
                        logger.warning("Article unchanged after refinement")
                    else:
                        st.session_state.current_article = refined
                        st.success(f"✅ Refined to version {refined['version']}!")
                        
                        # Show a brief summary of changes
                        old_len = len(result['article'])
                        new_len = len(refined['article'])
                        diff_len = new_len - old_len
                        if diff_len > 0:
                            st.info(f"📝 Article grew by {diff_len} characters")
                        elif diff_len < 0:
                            st.info(f"📝 Article shortened by {abs(diff_len)} characters")
                        else:
                            st.info("📝 Article length unchanged but content modified")
                    
                    logger.info("Triggering rerun...")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error during refinement: {str(e)}", exc_info=True)
                    st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("✅ Save as Perfect", type="secondary"):
            try:
                path = st.session_state.writer.save_perfect_example()
                st.success("✅ Perfect example saved!")
                st.balloons()
                st.session_state.current_article = None
            except Exception as e:
                st.error(f"Error saving: {str(e)}")