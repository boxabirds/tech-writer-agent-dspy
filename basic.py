import dspy
import os
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
import ast
import json

# Configure DSPy
dspy.settings.configure(lm=dspy.LM("anthropic/claude-3-5-sonnet-20241022"))

# Signatures for the writing process
class WriteTechArticle(dspy.Signature):
    """Write a technical article based on a brief and repository analysis."""
    brief = dspy.InputField(desc="User's brief for the article")
    article = dspy.OutputField(desc="Complete technical article in markdown")

# EXACT TOOLS FROM THE ORIGINAL CODE
def read_file(file_path):
    """Read and return the contents of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(file_path, content):
    """Write content to a file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def list_files(directory):
    """List all files in a directory recursively."""
    try:
        files = []
        for root, dirs, filenames in os.walk(directory):
            # Skip hidden directories and common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env']]
            for filename in filenames:
                if not filename.startswith('.'):
                    files.append(os.path.join(root, filename))
        return files
    except Exception as e:
        return f"Error listing files: {str(e)}"

def search_files(directory, search_term):
    """Search for a term in all files in a directory."""
    results = []
    files = list_files(directory)
    if isinstance(files, str):  # Error occurred
        return files
    
    for file_path in files:
        try:
            content = read_file(file_path)
            if isinstance(content, str) and search_term.lower() in content.lower():
                # Find line numbers where the term appears
                lines = content.split('\n')
                matching_lines = []
                for i, line in enumerate(lines, 1):
                    if search_term.lower() in line.lower():
                        matching_lines.append((i, line.strip()))
                
                results.append({
                    'file': file_path,
                    'matches': matching_lines[:5]  # Limit to first 5 matches per file
                })
        except:
            continue
    
    return results

def analyze_code_structure(file_path):
    """Analyze the structure of a Python file."""
    try:
        content = read_file(file_path)
        if isinstance(content, str) and file_path.endswith('.py'):
            tree = ast.parse(content)
            
            structure = {
                'classes': [],
                'functions': [],
                'imports': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    structure['classes'].append({
                        'name': node.name,
                        'methods': methods,
                        'line': node.lineno
                    })
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                    structure['functions'].append({
                        'name': node.name,
                        'line': node.lineno
                    })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        structure['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    structure['imports'].append(f"{node.module}.{', '.join([alias.name for alias in node.names])}")
            
            return structure
        else:
            return f"Not a Python file or error reading: {file_path}"
    except Exception as e:
        return f"Error analyzing code structure: {str(e)}"

def run_command(command):
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds"
    except Exception as e:
        return f"Error running command: {str(e)}"

def find_project_dependencies(directory):
    """Find project dependencies based on common dependency files."""
    dependencies = {}
    
    # Python
    req_file = os.path.join(directory, 'requirements.txt')
    if os.path.exists(req_file):
        content = read_file(req_file)
        if isinstance(content, str):
            dependencies['python'] = content.strip().split('\n')
    
    # Node.js
    package_file = os.path.join(directory, 'package.json')
    if os.path.exists(package_file):
        content = read_file(package_file)
        if isinstance(content, str):
            try:
                package_data = json.loads(content)
                dependencies['nodejs'] = {
                    'dependencies': list(package_data.get('dependencies', {}).keys()),
                    'devDependencies': list(package_data.get('devDependencies', {}).keys())
                }
            except:
                pass
    
    # Add other dependency files as needed (Gemfile, go.mod, etc.)
    
    return dependencies

def get_directory_tree(directory, max_depth=3):
    """Get a tree representation of directory structure."""
    tree = []
    
    def add_to_tree(path, prefix="", depth=0):
        if depth > max_depth:
            return
        
        try:
            items = sorted(os.listdir(path))
            dirs = [item for item in items if os.path.isdir(os.path.join(path, item)) and not item.startswith('.')]
            files = [item for item in items if os.path.isfile(os.path.join(path, item)) and not item.startswith('.')]
            
            # Add directories first
            for i, dir_name in enumerate(dirs):
                if dir_name in ['node_modules', '__pycache__', 'venv', '.git']:
                    continue
                is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                tree.append(f"{prefix}{'└── ' if is_last_dir else '├── '}{dir_name}/")
                add_to_tree(os.path.join(path, dir_name), prefix + ("    " if is_last_dir else "│   "), depth + 1)
            
            # Then add files
            for i, file_name in enumerate(files):
                is_last = i == len(files) - 1
                tree.append(f"{prefix}{'└── ' if is_last else '├── '}{file_name}")
        except PermissionError:
            pass
    
    tree.append(f"{os.path.basename(directory)}/")
    add_to_tree(directory)
    return '\n'.join(tree)

# Main Tech Writer Module using ReAct
class TechWriterDSPy(dspy.Module):
    def __init__(self):
        super().__init__()
        
        # ReAct module for article writing with tools
        self.writer = dspy.ReAct(
            WriteTechArticle,
            tools=[
                read_file,
                write_file,
                list_files,
                search_files,
                analyze_code_structure,
                run_command,
                find_project_dependencies,
                get_directory_tree
            ],
            max_iters=15
        )
    
    def forward(self, brief: str, repo_path: str = "."):
        """Generate a technical article based on the brief and repository."""
        
        # Change to repo directory for tool context
        original_dir = os.getcwd()
        try:
            os.chdir(repo_path)
            
            # Generate article using ReAct with all the tools
            result = self.writer(brief=brief)
            
            return result.article
            
        finally:
            # Always change back to original directory
            os.chdir(original_dir)

# Simple usage
if __name__ == "__main__":
    writer = TechWriterDSPy()
    
    brief = """
    Write a comprehensive technical guide about this codebase.
    Include:
    - Overview of what it does
    - Architecture and key components
    - How to set it up and use it
    - Code examples from the actual codebase
    - API documentation if relevant
    """
    
    article = writer(brief=brief, repo_path="./my-project")
    
    print(article)