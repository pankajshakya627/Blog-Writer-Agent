# Multi-Agent Blog Writing System 📝

A powerful multi-agent system built with **LangGraph** that generates industry-grade, Medium-compatible blog posts. The system employs three specialized AI agents working in iterative refinement loops to produce comprehensive, SEO-optimized technical content.

## ✨ Features

- **LangGraph Architecture**: Graph-based agent orchestration for reliable, stateful workflows
- **Multi-Agent System**: Three specialized AI agents working together

  - 🖊️ **Writer Agent**: Generates comprehensive technical content with code examples and tables
  - 🔍 **Reviewer Agent**: Provides quality assessment and improvement suggestions
  - 📈 **SEO Optimizer Agent**: Optimizes for search engines and readability

- **Iterative Refinement**: Content goes through 3 rounds of improvement
- **Medium-Compatible Output**: Markdown formatted for direct use in Medium
- **Rich Content Generation**:
  - Python code examples with syntax highlighting
  - Data tables and comparisons
  - Industry statistics and case studies
  - 5,000-8,000+ word articles

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  START ─► Writer ─► Reviewer ─► SEO ─► Controller ─┐        │
│              ▲                                      │        │
│              └──────────────────────────────────────┘        │
│                     (3 Iterations)                           │
│                            │                                 │
│                            ▼                                 │
│                          END                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /Volumes/Crucial_X9/Medium_articles/Blog_writter
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set API Key

The system uses OpenAI via LangChain. Set your API key in `.env`:

```bash
OPENAI_API_KEY=your-api-key-here
```

Or export it:

```bash
export OPENAI_API_KEY="your-api-key"
```

### 3. Generate a Blog

```bash
python main.py --topic "Your Topic Here"
```

## 📖 Usage Examples

### Basic Usage

```bash
python main.py --topic "Machine Learning Best Practices for Production Systems"
```

### With Custom Iterations

```bash
python main.py --topic "Python Data Structures" --iterations 5
```

### Using Different Model

```bash
python main.py --topic "API Design Patterns" --model gpt-4o-mini
```

### Export Multiple Formats

```bash
python main.py --topic "Cloud Architecture" --html --json
```

## 🛠️ Command Line Options

| Option         | Short | Description                     | Default        |
| -------------- | ----- | ------------------------------- | -------------- |
| `--topic`      | `-t`  | Blog topic (required)           | -              |
| `--iterations` | `-i`  | Number of refinement iterations | 3              |
| `--model`      | `-m`  | OpenAI model to use             | gpt-4o         |
| `--output`     | `-o`  | Output filename                 | Auto-generated |
| `--output-dir` | -     | Output directory                | output         |
| `--html`       | -     | Also export as HTML             | False          |
| `--json`       | -     | Export metadata as JSON         | False          |
| `--quiet`      | `-q`  | Suppress progress output        | False          |

## 📁 Project Structure

```
Blog_writter/
├── main.py                    # CLI entry point
├── graph.py                   # LangGraph workflow definition
├── config.py                  # Configuration & prompts
├── models.py                  # State and data models
├── formatters/                # Output formatting
│   ├── __init__.py
│   └── medium_formatter.py
├── agents/                    # Legacy agents (for reference)
├── output/                    # Generated blogs
├── requirements.txt           # Dependencies
└── venv/                      # Python virtual environment
```

## 🔄 LangGraph Workflow

The system uses LangGraph's `StateGraph` to manage the multi-agent workflow:

```python
# Graph Structure
graph = StateGraph(BlogState)

# Nodes (Agents)
graph.add_node("writer", writer_node)
graph.add_node("reviewer", reviewer_node)
graph.add_node("seo", seo_node)
graph.add_node("controller", iteration_controller)

# Flow
START -> writer -> reviewer -> seo -> controller -> (writer | END)
```

### State Management

```python
class BlogState(TypedDict):
    topic: str
    current_iteration: int
    max_iterations: int
    draft_content: str
    title: str
    review_feedback: Optional[Dict]
    seo_analysis: Optional[Dict]
    word_count: int
    code_block_count: int
    table_count: int
    messages: List[Dict]
    is_complete: bool
```

## 📊 Output Quality

Generated blogs include:

- **Comprehensive Content**: 5,000-8,000+ words of detailed technical content
- **Code Examples**: 3-5 Python code snippets with explanations
- **Data Tables**: 2-4 comparison tables and data summaries
- **SEO Optimization**: Keywords, meta description, optimized headings
- **Professional Structure**: Clear sections, transitions, and formatting

## 🔧 Configuration

Edit `config.py` to customize:

```python
class Config:
    LLM_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 8000
    NUM_ITERATIONS: int = 3
    MIN_WORD_COUNT: int = 3000
    TARGET_WORD_COUNT: int = 6000
```

## 📄 License

MIT License - feel free to use and modify for your projects.
