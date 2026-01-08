"""
LangGraph-based Blog Generation Graph.
Implements the multi-agent workflow using LangGraph StateGraph.
"""

import os
import re
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from models import (
    BlogState, create_initial_state,
    count_words, count_code_blocks, count_tables, extract_title
)
from config import (
    Config, WRITER_SYSTEM_PROMPT, REVIEWER_SYSTEM_PROMPT, 
    SEO_OPTIMIZER_SYSTEM_PROMPT, RESEARCHER_SYSTEM_PROMPT,
    RESEARCH_MAX_RESULTS
)


console = Console()


def get_llm() -> ChatOpenAI:
    """Get the configured LLM instance."""
    return ChatOpenAI(
        model=Config.LLM_MODEL,
        temperature=Config.LLM_TEMPERATURE,
        max_tokens=Config.LLM_MAX_TOKENS,
    )


# ============================================================================
# Web Search Functions
# ============================================================================

def search_with_tavily(query: str, max_results: int = 5) -> list:
    """Search using Tavily API for AI-optimized results."""
    try:
        from tavily import TavilyClient
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return []
        
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            include_answer=True,
            search_depth="advanced"
        )
        
        results = []
        if response.get("answer"):
            results.append({
                "source": "Tavily AI Summary",
                "content": response["answer"],
                "url": ""
            })
        
        for item in response.get("results", []):
            results.append({
                "source": item.get("title", "Unknown"),
                "content": item.get("content", ""),
                "url": item.get("url", "")
            })
        
        return results
    except Exception as e:
        console.print(f"  [dim]Tavily search failed: {e}[/dim]")
        return []


def search_with_duckduckgo(query: str, max_results: int = 5) -> list:
    """Fallback search using DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS
        
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "source": r.get("title", "Unknown"),
                    "content": r.get("body", ""),
                    "url": r.get("href", "")
                })
        
        return results
    except Exception as e:
        console.print(f"  [dim]DuckDuckGo search failed: {e}[/dim]")
        return []


def perform_research(topic: str, max_results: int = 10) -> dict:
    """
    Perform comprehensive web research on a topic.
    Uses Tavily if API key available, falls back to DuckDuckGo.
    """
    all_results = []
    
    # Generate multiple search queries
    queries = [
        f"{topic} comprehensive guide",
        f"{topic} best practices 2024 2025",
        f"{topic} research paper academic",
        f"{topic} Python implementation example",
        f"{topic} industry statistics data"
    ]
    
    # Check if Tavily is available
    use_tavily = bool(os.getenv("TAVILY_API_KEY"))
    search_func = search_with_tavily if use_tavily else search_with_duckduckgo
    search_source = "Tavily" if use_tavily else "DuckDuckGo"
    
    console.print(f"  [dim]Using {search_source} for research...[/dim]")
    
    for query in queries:
        results = search_func(query, max_results=max_results // len(queries))
        all_results.extend(results)
    
    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        url = r.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)
    
    return {
        "sources": unique_results[:max_results],
        "search_engine": search_source,
        "queries_used": queries,
        "total_results": len(unique_results)
    }


# ============================================================================
# Agent Node Functions
# ============================================================================

def researcher_node(state: BlogState) -> dict:
    """
    Research Agent Node - Searches web for documentation, papers, and statistics.
    Runs before the Writer to provide factual context.
    """
    console.print(f"\n[cyan]🔬 Research Agent[/cyan] - Gathering information from the web...")
    
    topic = state["topic"]
    iteration = state["current_iteration"]
    
    # Only do full research on first iteration
    if iteration > 1 and state.get("research_context"):
        console.print(f"  [dim]Using cached research from iteration 1[/dim]")
        return {"messages": [{"role": "researcher", "iteration": iteration, "status": "cached"}]}
    
    # Perform web research
    research_data = perform_research(topic, max_results=RESEARCH_MAX_RESULTS)
    
    # Use LLM to synthesize research into structured context
    llm = get_llm()
    
    sources_text = "\n\n".join([
        f"**Source:** {r['source']}\n**URL:** {r['url']}\n**Content:** {r['content'][:500]}..."
        for r in research_data.get("sources", [])[:8]
    ])
    
    synthesis_prompt = f"""Based on the following research results about "{topic}", extract and organize:

{sources_text}

---

Provide a structured research summary in this EXACT format:

KEY_FACTS:
- [fact 1 with source]
- [fact 2 with source]
- [fact 3 with source]

STATISTICS:
- [statistic 1 with year and source]
- [statistic 2 with year and source]

BEST_PRACTICES:
- [practice 1]
- [practice 2]
- [practice 3]

RECOMMENDED_TOPICS:
- [subtopic to cover]
- [subtopic to cover]

KEY_SOURCES:
- [source name]: [url]
- [source name]: [url]

Focus on factual, verifiable information from authoritative sources."""
    
    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(content=synthesis_prompt)
    ]
    
    response = llm.invoke(messages)
    synthesis = response.content
    
    # Parse the synthesis
    def extract_list(pattern: str) -> list:
        match = re.search(rf'{pattern}:\s*((?:[-•]\s*[^\n]+\n?)+)', synthesis, re.IGNORECASE)
        if match:
            items = re.findall(r'[-•]\s*([^\n]+)', match.group(1))
            return [item.strip() for item in items if item.strip()]
        return []
    
    research_context = {
        "raw_sources": research_data.get("sources", []),
        "search_engine": research_data.get("search_engine", ""),
        "key_facts": extract_list("KEY_FACTS"),
        "statistics": extract_list("STATISTICS"),
        "best_practices": extract_list("BEST_PRACTICES"),
        "recommended_topics": extract_list("RECOMMENDED_TOPICS"),
        "key_sources": extract_list("KEY_SOURCES"),
        "synthesis": synthesis
    }
    
    console.print(f"  [dim]Found {len(research_context['raw_sources'])} sources, {len(research_context['key_facts'])} key facts[/dim]")
    
    return {
        "research_context": research_context,
        "messages": [{
            "role": "researcher",
            "iteration": iteration,
            "sources_found": len(research_context["raw_sources"]),
            "facts_extracted": len(research_context["key_facts"])
        }]
    }


def writer_node(state: BlogState) -> dict:
    """
    Writer Agent Node - Generates or refines blog content.
    """
    console.print(f"\n[green]📝 Writer Agent[/green] - Iteration {state['current_iteration']}/{state['max_iterations']}")
    
    llm = get_llm()
    iteration = state["current_iteration"]
    topic = state["topic"]
    
    if iteration == 1:
        # Build research context section
        research_section = ""
        if state.get("research_context"):
            rc = state["research_context"]
            research_section = f"""
## Research Context (Use these facts and cite sources)

### Key Facts:
{chr(10).join('- ' + f for f in rc.get('key_facts', []))}

### Industry Statistics:
{chr(10).join('- ' + s for s in rc.get('statistics', []))}

### Best Practices to Cover:
{chr(10).join('- ' + p for p in rc.get('best_practices', []))}

### Key Sources to Reference:
{chr(10).join('- ' + s for s in rc.get('key_sources', []))}
"""
        
        # Initial draft generation
        prompt = f"""Write a comprehensive, industry-grade blog post about:

**Topic:** {topic}
{research_section}

Requirements:
1. **Length**: Write a detailed post of 5000-8000 words minimum
2. **Structure**: Include at least 6-8 major sections with subsections
3. **Code Examples**: Include 4-5 practical Python code examples with detailed explanations
4. **Tables**: Include 3-4 data tables (comparisons, feature matrices, statistics)
5. **Real Data**: Include industry statistics, research findings, and real-world examples
6. **Depth**: Go deep into technical concepts, best practices, and practical applications
7. **Citations**: Reference the sources provided in Research Context where appropriate

Content Structure:
- Engaging introduction with a hook and clear thesis
- Multiple main sections covering different aspects
- Practical examples and case studies
- Common pitfalls and how to avoid them
- Best practices and industry standards
- Actionable takeaways
- Strong conclusion with key points

Format the entire blog in clean Markdown suitable for Medium.
Include proper code blocks with ```python syntax highlighting.
Use tables with | for structured data comparisons.

Begin writing the complete blog post now:"""
    else:
        # Refinement based on feedback
        feedback_section = ""
        
        if state.get("review_feedback"):
            rf = state["review_feedback"]
            feedback_section += f"""
## Reviewer Feedback (Score: {rf.get('overall_score', 'N/A')}/10)

### Strengths:
{chr(10).join('- ' + s for s in rf.get('strengths', []))}

### Areas to Improve:
{chr(10).join('- ' + i for i in rf.get('improvements', []))}

### Priority Fixes:
{chr(10).join('- ' + f for f in rf.get('priority_fixes', []))}

### Detailed Feedback:
{rf.get('detailed_feedback', '')}
"""

        if state.get("seo_analysis"):
            sa = state["seo_analysis"]
            feedback_section += f"""
## SEO Optimization Suggestions (Score: {sa.get('seo_score', 'N/A')}/10)

### Primary Keywords: {', '.join(sa.get('primary_keywords', []))}
### Suggested Title: {sa.get('optimized_title', '')}
### Meta Description: {sa.get('meta_description', '')}

### Content Suggestions:
{chr(10).join('- ' + c for c in sa.get('content_suggestions', []))}
"""

        prompt = f"""You are refining a blog post in iteration {iteration} of {state['max_iterations']}.

**Original Topic:** {topic}

**Current Draft:**
{state['draft_content']}

---

**Feedback to Address:**
{feedback_section}

---

**Your Task for Iteration {iteration}:**

{"Focus on: Expanding content depth, adding more code examples, improving explanations, and addressing all reviewer concerns." if iteration == 2 else "Focus on: Final polish, SEO optimization, ensuring all elements are perfect, adding any missing components, and maximizing reader value."}

Requirements:
1. Address ALL feedback points systematically
2. Expand sections that need more depth
3. Add more code examples if below 5
4. Add more tables if below 3
5. Improve flow and transitions
6. Strengthen introduction and conclusion
7. Ensure word count is 6000+ words
8. Apply SEO suggestions naturally

Write the complete improved blog post in Markdown format:"""

    messages = [
        SystemMessage(content=WRITER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    content = response.content
    
    # Update state
    word_count = count_words(content)
    code_blocks = count_code_blocks(content)
    tables = count_tables(content)
    title = extract_title(content, topic)
    
    console.print(f"  [dim]Generated {word_count:,} words, {code_blocks} code blocks, {tables} tables[/dim]")
    
    return {
        "draft_content": content,
        "title": title,
        "word_count": word_count,
        "code_block_count": code_blocks,
        "table_count": tables,
        "messages": [{
            "role": "writer",
            "iteration": iteration,
            "word_count": word_count,
            "content_preview": content[:200] + "..."
        }]
    }


def reviewer_node(state: BlogState) -> dict:
    """
    Reviewer Agent Node - Reviews content and provides feedback.
    """
    console.print(f"[yellow]🔍 Reviewer Agent[/yellow] - Analyzing content quality...")
    
    llm = get_llm()
    
    prompt = f"""Please review the following blog post draft (Iteration {state['current_iteration']}/{state['max_iterations']}).

**Topic:** {state['topic']}
**Current Statistics:**
- Word Count: {state['word_count']} words (Target: 6000+)
- Code Blocks: {state['code_block_count']} (Target: 5+)
- Tables: {state['table_count']} (Target: 3+)

---

**BLOG CONTENT:**

{state['draft_content']}

---

**REVIEW REQUIREMENTS:**

Please provide a comprehensive review in this EXACT format:

OVERALL_SCORE: [1-10]

STRENGTHS:
- [strength 1]
- [strength 2]
- [strength 3]

IMPROVEMENTS:
- [improvement 1]
- [improvement 2]
- [improvement 3]

PRIORITY_FIXES:
- [fix 1]
- [fix 2]
- [fix 3]

DETAILED_FEEDBACK:
[Your detailed feedback here]

Be constructive and specific. Focus on improvements that will significantly enhance the blog quality."""

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    review_text = response.content
    
    # Parse the review
    import re
    
    score = 5
    score_match = re.search(r'OVERALL_SCORE:\s*(\d+)', review_text)
    if score_match:
        score = min(10, max(1, int(score_match.group(1))))
    
    def extract_list(pattern: str) -> list:
        match = re.search(rf'{pattern}:\s*((?:[-•]\s*[^\n]+\n?)+)', review_text, re.IGNORECASE)
        if match:
            items = re.findall(r'[-•]\s*([^\n]+)', match.group(1))
            return [item.strip() for item in items if item.strip()]
        return []
    
    feedback = {
        "overall_score": score,
        "strengths": extract_list("STRENGTHS") or ["Content is well-structured"],
        "improvements": extract_list("IMPROVEMENTS") or ["Consider adding more depth"],
        "priority_fixes": extract_list("PRIORITY_FIXES") or ["Expand key sections"],
        "detailed_feedback": review_text
    }
    
    console.print(f"  [dim]Review Score: {score}/10[/dim]")
    
    return {
        "review_feedback": feedback,
        "messages": [{
            "role": "reviewer",
            "iteration": state["current_iteration"],
            "score": score
        }]
    }


def seo_node(state: BlogState) -> dict:
    """
    SEO Optimizer Agent Node - Analyzes and suggests SEO improvements.
    """
    console.print(f"[magenta]📈 SEO Optimizer Agent[/magenta] - Optimizing for search engines...")
    
    llm = get_llm()
    
    prompt = f"""Analyze and optimize the following blog post for SEO (Iteration {state['current_iteration']}/{state['max_iterations']}).

**Topic:** {state['topic']}
**Current Title:** {state['title']}

---

**BLOG CONTENT:**

{state['draft_content'][:15000]}

---

**SEO OPTIMIZATION REQUIREMENTS:**

Provide your analysis in this EXACT format:

SEO_SCORE: [1-10]

PRIMARY_KEYWORDS:
- [keyword 1]
- [keyword 2]
- [keyword 3]

SECONDARY_KEYWORDS:
- [keyword 1]
- [keyword 2]
- [keyword 3]

OPTIMIZED_TITLE: [Your optimized title here]

META_DESCRIPTION: [150-160 character meta description]

CONTENT_SUGGESTIONS:
- [suggestion 1]
- [suggestion 2]
- [suggestion 3]

DETAILED_ANALYSIS:
[Your detailed SEO analysis here]

Balance SEO optimization with natural, engaging writing."""

    messages = [
        SystemMessage(content=SEO_OPTIMIZER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    seo_text = response.content
    
    # Parse the SEO analysis
    import re
    
    seo_score = 5
    score_match = re.search(r'SEO_SCORE:\s*(\d+)', seo_text)
    if score_match:
        seo_score = min(10, max(1, int(score_match.group(1))))
    
    def extract_list(pattern: str) -> list:
        match = re.search(rf'{pattern}:\s*((?:[-•]\s*[^\n]+\n?)+)', seo_text, re.IGNORECASE)
        if match:
            items = re.findall(r'[-•]\s*([^\n]+)', match.group(1))
            return [item.strip() for item in items if item.strip()]
        return []
    
    def extract_value(pattern: str) -> str:
        match = re.search(rf'{pattern}:\s*([^\n]+)', seo_text, re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    analysis = {
        "seo_score": seo_score,
        "primary_keywords": extract_list("PRIMARY_KEYWORDS") or [state["topic"].lower()],
        "secondary_keywords": extract_list("SECONDARY_KEYWORDS") or [],
        "optimized_title": extract_value("OPTIMIZED_TITLE") or state["title"],
        "meta_description": extract_value("META_DESCRIPTION") or f"Learn about {state['topic']} in this comprehensive guide.",
        "content_suggestions": extract_list("CONTENT_SUGGESTIONS") or [],
        "detailed_analysis": seo_text
    }
    
    console.print(f"  [dim]SEO Score: {seo_score}/10[/dim]")
    
    return {
        "seo_analysis": analysis,
        "messages": [{
            "role": "seo_optimizer",
            "iteration": state["current_iteration"],
            "score": seo_score
        }]
    }


def iteration_controller(state: BlogState) -> dict:
    """
    Controls the iteration flow - increments iteration counter.
    """
    new_iteration = state["current_iteration"] + 1
    
    if new_iteration > state["max_iterations"]:
        console.print(f"\n[bold green]✓ All {state['max_iterations']} iterations complete![/bold green]")
        return {
            "is_complete": True,
            "final_review_score": state["review_feedback"].get("overall_score", 0) if state["review_feedback"] else 0,
            "final_seo_score": state["seo_analysis"].get("seo_score", 0) if state["seo_analysis"] else 0,
        }
    
    console.print(f"\n[blue]━━━ Moving to Iteration {new_iteration}/{state['max_iterations']} ━━━[/blue]")
    
    return {
        "current_iteration": new_iteration
    }


# ============================================================================
# Routing Functions
# ============================================================================

def should_continue(state: BlogState) -> Literal["writer", "end"]:
    """
    Determine if we should continue iterating or finish.
    """
    if state.get("is_complete", False):
        return "end"
    return "writer"


# ============================================================================
# Graph Builder
# ============================================================================

def create_blog_graph() -> StateGraph:
    """
    Create the LangGraph workflow for blog generation.
    
    Graph Structure:
    START -> researcher -> writer -> reviewer -> seo -> controller -> (writer | END)
    """
    # Create the graph
    graph = StateGraph(BlogState)
    
    # Add nodes
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("seo", seo_node)
    graph.add_node("controller", iteration_controller)
    
    # Add edges - Researcher runs first, then Writer
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_edge("reviewer", "seo")
    graph.add_edge("seo", "controller")
    
    # Conditional edge for iteration control
    graph.add_conditional_edges(
        "controller",
        should_continue,
        {
            "writer": "researcher",  # Go back through researcher (will use cache)
            "end": END
        }
    )
    
    # Compile the graph
    return graph.compile()


# ============================================================================
# Main Orchestrator Class
# ============================================================================

class BlogOrchestrator:
    """
    LangGraph-based orchestrator for multi-agent blog generation.
    """
    
    def __init__(self, verbose: bool = True):
        self.graph = create_blog_graph()
        self.verbose = verbose
        self.console = Console()
    
    def generate_blog(
        self,
        topic: str,
        num_iterations: int = 3
    ) -> dict:
        """
        Generate a complete blog post through iterative refinement.
        
        Args:
            topic: The blog topic/prompt
            num_iterations: Number of refinement iterations (default: 3)
        
        Returns:
            Final state with the complete blog
        """
        import time
        start_time = time.time()
        
        # Display start message
        if self.verbose:
            self.console.print()
            self.console.print(Panel(
                f"[bold cyan]LangGraph Multi-Agent Blog Generation[/bold cyan]\n\n"
                f"[white]Topic:[/white] {topic}\n"
                f"[white]Iterations:[/white] {num_iterations}\n"
                f"[white]Agents:[/white] Researcher → Writer → Reviewer → SEO Optimizer",
                title="🚀 Starting",
                border_style="cyan"
            ))
        
        # Create initial state
        initial_state = create_initial_state(topic, num_iterations)
        
        # Run the graph
        self.console.print(f"\n[blue]━━━ Iteration 1/{num_iterations} ━━━[/blue]")
        final_state = self.graph.invoke(initial_state)
        
        generation_time = time.time() - start_time
        
        # Display results
        if self.verbose:
            self._display_results(final_state, generation_time)
        
        # Add generation time to state
        final_state["generation_time"] = generation_time
        
        return final_state
    
    def _display_results(self, state: dict, generation_time: float):
        """Display final generation results."""
        table = Table(title="📊 Generation Complete", border_style="green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Word Count", f"{state.get('word_count', 0):,}")
        table.add_row("Code Blocks", str(state.get('code_block_count', 0)))
        table.add_row("Tables", str(state.get('table_count', 0)))
        table.add_row("Final Review Score", f"{state.get('final_review_score', 0)}/10")
        table.add_row("Final SEO Score", f"{state.get('final_seo_score', 0)}/10")
        table.add_row("Generation Time", f"{generation_time:.1f}s")
        
        self.console.print()
        self.console.print(table)
