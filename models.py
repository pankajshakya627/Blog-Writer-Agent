"""
Data models for the Multi-Agent Blog Writing System with LangGraph.
Uses Pydantic for validation and TypedDict for LangGraph state.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Annotated
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import operator


class AgentRole(str, Enum):
    """Roles for different agents in the system."""
    RESEARCHER = "researcher"
    WRITER = "writer"
    REVIEWER = "reviewer"
    SEO_OPTIMIZER = "seo_optimizer"
    ORCHESTRATOR = "orchestrator"


class BlogMetadata(BaseModel):
    """SEO and metadata for the blog post."""
    title: str = Field(default="", description="Blog post title")
    meta_description: str = Field(default="", description="Meta description for SEO")
    primary_keywords: List[str] = Field(default_factory=list, description="Primary SEO keywords")
    secondary_keywords: List[str] = Field(default_factory=list, description="Secondary keywords")
    estimated_read_time: int = Field(default=0, description="Estimated reading time in minutes")
    word_count: int = Field(default=0, description="Total word count")
    tags: List[str] = Field(default_factory=list, description="Blog tags")


class ReviewFeedback(BaseModel):
    """Feedback from the reviewer agent."""
    overall_score: int = Field(default=5, ge=1, le=10, description="Overall quality score")
    strengths: List[str] = Field(default_factory=list, description="What works well")
    improvements: List[str] = Field(default_factory=list, description="Areas to improve")
    priority_fixes: List[str] = Field(default_factory=list, description="Most important changes")
    suggested_additions: List[str] = Field(default_factory=list, description="Content to add")
    detailed_feedback: str = Field(default="", description="Full review text")


class SEOAnalysis(BaseModel):
    """SEO analysis and optimization suggestions."""
    seo_score: int = Field(default=5, ge=1, le=10, description="SEO optimization score")
    primary_keywords: List[str] = Field(default_factory=list, description="Main keywords")
    secondary_keywords: List[str] = Field(default_factory=list, description="Supporting keywords")
    optimized_title: str = Field(default="", description="SEO-optimized title")
    meta_description: str = Field(default="", description="Optimized meta description")
    heading_suggestions: List[str] = Field(default_factory=list, description="Heading improvements")
    content_suggestions: List[str] = Field(default_factory=list, description="Content optimizations")
    detailed_analysis: str = Field(default="", description="Full SEO analysis text")


class AgentMessage(BaseModel):
    """Message exchanged between agents."""
    role: AgentRole = Field(..., description="Agent sending the message")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")


# ============================================================================
# LangGraph State Definition
# ============================================================================

class BlogState(TypedDict):
    """
    State for the LangGraph blog generation workflow.
    This is the central state passed between all agents.
    """
    # Input
    topic: str
    
    # Current iteration tracking
    current_iteration: int
    max_iterations: int
    
    # Blog content
    draft_content: str
    title: str
    
    # Research context (from Researcher Agent)
    research_context: Optional[Dict[str, Any]]
    
    # Agent outputs
    review_feedback: Optional[Dict[str, Any]]
    seo_analysis: Optional[Dict[str, Any]]
    
    # Metadata
    word_count: int
    code_block_count: int
    table_count: int
    
    # Message history (using Annotated for reducer)
    messages: Annotated[List[Dict[str, Any]], operator.add]
    
    # Final output indicators
    is_complete: bool
    final_review_score: int
    final_seo_score: int


def create_initial_state(topic: str, max_iterations: int = 3) -> BlogState:
    """Create the initial state for blog generation."""
    return BlogState(
        topic=topic,
        current_iteration=1,
        max_iterations=max_iterations,
        draft_content="",
        title="",
        research_context=None,
        review_feedback=None,
        seo_analysis=None,
        word_count=0,
        code_block_count=0,
        table_count=0,
        messages=[],
        is_complete=False,
        final_review_score=0,
        final_seo_score=0
    )


# ============================================================================
# Utility Functions
# ============================================================================

def count_words(content: str) -> int:
    """Count words in content."""
    return len(content.split())


def count_code_blocks(content: str) -> int:
    """Count Python code blocks in content."""
    return content.count("```python") + content.count("```py")


def count_tables(content: str) -> int:
    """Count tables in markdown content."""
    lines = content.split("\n")
    table_lines = [l for l in lines if l.strip().startswith("|") and l.strip().endswith("|")]
    return max(0, len(table_lines) // 3)


def extract_title(content: str, fallback: str) -> str:
    """Extract title from markdown content."""
    lines = content.strip().split('\n')
    for line in lines:
        if line.startswith('# '):
            return line[2:].strip()
    return fallback.title()
