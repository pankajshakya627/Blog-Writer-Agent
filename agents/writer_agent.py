"""
Writer Agent for the Multi-Agent Blog Writing System.
Generates comprehensive, industry-grade blog content.
"""

from typing import Optional, Dict, Any

from agents.base_agent import BaseAgent
from config import WRITER_SYSTEM_PROMPT, Config
from models import AgentRole, BlogDraft, BlogMetadata, ReviewFeedback, SEOAnalysis


class WriterAgent(BaseAgent):
    """
    Expert blog writer agent that generates comprehensive technical content.
    
    Responsibilities:
    - Generate long-form blog posts (5000-8000+ words)
    - Include Python code examples with explanations
    - Create data tables for comparisons
    - Structure content with clear sections
    """
    
    def __init__(self):
        super().__init__(
            role=AgentRole.WRITER,
            system_prompt=WRITER_SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=Config.LLM_MAX_TOKENS
        )
    
    def process(
        self,
        input_data: str,
        iteration: int = 1,
        previous_draft: Optional[BlogDraft] = None,
        review_feedback: Optional[ReviewFeedback] = None,
        seo_analysis: Optional[SEOAnalysis] = None,
        **kwargs
    ) -> BlogDraft:
        """
        Generate or improve a blog draft.
        
        Args:
            input_data: The blog topic or prompt
            iteration: Current iteration number (1-3)
            previous_draft: Previous version of the draft (for iterations 2+)
            review_feedback: Feedback from the reviewer agent
            seo_analysis: SEO suggestions from the optimizer agent
        
        Returns:
            BlogDraft with the generated content
        """
        if iteration == 1:
            return self._generate_initial_draft(input_data)
        else:
            return self._refine_draft(
                input_data,
                iteration,
                previous_draft,
                review_feedback,
                seo_analysis
            )
    
    def _generate_initial_draft(self, topic: str) -> BlogDraft:
        """Generate the initial comprehensive blog draft."""
        
        prompt = f"""Write a comprehensive, industry-grade blog post about:

**Topic:** {topic}

Requirements:
1. **Length**: Write a detailed post of 5000-8000 words minimum
2. **Structure**: Include at least 6-8 major sections with subsections
3. **Code Examples**: Include 4-5 practical Python code examples with detailed explanations
4. **Tables**: Include 3-4 data tables (comparisons, feature matrices, statistics)
5. **Real Data**: Include industry statistics, research findings, and real-world examples
6. **Depth**: Go deep into technical concepts, best practices, and practical applications

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

        response = self.generate(prompt)
        
        return BlogDraft(
            topic=topic,
            title=self._extract_title(response, topic),
            raw_content=response,
            metadata=BlogMetadata(
                title=self._extract_title(response, topic),
                word_count=len(response.split())
            )
        )
    
    def _refine_draft(
        self,
        topic: str,
        iteration: int,
        previous_draft: BlogDraft,
        review_feedback: Optional[ReviewFeedback],
        seo_analysis: Optional[SEOAnalysis]
    ) -> BlogDraft:
        """Refine the draft based on feedback from other agents."""
        
        feedback_section = ""
        if review_feedback:
            feedback_section += f"""
## Reviewer Feedback (Score: {review_feedback.overall_score}/10)

### Strengths:
{chr(10).join(f'- {s}' for s in review_feedback.strengths)}

### Areas to Improve:
{chr(10).join(f'- {i}' for i in review_feedback.improvements)}

### Priority Fixes:
{chr(10).join(f'- {f}' for f in review_feedback.priority_fixes)}

### Suggested Additions:
{chr(10).join(f'- {a}' for a in review_feedback.suggested_additions)}

### Detailed Feedback:
{review_feedback.detailed_feedback}
"""

        if seo_analysis:
            feedback_section += f"""
## SEO Optimization Suggestions (Score: {seo_analysis.seo_score}/10)

### Primary Keywords: {', '.join(seo_analysis.primary_keywords)}
### Secondary Keywords: {', '.join(seo_analysis.secondary_keywords)}

### Suggested Title: {seo_analysis.optimized_title}
### Meta Description: {seo_analysis.meta_description}

### Heading Improvements:
{chr(10).join(f'- {h}' for h in seo_analysis.heading_suggestions)}

### Content Suggestions:
{chr(10).join(f'- {c}' for c in seo_analysis.content_suggestions)}
"""

        prompt = f"""You are refining a blog post in iteration {iteration} of 3.

**Original Topic:** {topic}

**Current Draft:**
{previous_draft.raw_content}

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

        response = self.generate(prompt)
        
        return BlogDraft(
            topic=topic,
            title=self._extract_title(response, topic),
            raw_content=response,
            metadata=BlogMetadata(
                title=self._extract_title(response, topic),
                word_count=len(response.split()),
                primary_keywords=seo_analysis.primary_keywords if seo_analysis else [],
                secondary_keywords=seo_analysis.secondary_keywords if seo_analysis else [],
                meta_description=seo_analysis.meta_description if seo_analysis else ""
            )
        )
    
    def _extract_title(self, content: str, fallback: str) -> str:
        """Extract the title from the markdown content."""
        lines = content.strip().split('\n')
        for line in lines:
            if line.startswith('# '):
                return line[2:].strip()
        return fallback.title()
