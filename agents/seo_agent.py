"""
SEO Optimizer Agent for the Multi-Agent Blog Writing System.
Optimizes content for search engines while maintaining readability.
"""

from typing import Optional, Dict, Any
import re

from agents.base_agent import BaseAgent
from config import SEO_OPTIMIZER_SYSTEM_PROMPT, Config
from models import AgentRole, BlogDraft, SEOAnalysis


class SEOAgent(BaseAgent):
    """
    Expert SEO optimization agent for blog content.
    
    Responsibilities:
    - Identify and optimize keywords
    - Improve title and meta description
    - Enhance heading structure
    - Suggest content improvements for SEO
    """
    
    def __init__(self):
        super().__init__(
            role=AgentRole.SEO_OPTIMIZER,
            system_prompt=SEO_OPTIMIZER_SYSTEM_PROMPT,
            temperature=0.5,
            max_tokens=3000
        )
    
    def process(
        self,
        input_data: BlogDraft,
        iteration: int = 1,
        **kwargs
    ) -> SEOAnalysis:
        """
        Analyze and optimize a blog draft for SEO.
        
        Args:
            input_data: The blog draft to optimize
            iteration: Current iteration number (1-3)
        
        Returns:
            SEOAnalysis with optimization suggestions
        """
        prompt = self._build_seo_prompt(input_data, iteration)
        response = self.generate(prompt)
        return self._parse_seo_response(response, input_data.topic)
    
    def _build_seo_prompt(self, draft: BlogDraft, iteration: int) -> str:
        """Build the SEO analysis prompt."""
        
        return f"""Analyze and optimize the following blog post for SEO (Iteration {iteration}/3).

**Topic:** {draft.topic}
**Current Title:** {draft.title}

---

**BLOG CONTENT:**

{draft.raw_content[:15000]}  # Limit content to avoid token issues

---

**SEO OPTIMIZATION REQUIREMENTS:**

Provide a comprehensive SEO analysis including:

1. **SEO Score (1-10)**: Current optimization level

2. **Primary Keywords** (3-5):
   - Main search terms this article should rank for
   - High-volume, relevant keywords

3. **Secondary Keywords** (5-10):
   - Supporting and long-tail keywords
   - Related search terms

4. **Optimized Title**:
   - Include primary keyword
   - Use power words
   - Keep under 60 characters
   - Make it compelling for clicks

5. **Meta Description**:
   - 150-160 characters
   - Include primary keyword naturally
   - Create urgency or curiosity
   - Include a call to action

6. **Heading Structure Analysis**:
   - Are headings optimized for keywords?
   - Suggestions for H2/H3 improvements
   - Keyword placement in headings

7. **Content Optimization Suggestions**:
   - Keyword density issues
   - Places to add keywords naturally
   - Internal linking opportunities
   - Featured snippet optimization

8. **Technical SEO Notes**:
   - URL slug suggestion
   - Image alt text suggestions
   - Schema markup recommendations

Be specific with suggestions. Focus on changes that will have the most impact on search rankings while maintaining natural readability.

{"Focus on foundational keyword research and structure." if iteration == 1 else "Focus on refinement and advanced optimization techniques."}"""
    
    def _parse_seo_response(self, response: str, topic: str) -> SEOAnalysis:
        """Parse the SEO analysis response."""
        
        # Extract SEO score
        seo_score = 5
        score_match = re.search(r'(?:seo\s+)?score[:\s]*(\d+)(?:\s*/\s*10)?', response.lower())
        if score_match:
            seo_score = min(10, max(1, int(score_match.group(1))))
        
        # Extract primary keywords
        primary_keywords = self._extract_keywords(response, 'primary')
        if not primary_keywords:
            # Generate from topic
            primary_keywords = [word.lower() for word in topic.split()[:5] if len(word) > 3]
        
        # Extract secondary keywords
        secondary_keywords = self._extract_keywords(response, 'secondary|supporting|long-tail')
        
        # Extract optimized title
        optimized_title = self._extract_single_item(response, 'optimized title|suggested title|new title')
        if not optimized_title:
            optimized_title = f"Complete Guide to {topic.title()}: Best Practices & Examples"
        
        # Extract meta description
        meta_description = self._extract_single_item(response, 'meta description')
        if not meta_description:
            meta_description = f"Learn everything about {topic.lower()}. This comprehensive guide covers best practices, code examples, and expert tips for modern developers."
        # Ensure meta description is proper length
        if len(meta_description) > 160:
            meta_description = meta_description[:157] + "..."
        
        # Extract heading suggestions
        heading_suggestions = self._extract_list_items(response, 'heading|h2|h3')
        
        # Extract content suggestions
        content_suggestions = self._extract_list_items(response, 'content|optimization|suggestion|recommend')
        
        return SEOAnalysis(
            seo_score=seo_score,
            primary_keywords=primary_keywords[:5],
            secondary_keywords=secondary_keywords[:10],
            optimized_title=optimized_title,
            meta_description=meta_description,
            heading_suggestions=heading_suggestions[:5],
            content_suggestions=content_suggestions[:8],
            detailed_analysis=response
        )
    
    def _extract_keywords(self, text: str, section_pattern: str) -> list:
        """Extract keywords from a specific section."""
        keywords = []
        
        # Look for keyword sections
        pattern = rf'(?:{section_pattern})\s*keywords?[:\s]*([^\n]+(?:\n[-*•]\s*[^\n]+)*)'
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            section = match.group(1)
            # Extract individual keywords
            # Handle comma-separated
            if ',' in section:
                keywords.extend([k.strip().strip('"\'') for k in section.split(',') if k.strip()])
            # Handle bullet points
            bullet_pattern = r'[-*•]\s*([^\n]+)'
            for bullet_match in re.finditer(bullet_pattern, section):
                kw = bullet_match.group(1).strip().strip('"\'')
                # Clean up keyword
                kw = re.sub(r'\s*\(.*?\)\s*', '', kw)  # Remove parenthetical
                if kw and len(kw) < 50:
                    keywords.append(kw.lower())
        
        return list(set(keywords))
    
    def _extract_single_item(self, text: str, pattern: str) -> str:
        """Extract a single item (like title or meta description)."""
        regex = rf'(?:{pattern})[:\s]*["\']?([^\n"\']+)["\']?'
        match = re.search(regex, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""
    
    def _extract_list_items(self, text: str, section_pattern: str) -> list:
        """Extract list items from a section."""
        items = []
        
        pattern = rf'(?:{section_pattern})[^\n]*\n((?:[-*•]\s*[^\n]+\n?)+)'
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        
        for match in matches:
            section_text = match.group(1)
            item_pattern = r'[-*•]\s*([^\n]+)'
            for item_match in re.finditer(item_pattern, section_text):
                item = item_match.group(1).strip()
                if item and len(item) > 10:
                    items.append(item)
        
        return list(set(items))
