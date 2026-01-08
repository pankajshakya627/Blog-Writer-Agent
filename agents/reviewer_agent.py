"""
Reviewer Agent for the Multi-Agent Blog Writing System.
Provides quality assurance and improvement suggestions.
"""

from typing import Optional, Dict, Any
import re

from agents.base_agent import BaseAgent
from config import REVIEWER_SYSTEM_PROMPT, Config
from models import AgentRole, BlogDraft, ReviewFeedback


class ReviewerAgent(BaseAgent):
    """
    Expert content reviewer agent that evaluates blog quality.
    
    Responsibilities:
    - Assess technical accuracy of code and claims
    - Check readability and clarity
    - Verify structure and logical flow
    - Provide actionable improvement suggestions
    """
    
    def __init__(self):
        super().__init__(
            role=AgentRole.REVIEWER,
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            temperature=0.5,  # Lower temperature for more consistent reviews
            max_tokens=4000
        )
    
    def process(
        self,
        input_data: BlogDraft,
        iteration: int = 1,
        **kwargs
    ) -> ReviewFeedback:
        """
        Review a blog draft and provide feedback.
        
        Args:
            input_data: The blog draft to review
            iteration: Current iteration number (1-3)
        
        Returns:
            ReviewFeedback with detailed improvement suggestions
        """
        prompt = self._build_review_prompt(input_data, iteration)
        response = self.generate(prompt)
        return self._parse_review_response(response)
    
    def _build_review_prompt(self, draft: BlogDraft, iteration: int) -> str:
        """Build the review prompt for the given draft."""
        
        word_count = draft.count_words()
        code_blocks = draft.count_code_blocks()
        tables = draft.count_tables()
        
        return f"""Please review the following blog post draft (Iteration {iteration}/3).

**Topic:** {draft.topic}
**Current Statistics:**
- Word Count: {word_count} words (Target: 6000+)
- Code Blocks: {code_blocks} (Target: 5+)
- Tables: {tables} (Target: 3+)

---

**BLOG CONTENT:**

{draft.raw_content}

---

**REVIEW REQUIREMENTS:**

Please provide a comprehensive review covering:

1. **Overall Score (1-10)**: Rate the blog quality
   - Consider: completeness, accuracy, readability, value

2. **Strengths** (list 3-5 things that work well):
   - What aspects are particularly strong?

3. **Areas for Improvement** (list 5-10 specific issues):
   - Be specific about location and nature of issues
   - Prioritize by importance

4. **Priority Fixes** (top 3-5 must-fix items):
   - Most critical changes needed
   - Changes that will have the biggest impact

5. **Suggested Additions**:
   - Missing topics or sections
   - Additional code examples needed
   - Tables that would enhance the content
   - Examples or case studies to add

6. **Technical Accuracy Check**:
   - Are code examples correct?
   - Are technical claims accurate?
   - Any outdated information?

7. **Readability Assessment**:
   - Is the content accessible?
   - Are complex concepts well-explained?
   - Flow and transitions

Format your review clearly with headers. Be constructive and specific - vague feedback is not helpful.

{"Focus on foundational issues since this is the first draft." if iteration == 1 else "Focus on refinement and polish since this is iteration " + str(iteration) + "."}"""
    
    def _parse_review_response(self, response: str) -> ReviewFeedback:
        """Parse the review response into structured feedback."""
        
        # Extract overall score
        score = 5  # Default
        score_match = re.search(r'(?:overall\s+)?score[:\s]*(\d+)(?:\s*/\s*10)?', response.lower())
        if score_match:
            score = min(10, max(1, int(score_match.group(1))))
        
        # Extract strengths
        strengths = self._extract_list_items(response, 'strength')
        
        # Extract improvements
        improvements = self._extract_list_items(response, 'improvement|issue|problem|weakness')
        
        # Extract priority fixes
        priority_fixes = self._extract_list_items(response, 'priority|fix|must-fix|critical')
        
        # Extract suggested additions
        suggested_additions = self._extract_list_items(response, 'addition|suggest|add|missing')
        
        return ReviewFeedback(
            overall_score=score,
            strengths=strengths[:5] if strengths else ["Content is coherent and well-structured"],
            improvements=improvements[:10] if improvements else ["Consider adding more depth"],
            priority_fixes=priority_fixes[:5] if priority_fixes else ["Expand key sections"],
            suggested_additions=suggested_additions[:5] if suggested_additions else ["Add more examples"],
            detailed_feedback=response
        )
    
    def _extract_list_items(self, text: str, section_pattern: str) -> list:
        """Extract bullet point items from a section of the review."""
        items = []
        
        # Find the section
        pattern = rf'(?:{section_pattern})[^\n]*\n((?:[-*•]\s*[^\n]+\n?)+)'
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        
        for match in matches:
            section_text = match.group(1)
            # Extract individual items
            item_pattern = r'[-*•]\s*([^\n]+)'
            for item_match in re.finditer(item_pattern, section_text):
                item = item_match.group(1).strip()
                if item and len(item) > 10:  # Skip very short items
                    items.append(item)
        
        # Also try numbered lists
        numbered_pattern = rf'(?:{section_pattern})[^\n]*\n((?:\d+\.\s*[^\n]+\n?)+)'
        matches = re.finditer(numbered_pattern, text, re.IGNORECASE | re.MULTILINE)
        
        for match in matches:
            section_text = match.group(1)
            item_pattern = r'\d+\.\s*([^\n]+)'
            for item_match in re.finditer(item_pattern, section_text):
                item = item_match.group(1).strip()
                if item and len(item) > 10:
                    items.append(item)
        
        return list(set(items))  # Remove duplicates
