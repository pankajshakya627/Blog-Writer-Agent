"""
Medium-compatible Markdown formatter for the Blog Writing System.
Updated for LangGraph state-based workflow.
"""

import re
from typing import Optional, Dict, Any
from datetime import datetime


class MediumFormatter:
    """
    Formats blog content for Medium compatibility.
    """
    
    @staticmethod
    def format_blog(state: Dict[str, Any], include_meta: bool = True) -> str:
        """
        Format the blog state for Medium.
        
        Args:
            state: The LangGraph final state
            include_meta: Include metadata section at top
        
        Returns:
            Formatted markdown string
        """
        content = state.get("draft_content", "")
        content = MediumFormatter._clean_content(content)
        
        parts = []
        
        if include_meta:
            meta = MediumFormatter._generate_meta_block(state)
            parts.append(meta)
        
        parts.append(content)
        
        footer = MediumFormatter._generate_footer(state)
        parts.append(footer)
        
        return "\n\n".join(parts)
    
    @staticmethod
    def _clean_content(content: str) -> str:
        """Clean and normalize the markdown content."""
        content = content.strip()
        content = re.sub(r'```(\w+)\n', r'```\1\n', content)
        content = re.sub(r'\n(\|)', r'\n\n\1', content)
        content = re.sub(r'(\|)\n(?!\|)', r'\1\n\n', content)
        content = re.sub(r'\n{4,}', '\n\n\n', content)
        content = re.sub(r'([^\n])\n(#{1,6}\s)', r'\1\n\n\2', content)
        return content
    
    @staticmethod
    def _generate_meta_block(state: Dict[str, Any]) -> str:
        """Generate metadata block for reference."""
        seo = state.get("seo_analysis", {}) or {}
        
        meta_block = f"""<!--
BLOG METADATA (Remove before publishing)
========================================
Title: {state.get('title', '')}
Meta Description: {seo.get('meta_description', '')}
Primary Keywords: {', '.join(seo.get('primary_keywords', []))}
Word Count: {state.get('word_count', 0):,}
Code Examples: {state.get('code_block_count', 0)}
Tables: {state.get('table_count', 0)}
Review Score: {state.get('final_review_score', 0)}/10
SEO Score: {state.get('final_seo_score', 0)}/10
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
-->"""
        
        return meta_block
    
    @staticmethod
    def _generate_footer(state: Dict[str, Any]) -> str:
        """Generate a footer with article info."""
        seo = state.get("seo_analysis", {}) or {}
        word_count = state.get("word_count", 0)
        
        footer = f"""---

*📖 Reading time: ~{word_count // 200} minutes | 📝 {word_count:,} words*

**Tags:** {', '.join(seo.get('primary_keywords', [state.get('topic', '')])[:5])}

---

*This article was generated with AI assistance and reviewed for quality and accuracy.*"""
        
        return footer
    
    @staticmethod
    def export_to_file(state: Dict[str, Any], output_path: str, include_meta: bool = True) -> str:
        """Export the formatted blog to a file."""
        formatted = MediumFormatter.format_blog(state, include_meta=include_meta)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted)
        
        return output_path


class BlogExporter:
    """Additional export formats for the blog."""
    
    @staticmethod
    def to_html(state: Dict[str, Any]) -> str:
        """Convert the blog to basic HTML."""
        import html as html_lib
        
        content = state.get("draft_content", "")
        title = state.get("title", "Blog Post")
        seo = state.get("seo_analysis", {}) or {}
        
        # Basic markdown to HTML conversion
        content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
        content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
        content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)
        
        content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
        content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
        
        def replace_code_block(match):
            lang = match.group(1) or ''
            code = html_lib.escape(match.group(2))
            return f'<pre><code class="language-{lang}">{code}</code></pre>'
        
        content = re.sub(r'```(\w*)\n(.*?)```', replace_code_block, content, flags=re.DOTALL)
        content = re.sub(r'`([^`]+)`', r'<code>\1</code>', content)
        
        paragraphs = content.split('\n\n')
        content = '\n'.join(
            f'<p>{p}</p>' if not p.startswith('<') else p
            for p in paragraphs if p.strip()
        )
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="{seo.get('meta_description', '')}">
    <title>{title}</title>
    <style>
        body {{ font-family: 'Georgia', serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.8; }}
        h1, h2, h3 {{ font-family: 'Helvetica', sans-serif; }}
        pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #f8f8f8; }}
    </style>
</head>
<body>
{content}
</body>
</html>"""
    
    @staticmethod
    def to_json(state: Dict[str, Any]) -> str:
        """Export blog metadata and content as JSON."""
        import json
        
        seo = state.get("seo_analysis", {}) or {}
        
        return json.dumps({
            "title": state.get("title", ""),
            "topic": state.get("topic", ""),
            "content": state.get("draft_content", ""),
            "metadata": {
                "word_count": state.get("word_count", 0),
                "code_blocks": state.get("code_block_count", 0),
                "tables": state.get("table_count", 0),
                "primary_keywords": seo.get("primary_keywords", []),
                "meta_description": seo.get("meta_description", ""),
                "review_score": state.get("final_review_score", 0),
                "seo_score": state.get("final_seo_score", 0),
            },
            "generation": {
                "iterations": state.get("max_iterations", 3),
                "time_seconds": state.get("generation_time", 0)
            }
        }, indent=2)
