#!/usr/bin/env python3
"""
Multi-Agent Blog Writing System (LangGraph Edition)
====================================================
Generate industry-grade, Medium-compatible blog posts using LangGraph agents.

Usage:
    python main.py --topic "Your Topic Here"
    python main.py --topic "Machine Learning Best Practices" --iterations 3
    python main.py --topic "Python Tips" --model gpt-4o
"""

import argparse
import os
import sys
import re
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# Load environment variables
load_dotenv()

from config import Config
from graph import BlogOrchestrator
from formatters import MediumFormatter


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s_]+', '_', text)
    text = text.strip('_')
    return text[:50]


def main():
    """Main entry point for the blog generation system."""
    console = Console()
    
    parser = argparse.ArgumentParser(
        description="Multi-Agent Blog Writing System (LangGraph)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --topic "Machine Learning Best Practices for 2024"
  python main.py --topic "Python Data Structures" --iterations 5
  python main.py --topic "API Design" --model gpt-4o-mini
  python main.py --topic "Cloud Architecture" --output custom_name.md
        """
    )
    
    parser.add_argument(
        "--topic", "-t",
        required=True,
        help="The blog topic to write about"
    )
    
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=3,
        help="Number of refinement iterations (default: 3)"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output filename (default: auto-generated from topic)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory (default: output)"
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Also export as HTML"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also export metadata as JSON"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Configure the system
    Config.LLM_MODEL = args.model
    Config.NUM_ITERATIONS = args.iterations
    
    # Verify API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error:[/red] OPENAI_API_KEY environment variable not set")
        console.print("\nPlease set your OpenAI API key:")
        console.print("  export OPENAI_API_KEY='your-key'")
        console.print("\nOr add it to your .env file")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    if args.output:
        output_name = args.output
    else:
        slug = slugify(args.topic)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_name = f"{slug}_{timestamp}.md"
    
    output_path = output_dir / output_name
    
    # Show configuration
    if not args.quiet:
        console.print(Panel(
            f"[bold]Topic:[/bold] {args.topic}\n"
            f"[bold]Model:[/bold] {Config.LLM_MODEL}\n"
            f"[bold]Iterations:[/bold] {args.iterations}\n"
            f"[bold]Output:[/bold] {output_path}",
            title="📝 Blog Generation Configuration",
            border_style="blue"
        ))
    
    # Create orchestrator and generate blog
    orchestrator = BlogOrchestrator(verbose=not args.quiet)
    
    try:
        result = orchestrator.generate_blog(
            topic=args.topic,
            num_iterations=args.iterations
        )
    except Exception as e:
        console.print(f"\n[red]Generation Error:[/red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Export the blog
    try:
        # Format and save the blog
        content = result.get("draft_content", "")
        
        # Add metadata header
        seo = result.get("seo_analysis", {})
        meta_block = f"""<!--
BLOG METADATA (Remove before publishing)
========================================
Title: {result.get('title', '')}
Meta Description: {seo.get('meta_description', '')}
Primary Keywords: {', '.join(seo.get('primary_keywords', []))}
Word Count: {result.get('word_count', 0):,}
Code Examples: {result.get('code_block_count', 0)}
Tables: {result.get('table_count', 0)}
Review Score: {result.get('final_review_score', 0)}/10
SEO Score: {result.get('final_seo_score', 0)}/10
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
-->

"""
        
        # Footer
        footer = f"""

---

*📖 Reading time: ~{result.get('word_count', 0) // 200} minutes | 📝 {result.get('word_count', 0):,} words*

**Tags:** {', '.join(seo.get('primary_keywords', [result.get('topic', '')])[:5])}

---

*This article was generated with AI assistance and reviewed for quality and accuracy.*
"""
        
        full_content = meta_block + content + footer
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        console.print(f"\n[green]✓[/green] Blog saved to: [cyan]{output_path}[/cyan]")
        
        # Optional HTML export
        if args.html:
            html_path = output_path.with_suffix('.html')
            from formatters import BlogExporter
            
            # Create a simple result object for the exporter
            class SimpleResult:
                def __init__(self, state):
                    self.topic = state.get("topic", "")
                    self.final_draft = type('obj', (object,), {
                        'raw_content': state.get("draft_content", ""),
                        'title': state.get("title", ""),
                        'metadata': type('obj', (object,), {
                            'meta_description': state.get("seo_analysis", {}).get("meta_description", ""),
                            'primary_keywords': state.get("seo_analysis", {}).get("primary_keywords", []),
                        })()
                    })()
            
            html_content = BlogExporter.to_html(SimpleResult(result))
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            console.print(f"[green]✓[/green] HTML saved to: [cyan]{html_path}[/cyan]")
        
        # Optional JSON export
        if args.json:
            import json
            json_path = output_path.with_suffix('.json')
            json_data = {
                "title": result.get("title", ""),
                "topic": result.get("topic", ""),
                "word_count": result.get("word_count", 0),
                "code_blocks": result.get("code_block_count", 0),
                "tables": result.get("table_count", 0),
                "review_score": result.get("final_review_score", 0),
                "seo_score": result.get("final_seo_score", 0),
                "seo_analysis": result.get("seo_analysis", {}),
                "generation_time": result.get("generation_time", 0),
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            console.print(f"[green]✓[/green] JSON saved to: [cyan]{json_path}[/cyan]")
        
    except Exception as e:
        console.print(f"\n[red]Export Error:[/red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Show final statistics
    if not args.quiet:
        console.print()
        console.print(Panel(
            f"[bold green]Blog generation complete![/bold green]\n\n"
            f"📄 Word Count: {result.get('word_count', 0):,}\n"
            f"💻 Code Examples: {result.get('code_block_count', 0)}\n"
            f"📊 Tables: {result.get('table_count', 0)}\n"
            f"⭐ Review Score: {result.get('final_review_score', 0)}/10\n"
            f"🔍 SEO Score: {result.get('final_seo_score', 0)}/10\n"
            f"⏱️ Generation Time: {result.get('generation_time', 0):.1f}s\n\n"
            f"Your blog is ready to publish on Medium!",
            title="🎉 Success",
            border_style="green"
        ))


if __name__ == "__main__":
    main()
