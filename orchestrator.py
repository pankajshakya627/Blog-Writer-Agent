"""
Blog Orchestrator for the Multi-Agent Blog Writing System.
Coordinates the Writer, Reviewer, and SEO agents through iterative refinement.
"""

import time
from typing import Optional, Callable
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from agents import WriterAgent, ReviewerAgent, SEOAgent
from models import (
    BlogDraft, BlogGenerationResult, IterationState,
    ReviewFeedback, SEOAnalysis, AgentRole
)
from config import Config


class BlogOrchestrator:
    """
    Orchestrates the multi-agent blog writing process.
    
    Coordinates three agents through iterative refinement:
    1. Writer Agent - Generates and refines content
    2. Reviewer Agent - Provides quality feedback
    3. SEO Agent - Optimizes for search engines
    """
    
    def __init__(self, verbose: bool = True):
        self.writer = WriterAgent()
        self.reviewer = ReviewerAgent()
        self.seo_optimizer = SEOAgent()
        self.verbose = verbose
        self.console = Console()
        self.iterations: list[IterationState] = []
    
    def generate_blog(
        self,
        topic: str,
        num_iterations: int = None,
        progress_callback: Optional[Callable] = None
    ) -> BlogGenerationResult:
        """
        Generate a complete blog post through iterative refinement.
        
        Args:
            topic: The blog topic/prompt
            num_iterations: Number of refinement iterations (default: 3)
            progress_callback: Optional callback for progress updates
        
        Returns:
            BlogGenerationResult with the final blog and iteration history
        """
        num_iterations = num_iterations or Config.NUM_ITERATIONS
        start_time = time.time()
        
        self._log_start(topic, num_iterations)
        
        current_draft: Optional[BlogDraft] = None
        review_feedback: Optional[ReviewFeedback] = None
        seo_analysis: Optional[SEOAnalysis] = None
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            disable=not self.verbose
        ) as progress:
            
            main_task = progress.add_task(
                f"[cyan]Generating blog...", 
                total=num_iterations * 3  # 3 agents per iteration
            )
            
            for iteration in range(1, num_iterations + 1):
                self._log_iteration_start(iteration, num_iterations)
                
                iteration_state = IterationState(
                    iteration_number=iteration,
                    draft=current_draft or BlogDraft(topic=topic, title="")
                )
                
                # Step 1: Writer Agent
                progress.update(main_task, description=f"[green]Iteration {iteration}: Writing content...")
                self._log_agent_start("Writer", iteration)
                
                current_draft = self.writer.process(
                    input_data=topic,
                    iteration=iteration,
                    previous_draft=current_draft,
                    review_feedback=review_feedback,
                    seo_analysis=seo_analysis
                )
                
                iteration_state.draft = current_draft
                iteration_state.add_message(
                    AgentRole.WRITER,
                    f"Generated draft with {current_draft.count_words()} words"
                )
                progress.advance(main_task)
                
                self._log_agent_complete("Writer", {
                    "words": current_draft.count_words(),
                    "code_blocks": current_draft.count_code_blocks(),
                    "tables": current_draft.count_tables()
                })
                
                # Step 2: Reviewer Agent
                progress.update(main_task, description=f"[yellow]Iteration {iteration}: Reviewing content...")
                self._log_agent_start("Reviewer", iteration)
                
                review_feedback = self.reviewer.process(
                    input_data=current_draft,
                    iteration=iteration
                )
                
                iteration_state.review_feedback = review_feedback
                iteration_state.add_message(
                    AgentRole.REVIEWER,
                    f"Review complete: Score {review_feedback.overall_score}/10"
                )
                progress.advance(main_task)
                
                self._log_agent_complete("Reviewer", {
                    "score": review_feedback.overall_score,
                    "improvements": len(review_feedback.improvements)
                })
                
                # Step 3: SEO Optimizer Agent
                progress.update(main_task, description=f"[magenta]Iteration {iteration}: Optimizing SEO...")
                self._log_agent_start("SEO Optimizer", iteration)
                
                seo_analysis = self.seo_optimizer.process(
                    input_data=current_draft,
                    iteration=iteration
                )
                
                iteration_state.seo_analysis = seo_analysis
                iteration_state.add_message(
                    AgentRole.SEO_OPTIMIZER,
                    f"SEO analysis complete: Score {seo_analysis.seo_score}/10"
                )
                progress.advance(main_task)
                
                self._log_agent_complete("SEO Optimizer", {
                    "seo_score": seo_analysis.seo_score,
                    "keywords": len(seo_analysis.primary_keywords)
                })
                
                self.iterations.append(iteration_state)
                
                if progress_callback:
                    progress_callback(iteration, num_iterations, current_draft)
        
        generation_time = time.time() - start_time
        
        # Update final metadata
        if seo_analysis:
            current_draft.metadata.primary_keywords = seo_analysis.primary_keywords
            current_draft.metadata.secondary_keywords = seo_analysis.secondary_keywords
            current_draft.metadata.meta_description = seo_analysis.meta_description
            if seo_analysis.optimized_title:
                current_draft.title = seo_analysis.optimized_title
                current_draft.metadata.title = seo_analysis.optimized_title
        
        current_draft.metadata.word_count = current_draft.count_words()
        current_draft.metadata.estimated_read_time = current_draft.count_words() // 200
        
        result = BlogGenerationResult(
            topic=topic,
            final_draft=current_draft,
            iterations=self.iterations,
            total_iterations=num_iterations,
            generation_time_seconds=generation_time
        )
        
        self._log_complete(result)
        
        return result
    
    def _log_start(self, topic: str, iterations: int):
        """Log the start of blog generation."""
        if not self.verbose:
            return
        
        self.console.print()
        self.console.print(Panel(
            f"[bold cyan]Multi-Agent Blog Generation[/bold cyan]\n\n"
            f"[white]Topic:[/white] {topic}\n"
            f"[white]Iterations:[/white] {iterations}\n"
            f"[white]Agents:[/white] Writer → Reviewer → SEO Optimizer",
            title="🚀 Starting",
            border_style="cyan"
        ))
    
    def _log_iteration_start(self, iteration: int, total: int):
        """Log the start of an iteration."""
        if not self.verbose:
            return
        
        self.console.print()
        self.console.print(f"[bold blue]━━━ Iteration {iteration}/{total} ━━━[/bold blue]")
    
    def _log_agent_start(self, agent_name: str, iteration: int):
        """Log when an agent starts processing."""
        if not self.verbose:
            return
        
        self.console.print(f"  [dim]→ {agent_name} Agent processing...[/dim]")
    
    def _log_agent_complete(self, agent_name: str, stats: dict):
        """Log when an agent completes processing."""
        if not self.verbose:
            return
        
        stats_str = ", ".join(f"{k}: {v}" for k, v in stats.items())
        self.console.print(f"  [green]✓[/green] {agent_name} complete ({stats_str})")
    
    def _log_complete(self, result: BlogGenerationResult):
        """Log the completion of blog generation."""
        if not self.verbose:
            return
        
        stats = result.get_statistics()
        
        table = Table(title="📊 Generation Complete", border_style="green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Word Count", f"{stats['word_count']:,}")
        table.add_row("Code Blocks", str(stats['code_blocks']))
        table.add_row("Tables", str(stats['tables']))
        table.add_row("Sections", str(stats['sections']))
        table.add_row("Iterations", str(stats['iterations']))
        table.add_row("Generation Time", stats['generation_time'])
        
        self.console.print()
        self.console.print(table)
        
        # Show final scores
        if result.iterations:
            last_iteration = result.iterations[-1]
            if last_iteration.review_feedback and last_iteration.seo_analysis:
                self.console.print()
                self.console.print(Panel(
                    f"[bold]Review Score:[/bold] {last_iteration.review_feedback.overall_score}/10\n"
                    f"[bold]SEO Score:[/bold] {last_iteration.seo_analysis.seo_score}/10",
                    title="Final Scores",
                    border_style="yellow"
                ))
