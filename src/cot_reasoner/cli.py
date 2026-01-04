"""Command-line interface for the Chain of Thought Reasoner."""

import sys
from typing import Optional

import uuid

from dotenv import load_dotenv
import typer

# Load environment variables from .env file
load_dotenv()
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from cot_reasoner.core.reasoner import Reasoner
from cot_reasoner.db import get_db

app = typer.Typer(
    name="cot-reasoner",
    help="Chain of Thought Reasoner - Break down complex problems step by step",
    add_completion=False,
    invoke_without_command=True,
)

console = Console()


@app.callback()
def main_callback(
    ctx: typer.Context,
    query: Optional[str] = typer.Argument(None, help="The problem or question to reason about"),
    provider: str = typer.Option("openai", "--provider", "-p", help="LLM provider (openai, anthropic)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    strategy: str = typer.Option("standard", "--strategy", "-s", help="Reasoning strategy"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    stream: bool = typer.Option(False, "--stream", help="Stream the response"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    memory: bool = typer.Option(True, "--memory/--no-memory", help="Enable conversation memory (interactive mode)"),
):
    """
    Chain of Thought Reasoner - Break down complex problems step by step.

    Examples:
        cot-reasoner "What is 15% of 240?"
        cot-reasoner -p anthropic "Explain quantum entanglement"
        cot-reasoner -s self_consistency "Complex math problem"
        cot-reasoner --no-memory  # Interactive without memory
    """
    # Only run if no subcommand was invoked
    if ctx.invoked_subcommand is not None:
        return

    # If no query, start interactive mode
    if query is None:
        interactive_mode(provider=provider, model=model, strategy=strategy, temperature=temperature, verbose=verbose, memory=memory)
        return

    # Run reasoning
    run_reasoning(
        query=query,
        provider=provider,
        model=model,
        strategy=strategy,
        temperature=temperature,
        verbose=verbose,
        stream=stream,
        json_output=json_output,
    )


def print_reasoning_chain(chain, verbose: bool = False):
    """Pretty print a reasoning chain."""
    # Print query
    console.print(Panel(chain.query, title="Query", border_style="blue"))
    console.print()

    # Print reasoning steps
    console.print("[bold cyan]Reasoning Steps:[/bold cyan]")
    for step in chain.steps:
        console.print(f"  [yellow]Step {step.number}:[/yellow] {step.content}")
    console.print()

    # Print answer
    if chain.answer:
        answer_panel = Panel(
            f"[bold green]{chain.answer}[/bold green]\n\n"
            f"[dim]Confidence: {chain.confidence:.0%}[/dim]",
            title="Answer",
            border_style="green",
        )
        console.print(answer_panel)

    # Print metadata if verbose
    if verbose:
        console.print()
        table = Table(title="Metadata", show_header=False)
        table.add_column("Key", style="cyan")
        table.add_column("Value")
        table.add_row("Provider", chain.provider)
        table.add_row("Model", chain.model)
        table.add_row("Strategy", chain.strategy)
        table.add_row("Total Tokens", str(chain.total_tokens))
        table.add_row("Steps", str(chain.step_count))
        console.print(table)


def run_reasoning(
    query: str,
    provider: str,
    model: Optional[str],
    strategy: str,
    temperature: float,
    verbose: bool,
    stream: bool,
    json_output: bool,
):
    """Core reasoning logic."""
    try:
        reasoner = Reasoner(
            provider=provider,
            model=model,
            strategy=strategy,
            temperature=temperature,
        )

        if stream:
            # Streaming mode
            console.print(Panel(query, title="Query", border_style="blue"))
            console.print()
            console.print("[bold cyan]Reasoning:[/bold cyan]")
            for chunk in reasoner.reason_stream(query):
                console.print(chunk, end="")
            console.print()
        else:
            # Standard mode with progress spinner
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Thinking...", total=None)
                result = reasoner.reason(query)

            # Save to database
            db = get_db()
            result_id = str(uuid.uuid4())
            db.save_result(result_id, result, status="completed")

            if json_output:
                console.print(result.to_json())
            else:
                print_reasoning_chain(result, verbose=verbose)
                if verbose:
                    console.print(f"\n[dim]Saved to database with ID: {result_id}[/dim]")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def interactive_mode(
    provider: str = "openai",
    model: Optional[str] = None,
    strategy: str = "standard",
    temperature: float = 0.7,
    verbose: bool = False,
    memory: bool = True,
):
    """Start an interactive reasoning session."""
    memory_status = "[green]enabled[/green]" if memory else "[dim]disabled[/dim]"
    console.print(
        Panel(
            "[bold]Chain of Thought Reasoner[/bold]\n\n"
            "Enter your questions for step-by-step reasoning.\n"
            f"Memory: {memory_status}\n\n"
            "Commands: [cyan]quit[/cyan], [cyan]exit[/cyan], [cyan]config[/cyan], [cyan]clear[/cyan], [cyan]history[/cyan], [cyan]debug[/cyan]",
            border_style="blue",
        )
    )

    try:
        reasoner = Reasoner(
            provider=provider,
            model=model,
            strategy=strategy,
            temperature=temperature,
            memory=memory,
        )
        console.print(f"[dim]Using {reasoner}[/dim]\n")
    except ValueError as e:
        console.print(f"[red]Error initializing reasoner:[/red] {e}")
        raise typer.Exit(1)

    while True:
        try:
            query = Prompt.ask("\n[bold cyan]Question[/bold cyan]")

            if query.lower() in ("quit", "exit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break

            if query.lower() == "config":
                table = Table(title="Current Configuration")
                table.add_column("Setting", style="cyan")
                table.add_column("Value")
                table.add_row("Provider", provider)
                table.add_row("Model", reasoner.model)
                table.add_row("Strategy", strategy)
                table.add_row("Temperature", str(temperature))
                table.add_row("Memory", "enabled" if memory else "disabled")
                if reasoner.memory:
                    table.add_row("Memory Turns", str(len(reasoner.memory._history)))
                console.print(table)
                continue

            if query.lower() == "clear":
                if reasoner.memory:
                    reasoner.clear_memory()
                    console.print("[green]Conversation memory cleared.[/green]")
                else:
                    console.print("[dim]Memory is not enabled.[/dim]")
                continue

            if query.lower() == "history":
                if reasoner.memory and not reasoner.memory.is_empty:
                    console.print("\n[bold cyan]Conversation History:[/bold cyan]")
                    for i, turn in enumerate(reasoner.memory._history, 1):
                        console.print(f"  [yellow]Q{i}:[/yellow] {turn.query}")
                        console.print(f"  [green]A{i}:[/green] {turn.answer}\n")
                else:
                    console.print("[dim]No conversation history.[/dim]")
                continue

            if query.lower() == "debug":
                if reasoner.memory and not reasoner.memory.is_empty:
                    console.print("\n[bold cyan]Context being sent to LLM:[/bold cyan]")
                    console.print(Panel(reasoner.memory.get_context(), border_style="yellow"))
                else:
                    console.print("[dim]Memory is empty - no context will be sent.[/dim]")
                continue

            if not query.strip():
                continue

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Thinking...", total=None)
                result = reasoner.reason(query)

            # Save to database
            db = get_db()
            result_id = str(uuid.uuid4())
            db.save_result(result_id, result, status="completed")

            console.print()
            print_reasoning_chain(result, verbose=verbose)

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type 'quit' to exit.[/dim]")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


@app.command()
def interactive(
    provider: str = typer.Option("openai", "--provider", "-p", help="LLM provider"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    strategy: str = typer.Option("standard", "--strategy", "-s", help="Reasoning strategy"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    memory: bool = typer.Option(True, "--memory/--no-memory", help="Enable conversation memory"),
):
    """Start an interactive reasoning session with conversation memory."""
    interactive_mode(provider=provider, model=model, strategy=strategy, temperature=temperature, verbose=verbose, memory=memory)


@app.command()
def providers():
    """List available LLM providers."""
    table = Table(title="Available Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Description")

    providers_info = {
        "openai": "OpenAI GPT models (GPT-4, GPT-4o, etc.)",
        "anthropic": "Anthropic Claude models (Claude 3.5, etc.)",
    }

    for name in Reasoner.list_providers():
        desc = providers_info.get(name, "Custom provider")
        table.add_row(name, desc)

    console.print(table)


@app.command()
def strategies():
    """List available reasoning strategies."""
    table = Table(title="Available Strategies")
    table.add_column("Strategy", style="cyan")
    table.add_column("Description")

    strategies_info = {
        "standard": "Standard CoT with explicit step-by-step prompting",
        "zero_shot": "Minimal prompting with 'Let's think step by step'",
        "self_consistency": "Multiple reasoning paths with majority voting",
    }

    for name in Reasoner.list_strategies():
        desc = strategies_info.get(name, "Custom strategy")
        table.add_row(name, desc)

    console.print(table)


@app.command()
def version():
    """Show version information."""
    from cot_reasoner import __version__

    console.print(f"CoT Reasoner version [cyan]{__version__}[/cyan]")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
