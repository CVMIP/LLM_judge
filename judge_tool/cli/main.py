import os
import yaml
from typing import Optional, Dict
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv

from judge_tool.models.schemas import EvaluationInput, Rubric, PairwiseInput
from judge_tool.core.judge import AbsoluteScorer, PairwiseScorer

import pandas as pd
from judge_tool.core.batch import BatchEvaluator

# Load environment variables from .env
load_dotenv()

app = typer.Typer(help="LLM-as-a-Judge Evaluation Tool")
console = Console()

def validate_api_keys():
    """Check for essential API keys."""
    # This can be expanded based on the models supported
    if not any([os.getenv("OPENAI_API_KEY"), os.getenv("ANTHROPIC_API_KEY"), os.getenv("GOOGLE_API_KEY")]):
        console.print("[yellow]Warning: No common API keys found (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY). LiteLLM may fail unless you have set other keys.[/yellow]")

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        console.print(f"[red]Error: Input file not found at {path}[/red]")
        raise typer.Exit(code=1)
        
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        console.print("[red]Error: Unsupported file format. Use CSV, JSONL, or Parquet.[/red]")
        raise typer.Exit(code=1)

@app.command()
def evaluate_dataset(
    input_path: str = typer.Option(..., "--input", "-i", help="Path to CSV or JSONL dataset."),
    output_path: str = typer.Option("results.csv", "--output", "-o", help="Path to save results."),
    rubric_path: str = typer.Option("configs/helpfulness.yaml", "--rubric", "-rb", help="Path to rubric YAML."),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Judge model."),
    workers: int = typer.Option(5, "--workers", "-w", help="Number of parallel workers."),
    prompt_col: str = typer.Option("prompt", "--prompt-col", help="Name of the prompt column."),
    response_col: str = typer.Option("response", "--response-col", help="Name of the response column."),
    ref_col: Optional[str] = typer.Option(None, "--ref-col", help="Name of the reference answer column.")
):
    """
    Evaluate a dataset of prompts and responses with absolute scoring.
    """
    validate_api_keys()
    if not os.path.exists(rubric_path):
        console.print(f"[red]Error: Rubric file not found at {rubric_path}[/red]")
        raise typer.Exit(code=1)

    df = load_dataset(input_path)

    with open(rubric_path, "r") as f:
        rubric = Rubric(**yaml.safe_load(f))

    scorer = AbsoluteScorer(model_name=model)
    evaluator = BatchEvaluator(max_workers=workers)

    column_mapping = {
        "prompt": prompt_col,
        "response": response_col
    }
    if ref_col:
        column_mapping["reference"] = ref_col

    results_df = evaluator.evaluate_dataframe(df, rubric, scorer, column_mapping)

    results_df.to_csv(output_path, index=False)
    console.print(f"[bold green]Evaluation complete! Results saved to {output_path}[/bold green]")
    
    # Show summary
    avg_score = results_df["score"].mean()
    console.print(Panel(f"Average Score: {avg_score:.2f} / {rubric.max_score}", title="Summary"))

@app.command()
def compare_dataset(
    input_path: str = typer.Option(..., "--input", "-i", help="Path to CSV or JSONL dataset."),
    output_path: str = typer.Option("compare_results.csv", "--output", "-o", help="Path to save results."),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Judge model."),
    workers: int = typer.Option(5, "--workers", "-w", help="Number of parallel workers."),
    prompt_col: str = typer.Option("prompt", "--prompt-col", help="Name of the prompt column."),
    res_a_col: str = typer.Option("response_a", "--res-a-col", help="Name of the response A column."),
    res_b_col: str = typer.Option("response_b", "--res-b-col", help="Name of the response B column."),
    ref_col: Optional[str] = typer.Option(None, "--ref-col", help="Name of the reference answer column.")
):
    """
    Compare two responses across a dataset.
    """
    validate_api_keys()
    df = load_dataset(input_path)

    scorer = PairwiseScorer(model_name=model)
    evaluator = BatchEvaluator(max_workers=workers)

    column_mapping = {
        "prompt": prompt_col,
        "response_a": res_a_col,
        "response_b": res_b_col
    }
    if ref_col:
        column_mapping["reference"] = ref_col

    results_df = evaluator.evaluate_pairwise_dataframe(df, scorer, column_mapping)

    results_df.to_csv(output_path, index=False)
    console.print(f"[bold green]Comparison complete! Results saved to {output_path}[/bold green]")
    
    # Show summary
    winner_counts = results_df["winner"].value_counts().to_dict()
    summary_text = "\n".join([f"{winner}: {count}" for winner, count in winner_counts.items()])
    console.print(Panel(summary_text, title="Winner Summary"))

@app.command()
def evaluate_single(
    prompt: str = typer.Option(..., "--prompt", "-p", help="The original prompt."),
    response: str = typer.Option(..., "--response", "-r", help="The LLM response to evaluate."),
    rubric_path: str = typer.Option("configs/helpfulness.yaml", "--rubric", "-rb", help="Path to the rubric YAML file."),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="The judge model name."),
    reference: Optional[str] = typer.Option(None, "--reference", "-ref", help="Optional reference answer.")
):
    """
    Evaluate a single LLM response using a specific rubric.
    """
    validate_api_keys()
    if not os.path.exists(rubric_path):
        console.print(f"[red]Error: Rubric file not found at {rubric_path}[/red]")
        raise typer.Exit(code=1)

    with open(rubric_path, "r") as f:
        rubric_data = yaml.safe_load(f)
        rubric = Rubric(**rubric_data)

    scorer = AbsoluteScorer(model_name=model)
    input_data = EvaluationInput(prompt=prompt, response=response, reference_answer=reference)

    with console.status(f"[bold green]Judging with {model}..."):
        result = scorer.score(input_data, rubric)

    # Display results
    console.print(Panel(f"[bold cyan]Criterion:[/bold cyan] {result.criterion}\n[bold cyan]Score:[/bold cyan] {result.score}/{rubric.max_score}", title="Evaluation Result"))
    console.print(Panel(Markdown(result.reasoning), title="Reasoning"))

@app.command()
def compare(
    prompt: str = typer.Option(..., "--prompt", "-p", help="The original prompt."),
    response_a: str = typer.Option(..., "--response-a", "-ra", help="Model A response."),
    response_b: str = typer.Option(..., "--response-b", "-rb", help="Model B response."),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="The judge model name."),
    reference: Optional[str] = typer.Option(None, "--reference", "-ref", help="Optional reference answer.")
):
    """
    Compare two model responses and determine a winner.
    """
    validate_api_keys()
    scorer = PairwiseScorer(model_name=model)
    input_data = PairwiseInput(
        prompt=prompt, 
        response_a=response_a, 
        response_b=response_b, 
        reference_answer=reference
    )

    with console.status(f"[bold green]Comparing with {model}..."):
        result = scorer.compare(input_data)

    # Display results
    color = "green" if result.winner == "A" else "blue" if result.winner == "B" else "yellow"
    console.print(Panel(f"[bold {color}]Winner:[/bold {color}] {result.winner}", title="Pairwise Comparison Result"))
    console.print(Panel(Markdown(result.reasoning), title="Reasoning"))

@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind.")
):
    """
    Launch the web-based UI.
    """
    import uvicorn
    from judge_tool.web.app import app as web_app
    
    console.print(f"[bold green]Starting UI at http://{host}:{port}[/bold green]")
    uvicorn.run(web_app, host=host, port=port)

if __name__ == "__main__":
    app()
