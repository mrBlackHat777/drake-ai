import typer
import os
from rich.console import Console

app = typer.Typer()
console = Console()

def validate_input_file(input_file: str) -> str:
    if not os.path.exists(input_file) or not input_file.endswith(".wav"):
        return ""
    return input_file

def convert_voice(input_file: str) -> str:
    return input_file.replace(".wav", "_converted.wav")

@app.command()
def convert(
    input_file: str = typer.Argument(..., help="Path to the input audio file."),
):
    input_file = validate_input_file(input_file)
    if not input_file:
        console.print("[bold red]Invalid input file. Please provide a valid audio file.[/bold red]")
        return

    console.print(f"[bold green]Converting {input_file}...[/bold green]")
    # output_file = convert_voice(input_file)
    # TODO
    console.print(f"[bold green]Conversion complete. Output: {output_file}[/bold green]")

if __name__ == "__main__":
    app()
