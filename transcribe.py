import argparse
import os
import sys
from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def transcribe_video(video_path, model_size="base", device="cpu", compute_type="int8"):
    if not os.path.exists(video_path):
        console.print(f"[red]Error:[/red] A fájl nem található: {video_path}")
        return

    output_path = os.path.splitext(video_path)[0] + ".txt"

    console.print(f"[cyan]Modell betöltése ({model_size})...[/cyan]")
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        console.print(f"[red]Hiba a modell betöltésekor:[/red] {e}")
        return

    console.print(f"[cyan]Feldolgozás megkezdése:[/cyan] {os.path.basename(video_path)}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(description="Átírás folyamatban...", total=None)
        
        segments, info = model.transcribe(video_path, beam_size=5)
        
        console.print(f"[green]Nyelv felismerve:[/green] {info.language} ({info.language_probability:.2f} valószínűség)")
        
        full_text = []
        for segment in segments:
            full_text.append(segment.text)
            progress.update(task, description=f"Átírás: {segment.text[:50]}...")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_text))

    console.print(f"\n[bold green]Sikeres átírás![/bold green]")
    console.print(f"Az eredmény mentve: [bold]{output_path}[/bold]")

def main():
    parser = argparse.ArgumentParser(description="Videó hanganyagának szöveggé alakítása.")
    parser.add_argument("video", help="A videó fájl elérési útja.")
    parser.add_argument("--model", default="base", help="Whisper modell mérete (tiny, base, small, medium, large-v3). Alapértelmezett: base")
    parser.add_argument("--device", default="cpu", help="Eszköz (cpu, cuda). Alapértelmezett: cpu")
    
    args = parser.parse_args()

    transcribe_video(args.video, model_size=args.model, device=args.device)

if __name__ == "__main__":
    main()
