"""Command line interface for the YOLO application."""

import os
import cv2
import typer
from typing import Optional

from .detector import DefectDetector
from .ui import main as gui_main

app = typer.Typer(help="YOLO defect detection toolkit")

@app.command()
def gui() -> None:
    """Launch the graphical user interface."""
    gui_main()

@app.command()
def detect_image(
    image: str = typer.Argument(..., help="Image file to process"),
    model: Optional[str] = typer.Option(None, "--model", help="Path to model"),
    output: Optional[str] = typer.Option(None, "--output", help="Where to save annotated image"),
    conf: float = typer.Option(0.25, "--conf", help="Confidence threshold"),
    device: Optional[str] = typer.Option(None, "--device", help="Inference device"),
    batch: int = typer.Option(1, "--batch", help="Batch size"),
) -> None:
    """Run defect detection on a single image."""
    det = DefectDetector(model_path=model, conf_thres=conf, device=device, batch_size=batch)
    results = det.detect(image)
    typer.echo(f"Detected {results['num_detections']} defects")
    if output:
        img = cv2.imread(image)
        result_img = det.draw_results(img, results)
        cv2.imwrite(output, result_img)
        json_path = os.path.splitext(output)[0] + ".json"
        det.save_results_to_json(results, json_path)
        typer.echo(f"Results saved to {output} and {json_path}")

@app.command()
def detect_video(
    video: str = typer.Argument(..., help="Video file to process"),
    model: Optional[str] = typer.Option(None, "--model", help="Path to model"),
    output: Optional[str] = typer.Option(None, "--output", help="Where to save annotated video"),
    conf: float = typer.Option(0.25, "--conf", help="Confidence threshold"),
    device: Optional[str] = typer.Option(None, "--device", help="Inference device"),
    batch: int = typer.Option(1, "--batch", help="Batch size"),
) -> None:
    """Run defect detection on a video."""
    det = DefectDetector(model_path=model, conf_thres=conf, device=device, batch_size=batch)
    det.process_video(video, output_path=output, show_preview=True)

if __name__ == "__main__":
    app()
