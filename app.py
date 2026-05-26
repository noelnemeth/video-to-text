import gradio as gr
import os
import time
from transcribe import transcribe_video

def process_video(video_file, model_size, fast_mode, progress=gr.Progress()):
    if video_file is None:
        return "Kérlek, húzz be egy videó fájlt!", None
    
    video_path = video_file.name
    beam_size = 1 if fast_mode else 5
    
    progress(0, desc="Modell betöltése...")
    
    text, output_path = transcribe_video(
        video_path, 
        model_size=model_size, 
        beam_size=beam_size,
        verbose=True
    )
    
    if text is None:
        return "Hiba történt az átírás során. Ellenőrizd a terminált a részletekért.", None
        
    return text, output_path

with gr.Blocks(title="Video to Text Transcriber") as demo:
    gr.Markdown("# Video to Text Transcriber")
    gr.Markdown("Alakítsd a videóidat szöveggé gyorsan és egyszerűen az OpenAI Whisper segítségével.")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.File(label="Videó fájl behúzása", file_types=["video"])
            model_dropdown = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large-v3"], 
                value="base", 
                label="Whisper Modell mérete"
            )
            fast_mode = gr.Checkbox(label="Gyors mód (kisebb pontosság)", value=True)
            transcribe_btn = gr.Button("🚀 Szöveggé alakítás", variant="primary")
        
        with gr.Column(scale=2):
            text_output = gr.Textbox(label="Átirat", lines=15)
            file_output = gr.File(label="Letölthető .txt fájl")

    gr.Markdown("---")
    gr.Markdown("### Tippek:")
    gr.Markdown("- **tiny/base**: Nagyon gyors, de angolhoz jobb.")
    gr.Markdown("- **small/medium**: Magyar nyelvhez ezeket ajánljuk.")
    gr.Markdown("- **Gyors mód**: Kikapcsolva lassabb, de pontosabb átiratot készít.")

    transcribe_btn.click(
        fn=process_video, 
        inputs=[video_input, model_dropdown, fast_mode], 
        outputs=[text_output, file_output]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
