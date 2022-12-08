from transformers import pipeline
import gradio as gr

pipe = pipeline(model="Yilin98/whisper-small-hi")  # change to "your-username/the-name-you-picked"

def transcribe(audio=None, file=None, youtube=None):
    if (audio is None) and (file is None) and (youtube is None):
        return "No audio provided!"
    elif audio is not None:
        input=audio
    elif file is not None:
        input=file
    elif youtube is not None:
        yt=pt.YouTube("https://www.youtube.com/watch?v=4KI9BBW_aP8")
        stream=yt.streams.filter(only_audio=True)[0]
        stream.download(filename="audio.mp3")
        input=audio
    text = pipe(input)["text"]
    return text

iface = gr.Interface(
    fn=transcribe, 
    inputs=[
        gr.Audio(source="microphone", type="filepath", interactive=True),
        gr.Audio(source="upload", type="filepath", interactive=True),
        gr.Text(label="URL (YouTube, etc.)")], 
    outputs="text",
    title="Whisper Small Swedish",
    description="Realtime demo for Swedish speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()