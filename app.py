import gradio as gr
from src.inference import predict_video


def run_detector(video):
    result = predict_video(video)
    return result


demo = gr.Interface(
    fn=run_detector,
    inputs=gr.Video(),
    outputs="json",
    title="Deepfake Video Detector",
    description="Upload a video and detect if it is real or fake"
)

demo.launch()