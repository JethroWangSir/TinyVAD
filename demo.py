import os
import torch
import torch.nn as nn
import gradio as gr
import numpy as np
import torchaudio
import torchaudio.transforms as T
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from model.tinyvad import TinyVAD

WINDOW_SIZE = 0.16

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

font_path = '/share/nas169/jethrowang/fonts/Times_New_Roman.ttf'
font_prop = FontProperties(fname=font_path, size=18)

# Load the model
if WINDOW_SIZE == 0.63:
    patch_size = 8
elif WINDOW_SIZE == 0.32:
    patch_size = 4
elif WINDOW_SIZE == 0.16:
    patch_size = 2
elif WINDOW_SIZE == 0.025:
    patch_size = 1
model = TinyVAD(1, 32, 64, patch_size, 2).to(device)
checkpoint_path = '/share/nas169/jethrowang/TinyVAD/exp/exp_0.16_tinyvad/model_epoch_112_auroc=0.8344.ckpt'
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

mel_spectrogram = T.MelSpectrogram(sample_rate=16000, n_mels=64, win_length=400, hop_length=160)
log_mel_spectrogram = T.AmplitudeToDB()

chunk_duration = WINDOW_SIZE
shift_duration = WINDOW_SIZE * 0.75

# Define the prediction function with dynamic plot updates
def predict(audio_file, threshold):
    start_time = time.time()

    try:
        waveform, sample_rate = torchaudio.load(audio_file)
    except Exception as e:
        # yield "Error loading audio file.", None, None, None
        yield None, None, None, None
        return
    
    # Calculate total audio duration
    audio_duration = waveform.size(1) / sample_rate

    # Check if the audio is shorter than the chunk_duration
    if audio_duration < chunk_duration:
        required_length = int(chunk_duration * sample_rate)  # Length in samples
        padding_length = required_length - waveform.size(1)  # How much padding is needed
        waveform = torch.nn.functional.pad(waveform, (0, padding_length))  # Pad with zeros on the right
    
    chunk_size = int(chunk_duration * sample_rate)  # Size of each chunk
    shift_size = int(shift_duration * sample_rate)  # Shift size for each window move
    num_chunks = (waveform.size(1) - chunk_size) // shift_size + 1  # Calculate number of shifts

    predictions = []
    time_stamps = []

    # Initialize the plot figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel('Time (seconds)', fontproperties=font_prop)
    ax.set_ylabel('Probability', fontproperties=font_prop)
    ax.set_title('Voice Activity Detection Probability Over Time', fontproperties=font_prop)
    ax.axhline(y=threshold, color='tab:red', linestyle='--', label='Threshold')
    ax.grid(True)
    ax.set_ylim([-0.05, 1.05])

    for i in range(num_chunks):
        start_idx = i * shift_size
        end_idx = start_idx + chunk_size
        chunk = waveform[:, start_idx:end_idx]

        if chunk.size(1) < chunk_size:  # If the last segment is smaller than chunk size, skip it
            break

        inputs = mel_spectrogram(chunk)
        inputs = log_mel_spectrogram(inputs).to(device).unsqueeze(0)

        with torch.no_grad():
            outputs = model(inputs)

        predictions.append(outputs.item())
        time_stamps.append(start_idx / sample_rate)  # Record start time for this chunk

        # Dynamically update the plot
        ax.plot(time_stamps, predictions, label='Speech Probability', color='tab:blue')

        yield "Processing...", None, None, gr.update(value=fig)

    avg_output = np.mean(predictions)
    prediction_time = time.time() - start_time

    # Final decision based on the threshold
    prediction = "Speech" if avg_output > threshold else "Non-speech"
    probability = f'{(float(avg_output) * 100):.2f}'
    inference_time = f'{prediction_time:.4f}'

    # Final update with result and plot
    yield prediction, probability, inference_time, gr.update(value=fig)

# Path to the logo image
logo_path = "./img/logo.png"

# Create Gradio interface
with gr.Blocks() as demo:
    # Add logo at the top
    gr.Image(logo_path, elem_id="logo", height=100)
    
    # Title and description
    gr.Markdown("<h1 style='text-align: center; color: black;'>Voice Activity Detection using TinyVAD</h1>")
    gr.Markdown("<h3 style='text-align: center; color: black;'>Upload an audio file or record using the microphone to predict if it contains speech and view the probability curve.</h3>")
    
    # Interface
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Upload or Record Audio")
            plot_output = gr.Plot(label="Probability Curve")  # Use gr.Plot for dynamic plot
        with gr.Column():
            threshold_input = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.1, label="Threshold")
            prediction_output = gr.Textbox(label="Prediction")
            probability_output = gr.Number(label="Average Probability (%)")
            time_output = gr.Textbox(label="Inference Time (seconds)")

    # Connect inputs and outputs
    # gr.Button("Predict").click(predict, [audio_input, threshold_input], [prediction_output, probability_output, time_output, plot_output])
    audio_input.change(predict, [audio_input, threshold_input], [prediction_output, probability_output, time_output, plot_output])

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)
