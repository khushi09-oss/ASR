import torch
import torchaudio
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# same model class
class TinyASR(torch.nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=29):
        super().__init__()
        self.conv = torch.nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.rnn = torch.nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)
    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.rnn(x)
        return self.fc(x)

# Load model
model = TinyASR()
model.load_state_dict(torch.load("tiny_asr.pth", map_location="cpu"))
model.eval()

transform = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)

def record_audio(seconds=5, samplerate=16000):
    print("🎤 Speak now...")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    write("sample.wav", samplerate, audio)
    print("✅ Recorded to sample.wav")
    return torch.tensor(audio.T)

def greedy_decode(logits):
    pred = torch.argmax(logits, dim=-1)
    mapping = {0: ' ', **{i: chr(i + 96) for i in range(1, 27)}}
    decoded = ''.join([mapping.get(p.item(), '') for p in pred[0]])
    return decoded.replace('  ', ' ').strip()

waveform = record_audio()
mfcc = transform(waveform).squeeze(0).transpose(0, 1).unsqueeze(0)
with torch.no_grad():
    output = model(mfcc)
    logits = output.log_softmax(2)
print("🗣️ Predicted text:", greedy_decode(logits))
