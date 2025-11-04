import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
import soundfile as sf
import numpy as np

# ---------------------------
# 1️⃣ Safe audio loader (no TorchCodec)
# ---------------------------
def safe_load(path):
    waveform, sr = sf.read(path, dtype="float32")
    if len(waveform.shape) == 1:
        waveform = np.expand_dims(waveform, axis=0)
    waveform = torch.tensor(waveform)
    return waveform, sr

torchaudio.load = safe_load  # override built-in loader

# ---------------------------
# 2️⃣ Dataset & preprocessing
# ---------------------------
from torchaudio.datasets import LIBRISPEECH
train_dataset = LIBRISPEECH("./data", url="test-clean", download=True)

transform = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)

def collate_fn(batch):
    inputs, targets = [], []
    input_lengths, target_lengths = [], []
    for waveform, _, _, _, transcript, _, _ in batch:
        mfcc = transform(waveform).squeeze(0).transpose(0, 1)
        inputs.append(mfcc)
        input_lengths.append(mfcc.shape[0])

        target = torch.tensor(
            [ord(c) - 96 if c != ' ' else 0 for c in transcript.lower() if c.isalpha() or c == ' '],
            dtype=torch.long,
        )
        targets.append(target)
        target_lengths.append(len(target))

    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets = torch.cat(targets)
    return inputs, targets, input_lengths, target_lengths

loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# ---------------------------
# 3️⃣ Tiny model
# ---------------------------
class TinyASR(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=29):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.rnn(x)
        return self.fc(x)

model = TinyASR()
criterion = nn.CTCLoss(blank=28)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# 4️⃣ Training loop (light)
# ---------------------------
print("🚀 Training started...")
for epoch in range(1):
    for i, (inputs, targets, input_lengths, target_lengths) in enumerate(loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.log_softmax(2)
        loss = criterion(
            outputs.transpose(0, 1),
            targets,
            input_lengths,
            target_lengths,
        )
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {epoch} | Step {i} | Loss {loss.item():.4f}")
        if i > 20:  # keep it short for demo
            break

torch.save(model.state_dict(), "tiny_asr.pth")
print("✅ Model saved as tiny_asr.pth")
