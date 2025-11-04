import torch
import torch.nn as nn
import torchaudio

labels = ["'"] + [chr(i) for i in range(97, 123)] + [" "]
label2index = {c: i for i, c in enumerate(labels)}
index2label = {i: c for c, i in label2index.items()}

mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000, n_mfcc=40,
    melkwargs={"n_fft": 400, "hop_length": 160}
)

def collate_fn(batch):
    features, targets, input_lengths, target_lengths = [], [], [], []

    for waveform, sample_rate, transcript, *_ in batch:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        mfcc = mfcc_transform(waveform).squeeze(0).transpose(0, 1)
        features.append(mfcc)
        input_lengths.append(mfcc.shape[0])

        target = torch.tensor([label2index[c] for c in transcript.lower() if c in label2index])
        targets.append(target)
        target_lengths.append(len(target))

    features_padded = nn.utils.rnn.pad_sequence(features, batch_first=True)
    targets_padded = torch.cat(targets)
    return features_padded, targets_padded, input_lengths, target_lengths


def greedy_decode(logits):
    pred = torch.argmax(logits, dim=-1)
    decoded = []
    for seq in pred:
        text = ""
        prev = None
        for idx in seq:
            if idx != prev and idx < len(index2label):
                text += index2label.get(idx.item(), "")
            prev = idx
        decoded.append(text)
    return decoded
