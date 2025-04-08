import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf

import torch
import torchaudio
import torchaudio.transforms as T

def preprocess_pitch(file_path):
    pitch_ls = [] # 'uttid'='pitch tensor'
    with open(file_path, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            utt_id, processed_line = extract_pitch_alongtime(line.strip())
            pitch_ls.append([utt_id, processed_line])

    return pitch_ls


def extract_pitch_alongtime(line):
    parts = line.split()
    uttid = parts[0]+".wav"
    data_unit = parts[1:]

    pitch_ls = []
    for i in range(0, len(data_unit), 2):
        suffix = data_unit[i].split('_')[-1]
        if suffix.isdigit():
            pitch_ls.append(int(suffix))
        else:
            i+=1 # '<svs_placeholder>'
    return uttid, pitch_ls

class AudioDataset(Dataset):
    def __init__(self, audio_dir, transform=None, sample_rate=16000):
        self.audio_dir = audio_dir
        self.audio_files = [
            f for f in os.listdir(audio_dir)
            if f.endswith('.wav') or f.endswith('.flac')
        ]
        self.transform = transform
        self.sample_rate = sample_rate

        self.label_file = "/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/dump/audio_raw_svs_acesinger/tr_no_dev/label"
        self.pitch_ls = preprocess_pitch(self.label_file)

    def __len__(self):
        return len(self.pitch_ls)

    def __getitem__(self, idx):
        
        filename, pitch = self.pitch_ls[idx]
        # filename = self.audio_files[idx]
        filepath = os.path.join(self.audio_dir, filename)
    
        waveform, sr = sf.read(filepath)
        waveform = torch.from_numpy(waveform).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        return {
            "waveform": waveform,              # shape: [1, T]
            "length": waveform.shape[1],       # T
            "filename": self.audio_files[idx],
            "pitch": pitch
        }



def collate_fn(batch):
    # padding to max length in a batch
    waveforms = [item['waveform'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch])
    filenames = [item['filename'] for item in batch]
    pitchs = [item['pitch'] for item in batch]

    max_len = max(lengths)

    longest_pitch = len(max(pitchs, key=len))
    padded_pitch = [item['pitch'] + [0] * (longest_pitch - len(item['pitch'])) for item in batch]

    pitchs_tensor = torch.tensor(padded_pitch)

    padded_waveforms = torch.zeros(len(waveforms), 1, max_len)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, :, :w.shape[1]] = w

    return padded_waveforms, lengths, filenames, pitchs_tensor



if __name__ == "__main__":
    dataset = AudioDataset("./datasets/wav_dump/", transform=None, sample_rate=16000)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)

    for batch_waveforms, lengths, filenames, pitch in dataloader:

        print("Batch shape:", batch_waveforms.shape)
        print("Filenames:", filenames)

        # TODO (yiwen) 1. map waveform length to token length, 
        # 2. get feature mask based on token length, 
        # 3. use the mask on predicted loss
