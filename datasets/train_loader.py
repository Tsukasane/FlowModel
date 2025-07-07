import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf

import torch
import torchaudio
import torchaudio.transforms as T
import kaldiio



def extract_pitch_alongtime(line):
    parts = line.split()
    uttid = parts[0]+".wav"
    data_unit = parts[1:]

    pitch_ls = []
    for i in range(len(data_unit)):
        suffix = data_unit[i].split('_')[-1]
        if suffix.isdigit():
            prev_data = data_unit[i-1]
            prev_suffix = prev_data.split('_')[-1]
            if prev_suffix == 'AP' or prev_suffix == 'SP': # skip the pitch of non_phoneme, aspirate slience
                pitch_ls.append(0)
                continue
            pitch_ls.append(int(suffix))
    return uttid, pitch_ls



class TrainDataset(Dataset):
    def __init__(self, audio_dir, scp_root, label_file, ark_root, sample_rate=16000):
        
        split_size = 16 # TODO(yiwen) change this to 16

        self.pitch_dict = {}
        self.flattenCode_dict = {}
        self.pitch_ls = []
        self.flattenCode_ls = []
        self.sample_rate = sample_rate
        self.label_file = label_file
        self.audio_dir = audio_dir
        
        self.preprocess_pitch(label_file)


        for i in range(split_size): # NOTE(yiwen) temp
            csplit = str(i+1)
            token_file_name = f'output.{csplit}/wav.scp/rank0_token_wav.scp.scp'
            # output.1/label/rank0_token_label.scp
            scp_path = os.path.join(scp_root, token_file_name)
    
            self.get_flattenCode_dict(scp_path, ark_root) # add new elements to flatten code dict

        for k, v in self.pitch_dict.items():
            self.pitch_ls.append([k,v])

        for k, v in self.flattenCode_dict.items():
            self.flattenCode_ls.append([k,v])


    def get_flattenCode_dict(self, scp_path, root_ark):
        '''
        Args:
            - scp_path (str): store the decoded results
            - root_ark (str): root dir of exp folder
        '''
        
        with open(scp_path, 'r') as file:
            for line in file:
                uttid = line.strip().split()[0]
                ark_file = line.strip().split()[1]
                ark_path = os.path.join(root_ark, ark_file)
                np_array = kaldiio.load_mat(ark_path) # (443,) the decoded token index
                
                self.flattenCode_dict[uttid] = np_array


    def preprocess_pitch(self, file_path):
        with open(file_path, 'r') as infile:
            lines = infile.readlines()
            for line in lines:
                utt_id, processed_line = extract_pitch_alongtime(line.strip())
                self.pitch_dict[utt_id] = processed_line
        

    def __len__(self):
        return len(self.flattenCode_dict.items()) # the lm predict token length may be shorter than GT label pitch

    def __getitem__(self, idx):
        filename, flatten_code = self.flattenCode_ls[idx]
        pitch_key = "_".join(filename.split('_')[1:-1]) + '.wav'
        pitch = self.pitch_dict[pitch_key]

        wav_filename = filename[4:-8] + '.wav' # svs_ _samplex
        filepath = os.path.join(self.audio_dir, wav_filename)
    
        waveform, sr = sf.read(filepath) # read the clean wave from wav_dump
        waveform = torch.from_numpy(waveform).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        return {
            "flatten_code": flatten_code,              # shape: [1, T]
            "waveform": waveform,
            "length_waveform": waveform.shape[1], 
            "length_code": flatten_code.shape[0],       # T
            "filename": filename,
            "pitch": pitch
        }


def collate_fn(batch):
    waveforms = [item['waveform'] for item in batch]
    flatten_codes = [item['flatten_code'] for item in batch]
    lengths_waveform = torch.tensor([item['length_waveform'] for item in batch])
    lengths_code = torch.tensor([item['length_code'] for item in batch])
    filenames = [item['filename'] for item in batch]
    pitchs = [item['pitch'] for item in batch]

    max_len_code = max(lengths_code)
    max_len_wave = max(lengths_waveform)
    
    longest_pitch = len(max(pitchs, key=len))
    padded_pitch = [item['pitch'] + [0] * (longest_pitch - len(item['pitch'])) for item in batch]
    pitchs_tensor = torch.tensor(padded_pitch)

    # NOTE(yiwen) the flatten code has probably already been padded
    # pad flatten_code to the max length in a batch
    padded_flatten_code = torch.zeros(len(flatten_codes), max_len_code)
    flatten_codes = [torch.tensor(fc) for fc in flatten_codes]

    for i, fc in enumerate(flatten_codes):
        padded_flatten_code[i, :fc.shape[0]] = fc

    padded_waveforms = torch.zeros(len(waveforms), 1, max_len_wave)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, :, :w.shape[1]] = w

    return padded_flatten_code, padded_waveforms, lengths_waveform, lengths_code, filenames, pitchs



if __name__ == "__main__":

    test_label_file = "/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/dump/audio_raw_svs_acesinger/test/label"

    dataset = TrainDataset(audio_dir="./datasets/wav_dump/",
                            scp_root="/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/exp/speechlm_opuslm_v1_1.7B_anneal_ext_phone_finetune_svs/decode_tts_espnet_sampling_temperature0.8_finetune_40epoch/svs_test/log",
                            label_file=test_label_file,
                            ark_root= "/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1",
                            sample_rate=16000, 
                        )
                        
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)

    for flatten_code, waveforms, lengths_waveform, lengths_code, filenames, pitch in dataloader:
        # NOTE(yiwen) do not support batch inference (set bs=1, so no padding)

        print("Batch shape:", flatten_code.shape) 
        print("Filenames:", filenames)
        
        import pdb
        pdb.set_trace()
        # length: flatten to the max code ([1242, 1683, 1242, 3132])

        # NOTE (yiwen) 1. map waveform length to token length, 
        # 2. get feature mask based on token length, 
        # 3. use the mask on predicted loss
