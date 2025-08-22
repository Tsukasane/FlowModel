import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
import random
import torchaudio.transforms as T
import kaldiio
from glob import glob

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


        for i in range(split_size):
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

    def get_spkprompt(self, current_key, base_dir, sample_rate, min_duration=3.0):
        """
        randomly select a waveform from the same singer,
        make sure the length>3s
        """
        singer_prefix = current_key.split('#')[0] + '#'
        current_path = os.path.join(base_dir, f"{current_key}.wav")
        
        pattern = os.path.join(base_dir, f"{singer_prefix}*.wav")
        candidates = [c for c in glob(pattern) if os.path.abspath(c) != os.path.abspath(current_path)]
        
        if not candidates:
            return None, None
        
        while candidates:
            candidate_path = random.choice(candidates)
            waveform, sr = sf.read(candidate_path) 
            duration = waveform.shape[0] / sr
            if duration >= min_duration:
                waveform = waveform[:int(sr*min_duration)]
                waveform = torch.from_numpy(waveform).float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)  # [1, T]
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                    waveform = resampler(waveform)
                return waveform
            else:
                candidates.remove(candidate_path)
        
        return None, None 

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

        spk_prompt = self.get_spkprompt(wav_filename, self.audio_dir, self.sample_rate) # [1, 48000]

        return {
            "flatten_code": flatten_code,              # shape: [1, T]
            "waveform": waveform,
            "length_waveform": waveform.shape[1], 
            "length_code": flatten_code.shape[0],       # T
            "filename": filename,
            "pitch": pitch,
            "spk_prompt": spk_prompt
        }


def collate_fn(batch):
    waveforms = [item['waveform'] for item in batch]
    flatten_codes = [item['flatten_code'] for item in batch]
    lengths_waveform = torch.tensor([item['length_waveform'] for item in batch])
    lengths_code = torch.tensor([item['length_code'] for item in batch])
    filenames = [item['filename'] for item in batch]
    pitchs = [item['pitch'] for item in batch]
    spk_prompt = [item['spk_prompt'] for item in batch]

    max_len_code = max(lengths_code)
    max_len_wave = max(lengths_waveform)
    
    # NOTE(yiwen) the flatten code has probably already been padded
    # pad flatten_code to the max length in a batch
    # padded_flatten_code = torch.zeros(len(flatten_codes), max_len_code)
    flatten_codes = [torch.tensor(fc) for fc in flatten_codes]

    # for i, fc in enumerate(flatten_codes):
    #     padded_flatten_code[i, :fc.shape[0]] = fc

    padded_waveforms = torch.zeros(len(waveforms), 1, max_len_wave)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, :, :w.shape[1]] = w

    batch_spk_prompt = torch.stack(spk_prompt, dim=0)
    return flatten_codes, padded_waveforms, lengths_waveform, lengths_code, filenames, pitchs, batch_spk_prompt




if __name__ == "__main__":

    train_label_file = "/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/dump/audio_raw_svs_acesinger/tr_no_dev/label"

    dataset = TrainDataset(audio_dir="./datasets/wav_dump/",
                                scp_root="/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/exp/speechlm_opuslm_v1_1.7B_anneal_ext_phone_finetune_svs/decode_tts_espnet_sampling_temperature0.8_finetune_68epoch/svs_tr_no_dev/log",
                                label_file=train_label_file,
                                ark_root= "/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1",
                                sample_rate=16000
                                )
                        
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)

    for flatten_code, waveforms, lengths_waveform, lengths_code, filenames, pitch, spk_prompt in dataloader:
        # NOTE(yiwen) do not support batch inference (set bs=1, so no padding)

        
        import pdb
        pdb.set_trace()
        # length: flatten to the max code ([1242, 1683, 1242, 3132])

        # NOTE (yiwen) 1. map waveform length to token length, 
        # 2. get feature mask based on token length, 
        # 3. use the mask on predicted loss
