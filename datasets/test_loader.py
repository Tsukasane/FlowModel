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



class TestDataset(Dataset):
    def __init__(self, scp_root, label_file, ark_root, sample_rate=16000):
        
        split_size = 1 # 32

        self.pitch_dict = {}
        self.flattenCode_dict = {}
        self.pitch_ls = []
        self.flattenCode_ls = []
        self.sample_rate = sample_rate
        self.label_file = label_file
        
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
        # debug TODO(yiwen) check what are these two scp
        # scp_path = '/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/exp/speechlm_naacl_demo_1.7B_lr5e-6/decode_tts_espnet_sampling_temperature0.8_finetune_47epoch/svs_test/log/output.1/wav.scp/rank0_token_wav.scp.scp'
        # scp_path = '/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/exp/speechlm_naacl_demo_1.7B_lr5e-6/decode_tts_espnet_sampling_temperature0.8_finetune_41epoch/svs_test/log/output.1/wav.scp/rank0_token_wav.scp.scp'
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
        pitch_key = "_".join(filename.split('_')[1:3]) + '.wav'
        pitch = self.pitch_dict[pitch_key]

        return {
            "flatten_code": flatten_code,              # shape: [1, T]
            "length": flatten_code.shape[0],       # T
            "filename": filename,
            "pitch": pitch
        }


def collate_fn(batch):
    flatten_codes = [item['flatten_code'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch])
    filenames = [item['filename'] for item in batch]
    pitchs = [item['pitch'] for item in batch]

    max_len = max(lengths)

    # pad the pitch to the max length to a batch
    # longest_pitch = len(max(pitchs, key=len))
    # padded_pitch = [item['pitch'] + [0] * (longest_pitch - len(item['pitch'])) for item in batch]
    # pitchs_tensor = torch.tensor(padded_pitch)

    # NOTE(yiwen) the flatten code has probably already been padded
    # pad flatten_code to the max length in a batch
    padded_flatten_code = torch.zeros(len(flatten_codes), max_len)
    flatten_codes = [torch.tensor(fc) for fc in flatten_codes]

    for i, fc in enumerate(flatten_codes):
        padded_flatten_code[i, :fc.shape[0]] = fc

    return padded_flatten_code, lengths, filenames, pitchs



if __name__ == "__main__":
    dataset = TestDataset(scp_root="/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/exp/speechlm_naacl_demo_1.7B_lr5e-6/decode_tts_espnet_sampling_temperature0.8_finetune_70epoch/svs_test/log",
                            ark_root= "/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1",
                            sample_rate=16000, 
                            )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

    for flatten_code, lengths, filenames, pitch in dataloader:
        # NOTE(yiwen) do not support batch inference (set bs=1, so no padding)
        import pdb
        pdb.set_trace()

        print("Batch shape:", flatten_code.shape) 
        print("Filenames:", filenames)

        # length: flatten to the max code ([1242, 1683, 1242, 3132])

        # NOTE (yiwen) 1. map waveform length to token length, 
        # 2. get feature mask based on token length, 
        # 3. use the mask on predicted loss
