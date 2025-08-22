# Copyright (c) 2024 Alibaba Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications:
# - Modified by Yiwen Zhao, 2025-08-09: adapt to SLMSVS

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchaudio.transforms as T
from omegaconf import DictConfig
from utils.mask import make_pad_mask
from modules.music_tokenizer.vqvae import VQVAE
from modules.transformer.encoder import ConformerEncoder
from modules.CFM.conditionalCFM import ConditionalCFM
from modules.CFM.conditional_decoder import ConditionalDecoder
from modules.CFM.length_regulator import InterpolateRegulator
from datasets.dataloader import AudioDataset
from datasets.test_loader import TestDataset
from datasets.train_loader import TrainDataset
from torch.utils.data import DataLoader

from modules.codec_ssl_tokenizer.codec_tokenizer import get_codec_tokenizer
import soundfile as sf
import kaldiio

import logging
logging.basicConfig(level=logging.INFO)

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
import torchaudio


def lr_lambda(step):
    if step < start_decay:
        return 1.0
    elif step < end_decay:
        return 1.0 - (step - start_decay) / (end_decay - start_decay) * (1 - final_lr / initial_lr)
    else:
        return final_lr / initial_lr


def set_seed(seed=42):
    # Python built-in random
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch (CPU and CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # For reproducibility in dataloaders with worker_init_fn
    os.environ['PYTHONHASHSEED'] = str(seed)


class MaskedDiff(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 128,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 encoder: torch.nn.Module = None,
                 length_regulator: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 codec_model: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 128, 'sampling_rate': 48000,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 48000},
                num_codebooks: int = 4,
                device: str = 'cpu',
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")

        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss
        self.dequantizer = codec_model
        self.dequantizer.eval()
        self.num_codebooks  = num_codebooks
        self.cond = None
        self.interpolate = False

        self.linear_cond_pitch = nn.Linear(1, 512).to(device)
        
                                  
    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            - batch:
                - codec_continuous_feats: clean codec embedding
                - codec_feats_len: length of clean codec embedding
                - pitch: pitch B, T
                - source_codec_feats: noisy codec embedding B, T, D
                - mel_spectrogram: mel-spectrogram
        """

        # codec_continuous_feats = batch['codec_continuous_feats']  # 32, 349, 512
        mel_spectrogram = batch['mel_spectrogram'].squeeze(1) # B, 1, 80, T' --> B, 80, T'

        codec_token_len = batch['codec_feats_len']
        pitch = batch['pitch']
        source_codec_feats = batch['source_codec_feats']

        h, h_lengths = self.encoder(source_codec_feats, codec_token_len) # in_features=512   32, 349, 512,    32, 1, 349 bool according to codec token len
        h, h_lengths = self.length_regulator(h, codec_token_len)

        # ori, just give part of the feats as conditions
        # if self.cond:
        #     conds = torch.zeros(feat.shape, device=token.device)
        #     for i, j in enumerate(feat_len):
        #         if random.random() < 0.5:
        #             continue
        #         index = random.randint(0, int(0.3 * j))
        #         conds[i, :index] = feat[i, :index]
        #     conds = conds.transpose(1, 2)
        # else:
        #     conds = None

        # conds is directly cat to x
        mask = (~make_pad_mask(codec_token_len)).to(h) # TODO(yiwen) check the meaning of .to(h1)

        # conds = None
        conds = pitch
        conds = conds.unsqueeze(-1).to(device).float()
        conds = self.linear_cond_pitch(conds).transpose(1,2)

        if conds.shape[2] != h.shape[1]:
            logging.warning(f"conds shape {conds.shape} does not match h shape {h.shape}, conds will be resized")
            conds = F.interpolate(conds, size=h.shape[1], mode='linear', align_corners=False)

        spk_prompt = batch['spk_mel'].squeeze(1)
        loss, _ = self.decoder.compute_loss( 
                x1=mel_spectrogram, # target
                mask=mask.unsqueeze(1), # mask
                mu=h.transpose(1, 2).contiguous(), # the encoder output (latent)
                spks=spk_prompt,
                cond=conds,)

        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  sample_rate,
                  pitch=None,
                  spk_prompt=None,
                  device='cuda:0'):
        """
        Args:
            - h: the encoded noisy codec condition
        """
        assert token.shape[0] == 1

        # flatten_code_embedding, lengths, pitch

        # token.shape [1, 185, 512]
        h, _ = self.encoder(token, token_len)
        h, _ = self.length_regulator(h, token_len)  

        # get conditions
        # conds = None
        conds = pitch
        conds = conds.unsqueeze(-1).to(device).float()
        conds = self.linear_cond_pitch(conds).transpose(1,2)

        if conds.shape[2] != h.shape[1]:
            logging.warning(f"conds shape {conds.shape} does not match h shape {h.shape}, conds will be resized")
            conds = F.interpolate(conds, size=h.shape[1], mode='linear', align_corners=False)

        mask = (~make_pad_mask(token_len)).to(h)

        feat = self.decoder( # NOTE(yiwen) the output is in the same shape as mu
            mu=h.transpose(1, 2).contiguous(), # the input is the source codec token embedding
            mask=mask.unsqueeze(1),
            spks=spk_prompt,
            cond=conds,
            n_timesteps=10
        ) # get the feature, then pass to the vocoder

        feat = feat.transpose(1,2)
        # print(f'debug -- feat.shape {feat.shape}') # 1, 512, 269

        return feat


def init_modules(device, batch_size=32):

    # -------- configs -------- #
    class DummyCFMParams:
        sigma_min=1e-06
        solver='euler'
        t_scheduler='cosine'
        training_cfg_rate=0.2
        inference_cfg_rate=0.7
        reg_loss_type='l1'
    cfm_params = DummyCFMParams()

    # -------- modules -------- #
    encoder = ConformerEncoder(
        input_size=512,
        output_size=512,
        attention_heads=4,
        linear_units=1024,
        num_blocks=3,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        normalize_before=True,
        input_layer='linear',
        pos_enc_layer_type='rel_pos_espnet',
        selfattention_layer_type='rel_selfattn',
        use_cnn_module=False,
        macaron_style=False,
    )

    codec_model = get_codec_tokenizer(device)

    cond_channel = 512
    in_channeldim = 80 + 512 + cond_channel # mel + codec + pitch
    decoder_estimator = ConditionalDecoder(
        in_channels=in_channeldim, # NOTE(yiwen) x || cond    1536 for codec embedding  
        out_channels=80, # NOTE(yiwen) 512 for codec embedding
        channels=[256, 256],
        dropout=0.0,
        attention_head_dim=64,
        n_blocks=4,
        num_mid_blocks=8,
        num_heads=8,
        act_fn='gelu',
    )

    decoder = ConditionalCFM(
        in_channels=240, 
        cfm_params=cfm_params, 
        estimator=decoder_estimator,
    )

    length_regulator = InterpolateRegulator(
        channels=512,
        sampling_ratios=[1, 1, 1, 1]
    )

    return encoder.to(device), length_regulator.to(device), decoder.to(device), codec_model.eval()


def waveform_to_mel(
    waveform: torch.Tensor,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 320,
    win_length: int = 1024,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float = 8000,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Convert waveform to log-Mel spectrogram using pure PyTorch.

    Args:
        waveform: (Tensor) shape (1, T) or (T,)
        sr: sampling rate
        device: "cpu" or "cuda"
    
    Returns:
        mel: (n_mels, T')
    """

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (1, T)

    mel_spec_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
        power=2.0,
    ).to(device)

    db_transform = T.AmplitudeToDB(stype="power").to(device) # this operation causes -100

    mel_spec = mel_spec_transform(waveform)  # (B, 1, n_mels, T')
    log_mel_spec = db_transform(mel_spec)    # (B, 1, n_mels, T')
    
    return log_mel_spec.squeeze(0)  # (B, n_mels, T')


def delete_zeros_from_segments(lst, num_to_delete=1):
    if num_to_delete == -1: # no 0
        return lst
    result = []
    i = 0
    n = len(lst)
    while i < n:
        if lst[i] != 0:
            result.append(lst[i])
            i += 1
        else:
            start = i
            while i < n and lst[i] == 0:
                i += 1
            segment_len = i - start
            remaining_zeros = [0] * max(0, segment_len - num_to_delete)
            result.extend(remaining_zeros)
    return result


def find_shortest_zero_segment(lst):
    min_len = float('inf')
    min_pos = None
    i = 0
    n = len(lst)
    while i < n:
        if lst[i] == 0:
            start = i
            while i < n and lst[i] == 0:
                i += 1
            seg_len = i - start
            if seg_len > 1 and seg_len < min_len:
                min_len = seg_len
                min_pos = start
        else:
            i += 1
    if min_pos is not None:
        return min_len
    else:
        return -1


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


def get_codec(codec_model, waveform, length):
    '''
    Args:
        - codec_model: pretrained codec tokenizer
        - length: waveform sample number
    Return:
        - codec_continuous_feats (tensor): detokenized continuous codecfeatures (B, T, D)
        - codec_feats_len (tensor): frame-wise length (B)
    '''
    
    batch = {}
    codec_hop_size = 320

    # NOTE(yiwen) this is from discrete ids (/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/espnet2/gan_codec/dac/dac.py)
    with torch.no_grad():
        flatten_codes, _ = codec_model(waveform)
        # print(f"flatten_codes", flatten_codes) #torch.Size([1, 1536])
        codec_continuous_feats = codec_model.detokenize(flatten_codes).transpose(1, 2)

    # feats length
    codec_token_len = (length+codec_hop_size-1) / codec_hop_size 
    codec_token_len = codec_token_len.to(int).to(device)
    # codec_token_len = torch.tensor([codec_continuous_feats.shape[1]]).to(device)
    batch['codec_continuous_feats'] = codec_continuous_feats
    batch['codec_feats_len'] = codec_token_len

    return batch


def visualize_mel(log_mel_spec, save_name):
    '''
    Args: B, 1, 80, T'
    '''
    mel = log_mel_spec[0][0].cpu().numpy()  # 取第一个样本，去掉 batch 和 channel

    plt.figure(figsize=(10, 4))
    plt.imshow(mel, origin='lower', aspect='auto', cmap='magma')
    plt.title("Log-Mel Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency Bin")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()

def train_one_epoch(model, 
                    optimizer, 
                    scheduler, 
                    train_data_loader, 
                    codec_model,
                    device,
                    epoch_id=0,
                    codec_per_frame=8,
                    ssl_per_frame=1,
                    embedding_dim=512,
                    save_path='./pitchandspkprompts_l.pth'):
        ''' Train one epoch
        Args:
            - flatten_token: noisy codec token
            - batch_waveforms: clean waveform
            - lengths_code: discrete noisy flatten codec token length
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} '.format(epoch_id+1, lr))
    
        total_loss = 0
        num_batch = 0

        num_batch_per_epoch = len(train_data_loader)

        codec_model.to(device)
        codec_model.eval()
        token_per_frame = codec_per_frame + ssl_per_frame

        vis_mel = True # visualize the first element at the start of each epoch

        for flatten_token, batch_waveforms, lengths_waveform, lengths_code, filenames, pitch, spk_prompt in train_data_loader:
            num_batch += 1
            input_batch = {}
            
            # NOTE GET CONDITION: discrete codec token to embedding
            lengths_code = lengths_code.to(device)
            flatten_code_embeddingls = []
            code_length_all = []
            max_code_length = 0

            with torch.no_grad():
                for i, fc in enumerate(flatten_token):
                    fc = fc.to(device).to(int) # already in range of codec, no padding, in shape [1, 1352]
                    flatten_code_embedding = codec_model.detokenize(fc.unsqueeze(0)).transpose(1, 2)
                    code_length_all.append(flatten_code_embedding.shape[1])
                    if flatten_code_embedding.shape[1]>max_code_length:
                        max_code_length = flatten_code_embedding.shape[1]                 
                    flatten_code_embeddingls.append(flatten_code_embedding)
            input_batch['codec_feats_len'] = torch.tensor(code_length_all).to(device) # valid continous codec embedding length for all elements in one batch

            # NOTE unify the length of the condition (noisy codec embeddings)
            padded_flatten_code = torch.zeros(len(flatten_token), max_code_length, embedding_dim)
            for i, fc in enumerate(flatten_code_embeddingls):
                padded_flatten_code[i:i+1, :fc.shape[1],:] = fc
            flatten_code_embedding = padded_flatten_code[:,:max_code_length,:]
            lengths = lengths_code // 8 # code_per_frame
            input_batch['source_codec_feats'] = flatten_code_embedding.to(device) # B, T_code, embedding_dim
            
            # NOTE GET TARGET: clean waveform to mel-spectrogram
            batch_waveforms = batch_waveforms.to(device) # 32, 1, 116718
            input_batch['mel_spectrogram'] = waveform_to_mel(batch_waveforms, device=device) # batchMEL
            mel_T = get_batch_mel_lengths(lengths_waveform) # a list, mel T length of each waveform sample in a batch
            mel_max_length = mel_T.max().item()

            batch_spk_prompt = spk_prompt.to(device)
            input_batch['spk_mel'] = waveform_to_mel(batch_spk_prompt, device=device) # fix length [B, 1, 80, 151]

            if vis_mel:
                visualize_mel(input_batch['mel_spectrogram'], f"GT_mel{num_batch}.png")
                vis_mel=False

            # NOTE align target (B, 1, 80, T) to source (B, T, 512)
            B, T = input_batch['source_codec_feats'].shape[0], input_batch['source_codec_feats'].shape[1]
            T_mel = input_batch['mel_spectrogram'].shape[-1]
            mel_modified = -100 + torch.zeros([B, 1, 80, T]).to(device) # 0 padding in waveform after wavetomel, becomes -100 (TODO confirm)
            minT = min(T, T_mel)
            mel_modified[:,:,:,:minT] = input_batch['mel_spectrogram'][:,:,:,:minT]
            
            input_batch['mel_spectrogram'] = mel_modified
            input_batch['source_codec_feats'] = input_batch['source_codec_feats']
            
            processed_pitch = []

            for pc in pitch:
                '''
                NOTE(yiwen) basicly aligned, super long song / short phn / round in duration may cause mismatch
                if pitch ls too short, do padding
                if pitch ls too long, del zero first (each segment del min 0 segment length)
                    then assume redundant zeros are all at the tail, cut them

                在acesinger中，实际很多情况音频比note更长
                '''
                # print(f'debug -- difference {len(pc)-minT}')
                if len(pc)-minT>20: # pitch_ls too long
                    del_length = find_shortest_zero_segment(pc)
                    pc = delete_zeros_from_segments(pc, del_length)
                if len(pc)<minT:
                    updated_pitch = pc + [0] * (minT - len(pc)) 
                    processed_pitch.append(updated_pitch)        
                elif len(pc)==minT:
                    processed_pitch.append(pc)
                else:
                    updated_pitch = pc[:minT]
                    processed_pitch.append(updated_pitch)

            # padded to maxpitch in preprocessing
            pitch_tensors = torch.tensor(processed_pitch)
            input_batch['pitch'] = pitch_tensors.to(device)


            # NOTE(yiwen) update models
            optimizer.zero_grad()
            output_loss_dic = model(input_batch, device)
            
            batch_loss = output_loss_dic['loss']
            if num_batch%10==0:
                print(f'batch {num_batch} / total {num_batch_per_epoch} -- batch loss {batch_loss.item()}')
            if num_batch%100==0:
                writer.add_scalar('Loss/train_batch', batch_loss.item(), epoch_id * len(train_data_loader) + num_batch) # cal by iteration
            
            total_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
        
            scheduler.step()

        total_loss /= num_batch
        logging.info(f'Training Loss for Epoch {epoch_id}: {total_loss}')

        if os.path.exists(save_path):
            os.rename(save_path, './pitchandspkprompts_b.pth')
        torch.save({
            'epoch': epoch_id,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss, # only the number with no gradient
            }, save_path)


def collate_fn_test(batch):

    flatten_codes = [item['flatten_code'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch])
    filenames = [item['filename'] for item in batch]
    pitchs = [item['pitch'] for item in batch]
    spk_prompts = [item['raw_spk_prompt'] for item in batch]

    max_len = max(lengths)

    # pad flatten_code to the max length in a batch
    # TODO(yiwen) check the padding id of flatten code (-62670)
    padded_flatten_code = torch.zeros(len(flatten_codes), max_len)
    flatten_codes = [torch.tensor(fc) for fc in flatten_codes]

    for i, fc in enumerate(flatten_codes):
        padded_flatten_code[i, :fc.shape[0]] = fc

    raw_spk_prompt = torch.stack(spk_prompts, dim=0)
    return padded_flatten_code, lengths, filenames, pitchs, raw_spk_prompt


def valid(valid_data_loader):
    pass


def save_h5(feats, save_path):
    '''
    feats: 80, T
    '''
    feats = feats.transpose(0,1).detach().cpu().numpy()
    import h5py
    with h5py.File(save_path, "w") as f:
        f.create_dataset("feats", data=feats)
    print(F"Saved to {save_path}")


def inference(model,
              dataloader,
              codec_model,
              device='cpu',
              vocoder=None,
              vocoder_name="hifigan",
              codec_per_frame=8,
              ssl_per_frame=1,
              sample_rate=16000,
              output_dir='./output_melh5',
              visualize=False): 
    '''
    Inputs:
        - pitch(list): as condition
        - noise z / noisy codec embedding
    Outputs:
        - clean codec embedding
        then waveform = self.decoder(clean_codec_embedding)
    '''
    codec_model.to(device)
    codec_model.eval()
    token_per_frame = codec_per_frame + ssl_per_frame

    current_cnt = 0
    for flatten_token, lengths, filenames, pitch, raw_spk_prompt in dataloader:
        # codec id to embedding
        flatten_codec = flatten_token.to(device).to(int) # already in range of codec, no padding, in shape [1, 1352]
        lengths = lengths.to(device)

        current_cnt += 1
        print(f"Processing -- {current_cnt}/{len(dataloader)}, {filenames}")
        with torch.no_grad():
            flatten_code_embedding = codec_model.detokenize(flatten_codec.to(int)).transpose(1, 2) # 1, 159, 512
        
        # valid flatten length --> valid codec embedding
        lengths = lengths // 8 # code_per_frame
        
        minT = flatten_code_embedding.shape[1]
        processed_pitch = []

        raw_spk_prompt = raw_spk_prompt.squeeze().to(device)
        spk_prompt = waveform_to_mel(raw_spk_prompt, device=device).unsqueeze(0) # fix length [B, 1, 80, 151]

        # NOTE(yiwen) better pitch alignment in inference
        for pc in pitch:
            '''
            if pitch ls too short, do padding
            if pitch ls too long, del zero first (each segment del min 0 segment length)
                then assume redundant zeros are all at the tail, cut them
            '''
            if len(pc)-minT>20: # pitch_ls too long
                del_length = find_shortest_zero_segment(pc)
                pc = delete_zeros_from_segments(pc, del_length)
            if len(pc)<minT:
                updated_pitch = pc + [0] * (minT - len(pc)) 
                processed_pitch.append(updated_pitch)        
            elif len(pc)==minT:
                processed_pitch.append(pc)
            else:
                updated_pitch = pc[:minT]
                processed_pitch.append(updated_pitch)
        pitch_tensors = torch.tensor(processed_pitch)
        
        output_feats = model.inference(flatten_code_embedding, lengths, sample_rate, pitch_tensors, spk_prompt)
        output_feats = output_feats.transpose(-1,-2).unsqueeze(1)

        if visualize:
            visualize_mel(output_feats, "Pred_mel.png") # B, 1, T', 80
        
        save_path = os.path.join(output_dir, f"{filenames[0]}.h5") # e.g. ./output_melh5/svs_acesinger_17#2086003208_sample1.h5
        
        # save mel in .h5 to a specific folder
        save_feats = output_feats.squeeze()
        save_h5(save_feats, save_path)


def get_batch_mel_lengths(
    lengths,
    n_fft=1024,
    hop_length=320,
    win_length=1024,
    center=True
):
    """
    Compute mel frame lengths for a batch of waveform lengths.

    Args:
        lengths: list or 1D tensor of waveform lengths (in samples)
        n_fft: FFT size
        hop_length: hop size
        win_length: window size
        center: whether to pad input (default True, as in torchaudio)
    Returns:
        1D tensor of mel frame lengths
    """
    if not torch.is_tensor(lengths):
        lengths = torch.tensor(lengths)
    pad = n_fft // 2 if center else 0
    T_padded = lengths + 2 * pad
    mel_lengths = 1 + torch.clamp((T_padded - win_length) // hop_length, min=0)
    return mel_lengths


if __name__=='__main__':

    set_seed(42)

    # TODO(yiwen) add some visualization
                # tSNE, mel spectrogram
    device = "cuda:0"
    valid_step = 10
    total_epochs = 30

    batch_size = 64
    train = True
    train_label_file = "/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/dump/audio_raw_svs_acesinger/tr_no_dev/label"
    test_label_file = "/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/dump/audio_raw_svs_acesinger/test/label"
    output_dir = './output_melh5_cond_spkp_pitch'

    latest_model_file = "/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/flow_model/pitchandspkprompts_l.pth"

    os.makedirs(output_dir, exist_ok=True)

    initial_lr = 3e-4
    final_lr = 1e-4
    start_decay = 200_000
    end_decay = 500_000

    vocoder_name = "hifigan"

    # ------ load models ------ #
    encoder, length_regulator, decoder, codec_model = init_modules(device, batch_size) # already loaded to the device
    maskedDiff = MaskedDiff(
        encoder=encoder,
        length_regulator=length_regulator,
        decoder=decoder,
        codec_model=codec_model,
        input_size=256,
        output_size=80,
        output_type='mel',
        vocab_size=4096,
        input_frame_rate=75,
        only_mask_loss=True,
        device=device,
    )

    optimizer = torch.optim.AdamW(maskedDiff.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    if train:
        writer = SummaryWriter()

        ### train dataloader
        train_dataset = TrainDataset(audio_dir="./datasets/wav_dump/",
                                        scp_root="/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/exp/speechlm_opuslm_v1_1.7B_anneal_ext_phone_finetune_svs/decode_tts_espnet_sampling_temperature0.8_finetune_68epoch/svs_tr_no_dev/log",
                                        label_file=train_label_file,
                                        ark_root= "/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1",
                                        sample_rate=16000
                                        )

        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=True,) # NOTE(yiwen) align batch with in_channel at flow.py NOTE 1
        pretrain_epoch = 0

        if os.path.exists(latest_model_file):
            checkpoint = torch.load(latest_model_file, weights_only=True)
            maskedDiff.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            pretrain_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            logging.info(f"Resume from {latest_model_file}, epoch {pretrain_epoch}")

        for epoch in range(pretrain_epoch, total_epochs):
            maskedDiff.train()
            train_one_epoch(maskedDiff, optimizer, scheduler, train_data_loader, codec_model, device, epoch)
        
    else:
        ### test dataloader (need flatten code, need corresponding pitch)
        test_dataset = TestDataset(scp_root="/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/exp/speechlm_opuslm_v1_1.7B_anneal_ext_phone_finetune_svs/decode_tts_espnet_sampling_temperature0.8_finetune_68epoch/svs_test/log",
                                    label_file=test_label_file,
                                    ark_root= "/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1",
                                    sample_rate=16000, 
                                    )
        # batchsize=1 in inference
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn_test)

        checkpoint = torch.load(latest_model_file, weights_only=True)
        maskedDiff.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        maskedDiff.eval()
        inference(maskedDiff, test_dataloader, codec_model, device, output_dir=output_dir)

