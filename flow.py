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
from torch.utils.data import DataLoader

from modules.codec_ssl_tokenizer.codec_tokenizer import get_codec_tokenizer
import soundfile as sf
import kaldiio

import logging
logging.basicConfig(level=logging.INFO)

from torch.utils.tensorboard import SummaryWriter
import numpy as np


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

        codec_continuous_feats = batch['codec_continuous_feats']  # 32, 349, 512
        codec_token_len = batch['codec_feats_len']
        pitch = batch['pitch']

        h, h_lengths = self.encoder(codec_continuous_feats, codec_token_len) # in_features=512   32, 349, 512,    32, 1, 349 bool according to codec token len
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
        # TODO(yiwen) stretch the pitch according to codec_token_len (different among samples in a batch)
        conds = pitch # B, T

        mask = (~make_pad_mask(codec_token_len)).to(h) # TODO(yiwen) check the meaning of .to(h1)
    
        conds = conds.unsqueeze(-1).to(device).float()
        conds = self.linear_cond_pitch(conds).transpose(1,2)

        loss, _ = self.decoder.compute_loss( 
                codec_continuous_feats.transpose(1,2), # the target for flow
                mask.unsqueeze(1),
                h.transpose(1, 2).contiguous(), # the encoder output (latent)
                None,
                cond=conds
        )

        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  sample_rate,
                  pitch,
                  device='cuda:0'):
        assert token.shape[0] == 1

        # flatten_code_embedding, lengths, pitch

        # token.shape [1, 185, 512]
        h, _ = self.encoder(token, token_len)
        h, _ = self.length_regulator(h, token_len)  

        # get conditions
        conds = pitch # pitch
        conds = conds.unsqueeze(-1).to(device).float()
        conds = self.linear_cond_pitch(conds).transpose(1,2)

        mask = (~make_pad_mask(token_len)).to(h)
        feat = self.decoder( # NOTE(yiwen) if not directly the source dis, pass t?
            mu=h.transpose(1, 2).contiguous(), # the input is the source codec token embedding
            mask=mask.unsqueeze(1),
            spks=None,
            cond=conds,
            n_timesteps=10
        ) # get the feature, then pass to the vocoder

        feat = feat.transpose(1,2)
        # print(f'debug -- feat.shape {feat.shape}') # 1, 512, 269

        return feat


def init_modules(device):

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

    decoder_estimator = ConditionalDecoder(
        in_channels=1536, # x || cond
        out_channels=512,
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


# TODO(yiwen) check mel is in the same downsample rate as codec model
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

    db_transform = T.AmplitudeToDB(stype="power").to(device)

    mel_spec = mel_spec_transform(waveform)  # (1, n_mels, T')
    log_mel_spec = db_transform(mel_spec)    # (1, n_mels, T')

    return log_mel_spec.squeeze(0)  # (n_mels, T')


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
    # padding to max length in a batch
    waveforms = [item['waveform'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch])
    filenames = [item['filename'] for item in batch]
    pitchs = [item['pitch'] for item in batch]

    max_len = max(lengths)

    padded_waveforms = torch.zeros(len(waveforms), 1, max_len)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, :, :w.shape[1]] = w

    return padded_waveforms, lengths, filenames, pitchs



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


def train_one_epoch(model, 
                    optimizer, 
                    scheduler, 
                    train_data_loader, 
                    codec_model,
                    device,
                    epoch_id=0,
                    save_path='./latest_pitch_new.pth'):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} '.format(epoch_id+1, lr))
    
        total_loss = 0
        num_batch = 0

        num_batch_per_epoch = len(train_data_loader)

        for batch_waveforms, lengths, filenames, pitch in train_data_loader:
            
            num_batch += 1
            batch_waveforms = batch_waveforms.to(device) # 32, 1, 116718
            input_batch = get_codec(codec_model, batch_waveforms, lengths) # codec embedding

            codec_T = input_batch['codec_continuous_feats'].shape[1]

            processed_pitch = []

            # NOTE(yiwen) better aligned pitch
            for pc in pitch:
                '''
                NOTE(yiwen) basicly aligned, super long song / short phn / round in duration may cause mismatch
                if pitch ls too short, do padding
                if pitch ls too long, del zero first (each segment del min 0 segment length)
                    then assume redundant zeros are all at the tail, cut them

                在acesinger中，实际很多情况音频比note更长
                '''
                # print(f'debug -- difference {len(pc)-codec_T}')
                if len(pc)-codec_T>20: # pitch_ls too long
                    del_length = find_shortest_zero_segment(pc)
                    pc = delete_zeros_from_segments(pc, del_length)
                if len(pc)<codec_T:
                    updated_pitch = pc + [0] * (codec_T - len(pc)) 
                    processed_pitch.append(updated_pitch)        
                elif len(pc)==codec_T:
                    processed_pitch.append(pc)
                else:
                    updated_pitch = pc[:codec_T]
                    processed_pitch.append(updated_pitch)
            
    
            pitch_tensors = torch.tensor(processed_pitch)
        
            input_batch['pitch'] = pitch_tensors.to(device)

            optimizer.zero_grad()
            output_loss_dic = model(input_batch, device)
            
            batch_loss = output_loss_dic['loss']
            if num_batch%10==0:
                print(f'batch {num_batch} / total {num_batch_per_epoch} -- batch loss {batch_loss.item()}')
            if num_batch%100==0:
                writer.add_scalar('Loss/train_batch', batch_loss.item(), epoch_id * len(train_data_loader) + num_batch) # cal by iteration
            # if num_batch==100: # for debug
            #     break
            total_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
        
        scheduler.step()

        total_loss /= num_batch
        logging.info(f'Training Loss for Epoch {epoch_id}: {total_loss}')

        # TODO(yiwen) save model (is able to resume on)
        # if epoch_id % 2 ==0:
        if os.path.exists(save_path):
            os.rename(save_path, './latest_bu_pitch.pth')
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

    max_len = max(lengths)

    # pad flatten_code to the max length in a batch
    # TODO(yiwen) check the padding id of flatten code (-62670)
    padded_flatten_code = torch.zeros(len(flatten_codes), max_len)
    flatten_codes = [torch.tensor(fc) for fc in flatten_codes]

    for i, fc in enumerate(flatten_codes):
        padded_flatten_code[i, :fc.shape[0]] = fc

    return padded_flatten_code, lengths, filenames, pitchs


def valid(valid_data_loader):
    pass


def inference(model,
              dataloader,
              codec_model,
              device='cpu',
              codec_per_frame=8,
              ssl_per_frame=1,
              sample_rate=16000,
              output_dir='./output'): 
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

    for flatten_token, lengths, filenames, pitch in dataloader:
        # codec id to embedding
        flatten_codec = flatten_token.to(device).to(int) # already in range of codec, no padding, in shape [1, 1352]
        lengths = lengths.to(device)

        with torch.no_grad():
            flatten_code_embedding = codec_model.detokenize(flatten_codec.to(int)).transpose(1, 2) # 1, 159, 512
        
        print(f'debug -- flatten_code_embedding.shape {flatten_code_embedding.shape}')
        # valid flatten length --> valid codec embedding
        lengths = lengths // 8 # code_per_frame
        
        codec_T = flatten_code_embedding.shape[1]
        processed_pitch = []

        # NOTE(yiwen) better pitch alignment in inference
        for pc in pitch:
            '''
            if pitch ls too short, do padding
            if pitch ls too long, del zero first (each segment del min 0 segment length)
                then assume redundant zeros are all at the tail, cut them
            '''
            if len(pc)-codec_T>20: # pitch_ls too long
                del_length = find_shortest_zero_segment(pc)
                pc = delete_zeros_from_segments(pc, del_length)
            if len(pc)<codec_T:
                updated_pitch = pc + [0] * (codec_T - len(pc)) 
                processed_pitch.append(updated_pitch)        
            elif len(pc)==codec_T:
                processed_pitch.append(pc)
            else:
                updated_pitch = pc[:codec_T]
                processed_pitch.append(updated_pitch)
        pitch_tensors = torch.tensor(processed_pitch)
        
        output_feats = model.inference(flatten_code_embedding, lengths, sample_rate, pitch_tensors)

        with torch.no_grad():
            waveform = codec_model.decode_continuous(output_feats).squeeze(1).cpu().numpy()
            # waveform = codec_model.decode_continuous(flatten_code_embedding).squeeze(1).cpu().numpy() # debug, decode the original codec embedding

        save_name = os.path.join(output_dir, f'save_{filenames[0]}.wav')
        sf.write(save_name, waveform.T, samplerate=sample_rate)


if __name__=='__main__':

    set_seed(42)

    # TODO(yiwen) larger batch size, more data?
                # can make only some of them as conditional data, and also do cfg in some portion.
                # if no supervision, only need waveform
    # TODO(yiwen) check whether need to provide t in inference 
                # (because the vector field is time dependent, and dirty distribution is not the orignal source distribution (Gaussian) in training)
    # TODO(yiwen) add some visualization
                # tSNE, mel spectrogram
    device = "cuda:0"
    valid_step = 10
    total_epochs = 100

    batch_size = 32
    train = False
    # train_label_file = "/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/kising/speechlm1/dump/audio_raw_svs_kising/tr_no_dev/label" # NOTE(yiwen) debugging
    # test_label_file = "/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/kising/speechlm1/dump/audio_raw_svs_kising/eval/label"
    train_label_file = "/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/dump/audio_raw_svs_acesinger/tr_no_dev/label" # NOTE(yiwen) debugging
    test_label_file = "/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/dump/audio_raw_svs_acesinger/test/label"
    output_dir = './output'

    latest_model_file = "/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/flow_model/latest_pitch_new.pth"

    os.makedirs(output_dir, exist_ok=True)

    # ------ load models ------ #
    encoder, length_regulator, decoder, codec_model = init_modules(device) # already loaded to the device
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

    # NOTE(yiwen) temp choice
    optimizer = torch.optim.AdamW(maskedDiff.parameters(), lr=0.0005, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)


    if train:
        writer = SummaryWriter()

        ### train dataloader
        train_dataset = AudioDataset("./datasets/wav_dump/", 
                                        transform=None, 
                                        sample_rate=16000, 
                                        label_file=train_label_file)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
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
        
        # TODO(yiwen) validation
        # if epoch % valid_step == 0:
        #     valid(maskedDiff, val_data_loader)

    else:
        ### test dataloader (need flatten code, need corresponding pitch)
        test_dataset = TestDataset(scp_root="/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/exp/speechlm_opuslm_v1_1.7B_anneal_ext_phone_finetune_svs/decode_tts_espnet_sampling_temperature0.8_finetune_20epoch/svs_test/log",
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
        inference(maskedDiff, test_dataloader, codec_model, device)

