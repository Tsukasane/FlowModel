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
from torch.utils.data import DataLoader

from modules.codec_ssl_tokenizer.codec_tokenizer import get_codec_tokenizer
import soundfile as sf
import kaldiio

import logging
logging.basicConfig(level=logging.INFO)

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
                generator_model_dir: str = "flow_model/pretrained_quantizer/music_tokenizer",
                num_codebooks: int = 4
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
                                  
    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:

        codec_continuous_feats = batch['codec_continuous_feats']  # 32, 349, 512
        codec_token_len = batch['codec_feats_len']
        pitch = batch['pitch']
        # log_mel = batch['log_mel'] # 32, 349, 80
        # h1, h1_lengths = self.encoder(log_mel, codec_token_len) # if using mel as x
        # h1, h1_lengths = self.length_regulator(h1, codec_token_len) 

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
        # conds = pitch # TODO(yiwen) gt pitch 
        conds = None
        mask = (~make_pad_mask(codec_token_len)).to(h) # TODO(yiwen) check the meaning of .to(h1)

        loss, _ = self.decoder.compute_loss( 
                codec_continuous_feats.transpose(1,2), # the target for flow
                mask.unsqueeze(1),
                h.transpose(1, 2).contiguous(), # the encoder output (latent)
                None,
                cond=conds
        )

        # loss, _ = self.decoder.compute_loss( 
        #         log_mel.transpose(1,2), # the target for flow
        #         mask.unsqueeze(1),
        #         h1.transpose(1, 2).contiguous(), # the encoder output (latent)
        #         None, # spk
        #         cond=conds # TODO(yiwen) check whether need to transpose
        # )

        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  sample_rate):
        assert token.shape[0] == 1

        token = self.input_embedding(torch.clamp(token, min=0)) 
        h, h_lengths = self.encoder(token, token_len)

        if sample_rate == 48000:
            token_len = 2 * token_len

        h, h_lengths = self.length_regulator(h, token_len)  

        # get conditions
        conds = None

        mask = (~make_pad_mask(token_len)).to(h)
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=None,
            cond=conds,
            n_timesteps=10
        ) # get the feature, then pass to the vocoder

        print(f'debug -- feat.shape {feat.shape}')
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
        in_channels=1024,
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



def collate_fn(batch):
    # padding to max length in a batch
    waveforms = [item['waveform'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch])
    filenames = [item['filename'] for item in batch]
    pitchs = [item['pitch'] for item in batch]

    max_len = max(lengths)

    longest_pitch = len(max(pitchs, key=len))
    padded_pitch = [item['pitch'] + [0] * (longest_pitch - len(item['pitch'])) for item in batch]

    pitch_tensors = torch.tensor(padded_pitch)

    padded_waveforms = torch.zeros(len(waveforms), 1, max_len)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, :, :w.shape[1]] = w

    return padded_waveforms, lengths, filenames, pitch_tensors



def get_codec(codec_model, waveform, length, down_sample_rate=40, codec_layer_num=8):
    batch = {}

    # NOTE(yiwen) this is from discrete ids (/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/espnet2/gan_codec/dac/dac.py)
    with torch.no_grad():
        flatten_codes, _ = codec_model(waveform)
        # print(f"flatten_codes", flatten_codes) #torch.Size([1, 1536])
        codec_continuous_feats = codec_model.detokenize(flatten_codes).transpose(1, 2)
        # codec_continuous_feats = codec_model.detokenize(codec_token_flatten).transpose(1, 2) # torch.Size([1, 192, 512])

    # # NOTE(yiwen) this is from the waveform (output of encoder)
    # with torch.no_grad():
    #     codec_coutinuous_feats = codec_model(waveform)

    
    codec_token_len = (length+down_sample_rate*codec_layer_num-1) / (down_sample_rate*codec_layer_num) 
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
                    save_path='./latest_model.pth'):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} '.format(epoch_id+1, lr))
    
        model.train()
        total_loss = 0
        num_batch = 0

        for batch_waveforms, lengths, filenames, pitch in train_data_loader:
            num_batch += 1
            # batch_waveforms, lengths, filenames = batch
            batch_waveforms = batch_waveforms.to(device) # 32, 1, 116718
            # log_mel = waveform_to_mel(batch_waveforms, device=device) # 32, 1, 80, 365
            # log_mel = log_mel.reshape(-1, *log_mel.shape[2:]).transpose(1, 2) # --> 32, 365, 80
        
            input_batch = get_codec(codec_model, batch_waveforms, lengths) # codec embedding
            # input_batch['codec_continuous_feats'].shape 32, 365, 512
            # input_batch['log_mel'] = log_mel

            input_batch['pitch'] = pitch

            optimizer.zero_grad()
            output_loss_dic = model(input_batch, device)
            
            batch_loss = output_loss_dic['loss']
            if num_batch%10==0:
                print(f'debug -- batch loss {batch_loss.item()}')

            # if num_batch==100: # for debug
            #     break
            total_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()

        total_loss /= num_batch
        logging.info(f'Training Loss for Epoch {epoch_id}: {total_loss}')

        # TODO(yiwen) save model (is able to resume on)
        if epoch_id % 2 ==0: # debug
            if os.path.exists(save_path):
                os.rename(save_path, './latest_bu.pth')
            torch.save({
                'epoch': epoch_id,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss, # only the number with no gradient
                }, save_path)



def valid(valid_data_loader):
    pass


def inference(test_data):
    pass
    '''
    Inputs:
        - pitch(list): as condition
        - noise z / noisy codec embedding
    Outputs:
        - clean codec embedding
        then waveform = self.decoder(clean_codec_embedding)
    '''

if __name__=='__main__':
    
    device = "cuda:0"
    valid_step = 10
    total_epochs = 20

    batch_size = 4
    train = True

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
    )

    # NOTE(yiwen) temp
    optimizer = torch.optim.Adam(maskedDiff.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # TODO(yiwen) add train data loader
    dataset = AudioDataset("./datasets/wav_dump/", transform=None, sample_rate=16000)
    train_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    # val_data_loader = None

    if train:
        for epoch in range(total_epochs):
            maskedDiff.train()
            train_one_epoch(maskedDiff, optimizer, scheduler, train_data_loader, codec_model, device, epoch)
        
        # TODO(yiwen) validation
        # if epoch % valid_step == 0:
        #     valid(maskedDiff, val_data_loader)

    else:
        checkpoint = torch.load("/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/flow_model/latest_model.pth", weights_only=True)
        maskedDiff.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        maskedDiff.eval()
