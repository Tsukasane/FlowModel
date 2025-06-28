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
import torch
import torch.nn.functional as F
from modules.CFM.matcha_flowmatching import BASECFM
import torch.nn as nn


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, estimator: torch.nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        # Just change the architecture of the estimator here
        self.estimator = estimator

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder NOTE(yiwen) first pass the original waveform to the encoder
                shape: (batch_size, n_feats, mel_timesteps) e.g. 2, 80, 50
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps) e.g. 2, 1, 50
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = self.forward_estimator(x, mask, mu, t, spks, cond)
            # Classifier-Free Guidance inference introduced in VoiceBox
            if self.inference_cfg_rate > 0:
                cfg_dphi_dt = self.forward_estimator( # NOTE(yiwen) 要在estimator（NN）的forward里对cond做处理
                    x, mask,
                    torch.zeros_like(mu), t,
                    torch.zeros_like(spks) if spks is not None else None,
                    torch.zeros_like(cond) if cond is not None else None
                )
                dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt -
                           self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1: # not the last step
                dt = t_span[step + 1] - t

        return sol[-1] # NOTE(yiwen) return 最后一个solution，是更新之后的数据点

    def forward_estimator(self, x, mask, mu, t, spks, cond):
        if isinstance(self.estimator, torch.nn.Module):
            return self.estimator.forward(x, mask, mu, t, spks, cond)
        else:
            self.estimator.set_input_shape('x', (2, 80, x.size(2)))
            self.estimator.set_input_shape('mask', (2, 1, x.size(2)))
            self.estimator.set_input_shape('mu', (2, 80, x.size(2)))
            self.estimator.set_input_shape('t', (2,))
            self.estimator.set_input_shape('spks', (2, 80))
            self.estimator.set_input_shape('cond', (2, 80, x.size(2)))
            # run trt engine
            self.estimator.execute_v2([x.contiguous().data_ptr(),
                                       mask.contiguous().data_ptr(),
                                       mu.contiguous().data_ptr(),
                                       t.contiguous().data_ptr(),
                                       spks.contiguous().data_ptr(),
                                       cond.contiguous().data_ptr(),
                                       x.data_ptr()])
            return x

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mo)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None. in TTS
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape # 32, 512, 360
        
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype) 
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi) # 32, 1, 1
        
        z = torch.randn_like(x1) # 32, 512, 360 TODO(yiwen) x1 gt 可以改成mel，从noise mel到good mel
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1 # position(interpolated distribution) along time
        u = x1 - (1 - self.sigma_min) * z # 32, 512, 360 velocity along time

        # u是学习的target，一个从random gaussian 到x1分布的 time dependent 速度场
        # during training, we randomly drop condition to trade off mode coverage and sample fidelity
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate # set the probability 32
            mu = mu * cfg_mask.view(-1, 1, 1)
            if cond is not None:
                cond = cond * cfg_mask.view(-1, 1, 1)

        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond) # NOTE(yiwen) should be the same shape as u

        ''' - u: B, 1, 80, T
            - pred: B, out_channel, T
            - mask: B, 1, T
        '''
        pred = pred.unsqueeze(1)
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        print(f'debug -- loss {loss}')
        return loss, y



class DummyEstimator(nn.Module):
    """ linear mapping estimator for testing ConditionalCFM """
    def forward(self, x, mask, mu, t, spks, cond):
        return x * 0.1  # 只是一个简单的线性缩放，真实情况应该是一个神经网络

def main():

    class DummyCFMParams:
        t_scheduler = 'cosine'
        training_cfg_rate = 0.1
        inference_cfg_rate = 0.1
        sigma_min = 0.01  # compute_loss 

    in_channels = 80  # mel dim=80
    cfm_params = DummyCFMParams()
    estimator = DummyEstimator()

    model = ConditionalCFM(in_channels, cfm_params, estimator)

    batch_size = 2
    mel_timesteps = 50
    spk_emb_dim = 80  # 假设 speaker embedding 也是 80 维

    mu = torch.randn(batch_size, in_channels, mel_timesteps)  # Encoder 输出
    mask = torch.ones(batch_size, 1, mel_timesteps)  # 全部有效
    x1 = torch.randn(batch_size, in_channels, mel_timesteps)  # 目标 mel 频谱
    spks = torch.randn(batch_size, spk_emb_dim) 
    cond = torch.randn(batch_size, in_channels, mel_timesteps)  # conditions should be LLM output tokens + pitch note gt

    # Tst Forward（gen mel-spectrogram）
    print("Testing forward...")
    with torch.no_grad():
        generated_mel = model(mu, mask, n_timesteps=10, spks=spks, cond=cond) # target distribution mel
        '''
        NOTE(yiwen) 
            - mu is from source distribution, mel
            - generated_mel is from target distribution, mel
        '''

    print("Generated mel shape:", generated_mel.shape)  # (batch_size, n_feats, mel_timesteps)

    # Compute Loss（训练损失）
    print("Testing compute_loss...")
    loss, y = model.compute_loss(x1, mask, mu, spks, cond)
    print("Loss:", loss.item())
    print("Flow shape:", y.shape)  # (batch_size, n_feats, mel_timesteps)

if __name__ == "__main__":
    main()
