#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import torch

from espnet2.speechlm.tokenizer.abs_tokenizer import AbsTokenizer
import soundfile as sf

class CodecTokenizer(AbsTokenizer):
    """Codec Tokenizer implementation

    Use cases:
        - use encode and decode for discrete (de)tokenization
        - use encode_continuous and decode_continuous for continuous
          (de)tokenization
        - use forward and detokenization for discrete (de)tokenization
          with flatten sequence style, which is more friendly for
          speechlm task
    """

    def __init__(
        self,
        codec_choice: str,
        codec_fs: int,
        device: str = "cpu",
        dump_audio: bool = False,
        checkpoint_path: str = None,
        config_path: str = None,
        hf_model_tag: str = None,
        max_token_per_frame: int = 8,
    ):
        """Codec Tokenizer initialization

        Each of the codec implementation should assign all following features:
            self.n_codebook (int): the number of codec codebooks.
            self.size_codebook (int): the dimension of codebooks.
            self.sample_rate (int): the sample rate the model trained on.
            self.subsample (int): the subsample rate, a.k.a., frame shift.
        """

        super(CodecTokenizer, self).__init__()
        self.codec_choice = codec_choice
        self.device = device
        self.dump_audio = dump_audio

        if self.codec_choice == "ESPnet":

            if hf_model_tag is not None:
                from espnet2.bin.gan_codec_inference import AudioCoding

                model = AudioCoding.from_pretrained(
                    hf_model_tag, device=str(device)
                ).model
            else:
                from espnet2.tasks.gan_codec import GANCodecTask

                model, _ = GANCodecTask.build_model_from_file(
                    config_path,
                    checkpoint_path,
                    device=str(device),
                )
            self.codec = model

            meta_info = self.codec.meta_info()
            self.n_codebook = min(meta_info["num_streams"], max_token_per_frame)
            self.size_codebook = meta_info["code_size_per_stream"][0]
            self.sample_rate = meta_info["fs"]
            self.subsample = meta_info["frame_shift"]

        elif self.codec_choice == "DAC":
            try:
                import dac
            except ImportError:
                raise ImportError("Install DAC with: pip install descript-audio-codec")

            model_path = dac.utils.download(
                model_type=str(codec_fs).replace("000", "khz")
            )
            self.codec = dac.DAC.load(model_path).to(device)
            self.n_codebook = self.codec.n_codebooks
            self.size_codebook = self.codec.codebook_size
            self.sample_rate = self.codec.sample_rate
            self.subsample = np.prod(self.codec.encoder_rates)

        elif self.codec_choice == "EnCodec":
            try:
                from encodec import EncodecModel
            except ImportError:
                raise ImportError("Please install Encodec with: pip install -U encodec")

            model_name = "encodec_model_" + str(codec_fs).replace("000", "khz")
            self.codec = getattr(EncodecModel, model_name)().to(device)
            # NOTE (Jinchuan): This Encodec model has 32 codebooks,
            # which is not necessary in usual cases.
            # We only adopt 8 first codebooks, a.k.a., 6kbps.
            bandwidth = 6.0
            self.codec.set_target_bandwidth(bandwidth)
            self.n_codebook = self.codec.quantizer.get_num_quantizers_for_bandwidth(
                self.codec.frame_rate, bandwidth
            )
            self.size_codebook = self.codec.quantizer.bins
            self.sample_rate = self.codec.sample_rate
            self.subsample = np.prod(self.codec.encoder.ratios)

        elif self.codec_choice == "inhouse":
            try:
                from models.soundstream import SoundStream
                from omegaconf import OmegaConf
            except ImportError:
                raise ImportError("fail to use inhouse codec")

            model_path = "encodec_16k_6kbps_multiDisc/ckpt_01135000.pth"
            model_config = "encodec_16k_6kbps_multiDisc/config.yaml"
            config = OmegaConf.load(model_config)
            model = SoundStream(**config.generator.config)

            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict["codec_model"])
            model = model.to(device)
            self.codec = model

            self.n_codebook = 8
            self.sample_rate = 16000
            self.size_codebook = 1024
            self.subsample = 320

        else:
            raise ValueError(f"Codec {codec_choice} is not supported")

    def encode(self, wavs):
        """
        Convert audio waveforms into codec codes
        Input:
            wavs (torch.Tensor): float tensor in shape [B, 1, n_sample],
        Output:
            codes (torch.Tensor): Int tensor in shape [B, T, n_codebook]
        """
        assert wavs.dim() == 3 and wavs.size(1) == 1

        if self.codec_choice == "ESPnet":
            codes = self.codec.encode(wavs)
            codes = codes.permute(1, 2, 0)[:, :, : self.n_codebook]

        elif self.codec_choice == "DAC":
            codes = self.codec.encode(wavs)[1]
            codes = codes.transpose(1, 2)

        elif self.codec_choice == "EnCodec":
            encoded_frames = self.codec.encode(wavs)
            codes = encoded_frames[0][0].transpose(1, 2)

        elif self.codec_choice == "inhouse":
            codes = self.codec.encode(wavs).permute(1, 2, 0)

        else:
            raise NotImplementedError

        return codes

    def encode_continuous(self, wavs):
        """
        Convert audio waveforms into continuous codec encoding results
        Input:
            wavs (torch.Tensor): float tensor in shape [B, 1, n_sample],
        Output:
            z (torch.Tensor): float tensor in shape [B, T, D]
        """

        if self.codec_choice == "ESPnet":
            z = self.codec.encode_continuous(wavs)
            z = z.transpose(1, 2)

        elif self.codec_choice == "DAC":
            z = self.codec.encode(wavs)[0]
            z = z.transpose(1, 2)

        else:
            raise NotImplementedError

        return z

    def decode(self, codes, return_intermediate=True):
        """
        Recover the waveform from the codes.
        Input:
            codes (torch.Tensor): Int tensor in shape [B, T, n_codebook]
        Output:
            waveform (torch.Tensor): float tensor in shape [B, n_sample]
        """

        # NOTE(Jinchuan) The very short input may raise errors, so simply
        # make the output as 0.0
        if codes.size(1) <= 10:
            B, T, _ = codes.size()
            return torch.zeros(
                (B, self.subsample * T),
                dtype=torch.float32,
                device=codes.device,
            )

        if self.codec_choice == "ESPnet":
            codes = codes.permute(2, 0, 1)
            waveform, quantized = self.codec.decode(codes)
            # quantized = quantized[0] # the two copy are the same
            waveform = waveform.squeeze(1)
            if return_intermediate:
                return quantized

        elif self.codec_choice == "DAC":
            z = self.codec.quantizer.from_codes(codes.transpose(1, 2))[0]
            waveform = self.codec.decode(z).squeeze(1)

        elif self.codec_choice == "EnCodec":
            encoded_frames = [(codes.transpose(1, 2), None)]
            waveform = self.codec.decode(encoded_frames).squeeze(1)

        elif self.codec_choice == "inhouse":
            codes = codes.permute(2, 0, 1)
            waveform = self.codec.decode(codes).squeeze(1)

        else:
            raise NotImplementedError

        return waveform

    def decode_continuous(self, z):
        """
        Recover the waveform from the continuous representations of codec
        Input:
            z (torch.Tensor): Float tensor in shape [B, T, D], codec
              continuous representations
        Output:
            waveform (torch.Tensor): float tensor in shape [B, n_sample]
        """
        if self.codec_choice == "ESPnet":
            z = z.transpose(1, 2)
            waveform = self.codec.decode_continuous(z).squeeze(1)

        elif self.codec_choice == "DAC":
            z = z.transpose(1, 2)
            waveform = self.codec.decode(z).squeeze(1)

        else:
            raise NotImplementedError

        return waveform

    def forward(self, wavs):
        """
        Convert audio waveforms into flatten codec codes and resynthesis the audio
        Input:
            wavs (torch.Tensor): float tensor in shape [B, 1, n_sample],
        Output:
            codes (torch.Tensor): Int tensor in shape [B, T * n_codebook],
            resyn_audio (torch.Tensor): float tensor in shape [B, n_samples]
        """
        codes = self.encode(wavs)
        if self.dump_audio:
            resyn_audio = self.decode(codes)
        else:
            resyn_audio = None

        shift = torch.arange(self.n_codebook).to(self.device)
        codes += shift.view(1, 1, -1) * self.size_codebook
        codes = codes.int().flatten(start_dim=1)

        return codes, resyn_audio

    def detokenize(self, codes, n_codebook=None):
        """
        Convert flatten codec codes into resynthesis the audio
        Input:
            codes (torch.Tensor): int tensor in shape [B, T * n_codebook],
                or [T * n_codebook]
        Output:
            waveform (torch.Tensor): float tensor in shape [B, n_sample],
                or [n_sample]
        """

        has_batch = codes.dim() == 2
        if not has_batch:
            codes = codes.unsqueeze(0)

        B, Tnq = codes.size()
        n_codebook = self.n_codebook if n_codebook is None else n_codebook
        assert Tnq % n_codebook == 0, (n_codebook, codes.size())
        codes = codes.view(B, Tnq // n_codebook, n_codebook)

        for l_idx in range(n_codebook):
            codes[:, :, l_idx] -= l_idx * self.size_codebook

        intermedia_feats = self.decode(codes, return_intermediate=True)

        if not has_batch:
            intermedia_feats = intermedia_feats.squeeze(0)

        return intermedia_feats


def get_codec_tokenizer(device):
    
    codec = CodecTokenizer(
        codec_choice="ESPnet",
        codec_fs=16000,
        device=device,
        dump_audio=True,
        hf_model_tag="ftshijt/espnet_codec_dac_large_v1.4_360epoch",
        checkpoint_path="/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/codec_test/360epoch.pth",
        config_path="/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/codec_test/config.yaml",
    )

    return codec

if __name__ == "__main__":
    # a simple use case for batch processing
    filename="/ocean/projects/cis210027p/yzhao16/speechlm2/espnet/egs2/acesinger/speechlm1/wav_dump/acesinger_9#2097003617.wav"
    waveform, sr = sf.read(filename)
    device = "cuda:0"

    waveform = (
        torch.from_numpy(waveform).view(1, 1, -1).to(device).float()
    )  # [B, C, n_sample]
    # waveform = waveform.repeat(2, 1, 1) # suppose it has a batch

    codec = get_codec_tokenizer(device)

    with torch.no_grad():

        flatten_codes, _ = codec(waveform)
        print(f"flatten_codes", flatten_codes.size())

        # B, D, T  1, 512, 192
        codec_continuous_feats = codec.detokenize(flatten_codes) # NOTE(yiwen) modified here to obtain continuous features
        

        import pdb
        pdb.set_trace()

        # # continuous
        # codes = codec.encode(waveform)
        # print(f"cdoes: ", codes.size())
        # continuous_feats = codec.decode(codes, return_intermediate=True)
