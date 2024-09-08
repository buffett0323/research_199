import math
import librosa
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import init
from torch.nn.parameter import Parameter


from models.e2e.bandit.bandsplit import BandSplitModule
from models.e2e.bandit.utils import MusicalBandsplitSpecification
from models.e2e.bandit.maskestim import OverlappingMaskEstimationModule
from models.e2e.bandit.tfmodel import SeqBandModellingModule
from models.e2e.querier.passt import Passt, PasstWrapper
from models.e2e.conditioners.film import FiLM
from models.types import InputType, OperationMode, SimpleishNamespace
from beats.BEATs import BEATs, BEATsConfig

from unet import UnetIquery


if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)

class BaseEndToEndModule(pl.LightningModule):

    def __init__(
        self,
    ) -> None:
        super().__init__()
        
        
class MyBandSplit(BaseEndToEndModule):
    def __init__(
        self,
        in_channel: int,
        stems: List[str],
        band_type: str = "musical",
        n_bands: int = 64,
        additive_film: bool = True,
        multiplicative_film: bool = True,
        film_depth: int = 2,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
        n_sqm_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
        mlp_dim: int = 512,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Dict | None = None,
        complex_mask: bool = True,
        use_freq_weights: bool = True,
        n_fft: int = 2048,
        win_length: int | None = 2048,
        hop_length: int = 512,
        window_fn: str = "hann_window",
        wkwargs: Dict | None = None,
        power: int | None = None,
        center: bool = True,
        normalized: bool = True,
        pad_mode: str = "constant",
        onesided: bool = True,
        use_beats: bool = True, # False
        fs: int = 44100,
    ):
        super().__init__()
        
        
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad_mode="constant",
            pad=0,
            window_fn=torch.__dict__["hann_window"],
            wkwargs=None,
            power=None,
            normalized=True,
            center=True,
            onesided=True,
        )
        
        
        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad_mode="constant",
            pad=0,
            window_fn=torch.__dict__["hann_window"],
            wkwargs=None,
            normalized=True,
            center=True,
            onesided=True,
        )
        
        
        self.instantiate_bandsplit(
            in_channel=in_channel,
            band_type=band_type,
            n_bands=n_bands,
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
            emb_dim=emb_dim,
            n_fft=n_fft,
            fs=fs,
        )
        
        
        self.instantiate_tf_modelling(
            n_sqm_modules=n_sqm_modules,
            emb_dim=emb_dim,
            rnn_dim=rnn_dim,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
        )
        

        self.instantiate_mask_estim(
            in_channel=in_channel,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            n_freq=n_fft // 2 + 1,
            use_freq_weights=use_freq_weights,
        )
        

        self.query_encoder = Passt(
            original_fs=fs,
            passt_fs=32000,
        )
        

        self.film = FiLM(
            self.query_encoder.PASST_EMB_DIM,
            emb_dim,
            additive=additive_film,
            multiplicative=multiplicative_film,
            depth=film_depth,
        )
        
        self.use_beats = use_beats
        if self.use_beats:
            self.instantiate_beats(beats_check_point_pth='beats/pt_dict/BEATs_iter3_plus_AS2M.pt')

    def instantiate_beats(
        self,
        beats_check_point_pth='beats/pt_dict/BEATs_iter3_plus_AS2M.pt',
    ):
        checkpoint = torch.load(beats_check_point_pth)

        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model'])
        self.beats = BEATs_model.eval().cuda()


    def instantiate_bandsplit(
        self,
        in_channel: int,
        band_type: str = "musical",
        n_bands: int = 64,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
        emb_dim: int = 128,
        n_fft: int = 2048,
        fs: int = 44100,
    ):

        assert band_type == "musical"

        self.band_specs = MusicalBandsplitSpecification(
            nfft=n_fft, fs=fs, n_bands=n_bands
        )

        self.band_split = BandSplitModule(
            in_channel=in_channel,
            band_specs=self.band_specs.get_band_specs(),
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
            emb_dim=emb_dim,
        )
        
        
        
    def beats_query(self, wav):
        wav = wav.mean(dim=1, keepdim=False)
        padding_mask = torch.zeros(1, wav.shape[1]).bool().cuda()  # Move padding mask to GPU
        embed = []

        for i in range(wav.shape[0]):
            embed.append(self.beats.extract_features(wav[i].unsqueeze(0), padding_mask=padding_mask)[0].mean(dim=1, keepdim=False))

        embed = torch.cat(embed, dim=0)
        return embed
    
    
    
    def instantiate_tf_modelling(
        self,
        n_sqm_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
    ):
        self.tf_model = SeqBandModellingModule(
            n_modules=n_sqm_modules,
            emb_dim=emb_dim,
            rnn_dim=rnn_dim,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
        )
        
        
    def instantiate_mask_estim(
        self,
        in_channel: int,
        emb_dim: int,
        mlp_dim: int,
        hidden_activation: str,
        hidden_activation_kwargs: Optional[Dict] = None,
        complex_mask: bool = True,
        n_freq: Optional[int] = None,
        use_freq_weights: bool = True,
    ):
        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}

        assert n_freq is not None

        self.mask_estim = OverlappingMaskEstimationModule(
            band_specs=self.band_specs.get_band_specs(),
            freq_weights=self.band_specs.get_freq_weights(),
            n_freq=n_freq,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            in_channel=in_channel,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            use_freq_weights=use_freq_weights,
        )
        
        
    def mask(self, x, m):
        return x * m
    
    
    def forward(self, batch: InputType):
        with torch.no_grad():
            x = self.stft(batch.mixture.audio)
            batch.mixture.spectrogram = x

            if "sources" in batch.keys():
                for stem in batch.sources.keys():
                    s = batch.sources[stem].audio
                    s = self.stft(s)
                    batch.sources[stem].spectrogram = s
                    
        batch = self.separate(batch)

        return batch
    
    
    def separate(self, batch):

        x, q, length = self.encode(batch)

        q = self.adapt_query(q, batch)

        m = self.mask_estim(q)
        s = self.mask(x, m)
        s = torch.reshape(s, x.shape)
        batch.estimates["target"] = SimpleishNamespace(
            audio=self.istft(s, length), spectrogram=s
        )

        return batch
    
    
    def encode(self, batch):
        x = batch.mixture.spectrogram
        length = batch.mixture.audio.shape[-1]
        z = self.band_split(x)  # (batch, emb_dim, n_band, n_time)
        q = self.tf_model(z)  # (batch, emb_dim, n_band, n_time)

        return x, q, length
    
    
    def adapt_query(self, q, batch):
        if self.use_beats:
            w = self.beats_query(batch.query.audio)
        else:
            w = self.query_encoder(batch.query.audio)
            
        # BF: torch.Size([1, 2, 441000]) 
        # AF: torch.Size([1, 768])
        q = torch.permute(q, (0, 3, 1, 2)) # (batch, n_band, n_time, emb_dim) -> (batch, emb_dim, n_band, n_time)
        q = self.film(q, w)
        q = torch.permute(q, (0, 2, 3, 1)) # -> (batch, n_band, n_time, emb_dim)
        
        return q
    




