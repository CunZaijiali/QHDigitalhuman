"""Wav2Lip 生成器网络（单文件自包含移植版）。

Source / 来源:
    D:\\XiangMu\\ScenicAgent\\ScenicAgentPy\\src\\domain\\lip_real\\models\\wav2lip_v2.py
    D:\\XiangMu\\ScenicAgent\\ScenicAgentPy\\src\\domain\\lip_real\\models\\conv.py

Purpose / 用途:
    Wav2Lip 唇形同步的生成器（generator）网络定义，用于推理。为保持与现成权重
    ``models/wav2lip/wav2lip.pth`` 的兼容性，类名与 ``state_dict`` 键名与源项目
    逐字一致；被用到的卷积层（``Conv2d`` / ``Conv2dTranspose``）已从源 ``conv.py``
    内联到本文件，因此本模块不依赖参考项目的任何模块。

Usage / 用法:
    from src.domain.models.wav2lip import build_wav2lip_model

    model = build_wav2lip_model("models/wav2lip/wav2lip.pth", device="cpu")
    out = model(mel, face)  # mel: (B, 1, 80, 16) / face: (B, 6, 256, 256)

Input contract / 输入约定:
    audio_sequences: (B, 1, 80, 16) float32 mel 频谱块；
    face_sequences : (B, 6, 256, 256) float32，6 通道 = 遮挡下 3 通道 + 参考上 3 通道；
    也可传入带时间维的 5 维输入 (B, 1, T, 80, 16) / (B, 6, T, 256, 256)，
    此时输出为 (B, 3, T, 256, 256)。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class Conv2d(nn.Module):
    """Conv2d + BatchNorm2d + ReLU，可选 residual 连接（内联自源 conv.py）。"""

    def __init__(
        self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding), nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class Conv2dTranspose(nn.Module):
    """ConvTranspose2d + BatchNorm2d + ReLU（内联自源 conv.py）。"""

    def __init__(
        self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


class Wav2Lip(nn.Module):
    """Wav2Lip 生成器（与源 ``wav2lip_v2.Wav2Lip`` 结构一致）。"""

    def __init__(self):
        super(Wav2Lip, self).__init__()

        self.face_encoder_blocks = nn.ModuleList(
            [
                nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),
                nn.Sequential(
                    Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2d(512, 512, kernel_size=4, stride=1, padding=0),
                    Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                ),
            ]
        )

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

        self.face_decoder_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                ),
                nn.Sequential(
                    Conv2dTranspose(1024, 512, kernel_size=4, stride=1, padding=0),
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2dTranspose(
                        1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2dTranspose(
                        1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2dTranspose(
                        768, 384, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2dTranspose(
                        512, 256, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2dTranspose(
                        320, 128, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2dTranspose(
                        160, 64, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                ),
            ]
        )

        self.output_block = nn.Sequential(
            Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def audio_forward(self, audio_sequences, a_alpha=1.0):
        """只跑音频编码器，返回音频嵌入 (B, 512, 1, 1)。"""
        audio_embedding = self.audio_encoder(audio_sequences)
        if a_alpha != 1.0:
            audio_embedding *= a_alpha
        return audio_embedding

    def inference(self, audio_embedding, face_sequences):
        """用预计算好的音频嵌入跑人脸编码 + 解码，返回 (B, 3, H, W)。"""
        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Wav2Lip skip-connection shape mismatch: "
                    f"decoder={tuple(x.shape)}, encoder={tuple(feats[-1].shape)}"
                ) from exc
            feats.pop()

        return self.output_block(x)

    def forward(self, audio_sequences, face_sequences, a_alpha=1.0):
        # audio_sequences = (B, T, 1, 80, 16) 或 (B, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat(
                [audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0
            )  # [bz, 5, 1, 80, 16]->[bz*5, 1, 80, 16]
            face_sequences = torch.cat(
                [face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0
            )  # [bz, 6, 5, 256, 256]->[bz*5, 6, 256, 256]

        audio_embedding = self.audio_encoder(
            audio_sequences
        )  # [bz*5, 1, 80, 16]->[bz*5, 512, 1, 1]
        if a_alpha != 1.0:
            audio_embedding *= a_alpha

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Wav2Lip skip-connection shape mismatch: "
                    f"decoder={tuple(x.shape)}, encoder={tuple(feats[-1].shape)}"
                ) from exc
            feats.pop()

        x = self.output_block(x)  # [bz*5, 80, 256, 256]->[bz*5, 3, 256, 256]

        if input_dim_size > 4:  # [bz*5, 3, 256, 256]->[B, 3, 5, 256, 256]
            x = torch.split(x, B, dim=0)
            outputs = torch.stack(x, dim=2)
        else:
            outputs = x

        return outputs


def _load_checkpoint(path: Path, device: torch.device):
    """加载 .pth；兼容新旧 torch 的 weights_only 默认值差异。"""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def build_wav2lip_model(
    checkpoint_path: str | Path | None = None, device: str | torch.device = "cpu"
) -> nn.Module:
    """构造 Wav2Lip 生成器，可选加载权重。

    Args:
        checkpoint_path: 权重文件路径。为 ``None`` 时只建网络、随机初始化。
            传入的权重可以是裸 ``state_dict``，也可以是含 ``state_dict`` 键的
            checkpoint；键名中的 ``module.`` 前缀会自动剥除。
        device: 目标设备，默认 ``"cpu"``。

    Returns:
        ``Wav2Lip``（``nn.Module``），处于 ``eval()`` 模式、位于 ``device`` 上。

    Raises:
        FileNotFoundError: 权重路径不存在。
        RuntimeError: ``strict=True`` 加载失败，异常信息含 missing / unexpected 键。
    """
    target = torch.device(device)
    model = Wav2Lip()

    if checkpoint_path is None:
        return model.to(target).eval()

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Wav2Lip checkpoint not found: {path.resolve()}")

    checkpoint = _load_checkpoint(path, torch.device("cpu"))
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError(
            f"Unsupported Wav2Lip checkpoint format at {path}: "
            f"expected a state_dict or a dict with 'state_dict', got {type(checkpoint).__name__}"
        )

    state_dict = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        raise RuntimeError(
            f"Wav2Lip checkpoint at {path} does not match the model "
            f"(strict load failed): missing={len(missing)} {missing[:5]}, "
            f"unexpected={len(unexpected)} {unexpected[:5]}; original error: {exc}"
        ) from exc

    return model.to(target).eval()


__all__ = ["Conv2d", "Conv2dTranspose", "Wav2Lip", "build_wav2lip_model"]
