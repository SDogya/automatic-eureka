"""Prompt-dependent log Z head и detached clean-prompt backbone features."""

import torch
from torch import nn

from .attention import staircase_attention


class LogZ(nn.Module):
    r"""Двухслойная prompt-dependent MLP для scalar log Z в TraFL.
    
    Head получает detached mean-pooled prompt features и обучается совместно с LoRA.
    Статья задаёт два слоя, pooling и detachment; hidden width, activation, biases
    и zero-output initialization — явно выбранные OWN G08/G09 настройки.
    Имена layers и формат state_dict сохранены для совместимости checkpoints."""
    def __init__(self, dimension, hidden, activation, bias, zero_output):
        r"""Создать dimension -> hidden -> 1 head с выбранной активацией.
        
        Args:
            dimension: Размерность backbone prompt features (1024 у выбранной Qwen).
            hidden: Ширина скрытого слоя.
            activation: Имя класса torch.nn, например SiLU.
            bias: Включить biases у обоих Linear.
            zero_output: Обнулить weight и, если есть, bias последнего Linear;
                начальный scalar log Z тогда равен нулю."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dimension, hidden, bias=bias),
            getattr(nn, activation)(),
            nn.Linear(hidden, 1, bias=bias),
        )
        if zero_output:
            nn.init.zeros_(self.layers[-1].weight)
            if bias:
                nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features):
        r"""Вернуть scalar log Z для каждого prompt в batch.
        
        Args:
            features: Tensor [B, H]; исходный граф backbone здесь явно detach.
        
        Returns:
            Tensor [B]. Features приводятся к FP32; gradients проходят через head
            parameters, но не через features к backbone."""
        return self.layers(features.detach().float()).squeeze(-1)



@torch.no_grad()
def prompt_features(policy, prompt, dtype, block_size):
    r"""Извлечь detached mean-pooled final states чистого prompt.
    
    Args:
        policy: Current PEFT policy; get_base_model().model возвращает backbone.
        prompt: Unpadded long tensor [P] на GPU без completion suffix.
        dtype: BF16/другая поддержанная autocast dtype для backbone forward.
        block_size: BD3LM block size, такой же как в текущем train/eval protocol.
    
    Returns:
        FP32 detached tensor [1, H], mean по всем P nonpadding prompt positions.
    
    Функция целиком no_grad; cache выключен. Выбор final states текущей policy
    на отдельном clean-prompt forward — OWN G09. Режим eval/train задаётся caller.
    Prompt forward использует тот же staircase attention, что policy scores."""
    attention_mask, position_ids = staircase_attention(prompt.numel(), block_size, prompt.device)
    with torch.autocast(device_type=prompt.device.type, dtype=dtype):
        hidden = (
            policy.get_base_model()
            .model(prompt[None], attention_mask=attention_mask,
                   position_ids=position_ids, use_cache=False)
            .last_hidden_state
        )
    return hidden.float().mean(dim=1).detach()

