"""BD3LM staircase attention для unpadded prompt/completion sequences."""

import torch


def staircase_attention(length, block_size, device):
    """Создать block-causal attention mask и абсолютные position IDs.

    Args:
        length: Длина всей последовательности prompt + completion, либо prompt.
        block_size: BD3LM block size, отсчитываемый от начала prompt; default
            модели в model card равен 32, runtime значение передаётся из YAML.
        device: Device input tokens и mask.

    Returns:
        Boolean attention_mask [1, 1, T, T] и long position_ids [1, T].
        True разрешает attention: key_block <= query_block. Внутри блока
        attention двунаправленный, будущие блоки полностью скрыты. Batch
        dimension broadcast при генерации нескольких completions.

    Loader/collate не создают padding; EOS — видимый обычный context token.
    Поэтому position IDs непрерывны, а padding mask не нужен. Это форма
    staircase mask из BD3LM model card для fully valid input sequences.
    Geometry не зависит от того, какие completion tokens сейчас corrupted.
    """
    positions = torch.arange(length, device=device)
    blocks = positions // block_size
    allowed = blocks[None, :] <= blocks[:, None]
    return allowed[None, None], positions[None]
