"""CPU antithetic completion masks для Eq. (4), включая OWN G11 endpoint."""

import torch


def antithetic_masks(eligible, replicates, generator, endpoint="repeat_full"):
    r"""Построить antithetic пары boolean masks для одной completion.
    
    Args:
        eligible: Одномерный CPU bool tensor [L]; True только у позиций, которые
            можно маскировать. Prompt в этот tensor не входит, EOS уже исключён.
        replicates: Число masks K; текущий protocol использует чётное K=4.
        generator: Выделенный CPU torch.Generator для mask sampling.
        endpoint: При repeat_full пустой complement заменяется полным eligible
            mask; другое значение оставляет пустой complement.
    
    Returns:
        CPU bool tensor [K, L]. При отсутствии eligible positions все masks пусты.
        В каждой паре level равномерно выбран из 1..N, subset — без replacement,
        второй mask — complement первого, кроме оговорённого full endpoint.
    
    K=4 и antithetic sampling заданы в статье; endpoint convention — OWN G11.
    Новые случайные draws изменяют только состояние переданного generator."""
    positions = eligible.nonzero().flatten()
    count = positions.numel()
    masks = torch.zeros(replicates, eligible.numel(), dtype=torch.bool)
    if count == 0:
        return masks
    for pair in range(replicates // 2):
        level = int(torch.randint(1, count + 1, (), generator=generator))
        selected = positions[torch.randperm(count, generator=generator)[:level]]
        masks[2 * pair, selected] = True
        complement = eligible & ~masks[2 * pair]
        masks[2 * pair + 1] = (
            complement
            if complement.any()
            else (eligible if endpoint == "repeat_full" else torch.zeros_like(eligible))
        )
    return masks

