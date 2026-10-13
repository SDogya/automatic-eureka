"""CPU parsing GSM8K rewards и EOS trimming; binary reward rule и OWN G15/G12."""

import re
from fractions import Fraction


def numeric_answer(text):
    r"""Преобразовать поддерживаемую числовую запись в точную рациональную дробь.

    Args:
        text: Строка с целым, decimal/scientific числом или простой LaTeX-дробью.
            Удаляются пробелы, разделители тысяч, долларовые delimiters и left/right;
            dfrac/tfrac приводятся к frac. Произвольные математические выражения
            и вложенные дроби не вычисляются.

    Returns:
        fractions.Fraction для поддерживаемой записи; None для неподдерживаемой
        записи или нулевого знаменателя. Сравнение не использует float tolerance.

    Это OWN G15: точные правила извлечения и нормализации не опубликованы авторами."""
    text = re.sub(r"\\(?:dfrac|tfrac)", r"\\frac", text.strip())
    text = text.replace(",", "").replace("$", "").replace(" ", "")
    text = re.sub(r"\\(?:left|right)", "", text)
    fraction = re.fullmatch(r"(-?)\\frac\{([^{}]+)\}\{([^{}]+)\}", text)
    if fraction:
        sign, numerator, denominator = fraction.groups()
        if not (
            re.fullmatch(r"[+-]?\d+(?:\.\d+)?", numerator)
            and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", denominator)
        ):
            return None
        divisor = Fraction(denominator)
        return (-1 if sign else 1) * Fraction(numerator) / divisor if divisor else None
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
        return Fraction(text)
    return None


def boxed_reward(completion, answer, invalid_reward=0.0):
    r"""Вычислить бинарную GSM8K-награду по последней команде boxed.

    Args:
        completion: Декодированный ответ модели. Разбирается последнее вхождение
            boxed с балансировкой фигурных скобок, включая скобки простой frac.
        answer: Эталон GSM8K; числовая часть берётся после последнего ####.
        invalid_reward: Возвращаемое значение при отсутствии boxed или незакрытых
            скобках. Неподдерживаемая числовая запись получает 0.0.

    Returns:
        1.0 при точном совпадении двух разобранных Fraction, иначе 0.0,
        кроме отдельно заданных выше случаев invalid_reward.

    Binary exact-match соответствует статье; конкретный parser — OWN G15.
    Функция работает со строками на CPU и не участвует в autograd."""
    starts = list(re.finditer(r"\\boxed\s*\{", completion))
    if not starts:
        return invalid_reward
    start = starts[-1].end()
    depth, end = 1, start
    while end < len(completion) and depth:
        depth += (completion[end] == "{") - (completion[end] == "}")
        end += 1
    if depth:
        return invalid_reward
    prediction = numeric_answer(completion[start : end - 1])
    target = numeric_answer(answer.rsplit("####", 1)[-1])
    return float(prediction is not None and target is not None and prediction == target)


def trim_completion(ids, eos_id):
    r"""Обрезать token IDs после первого EOS, сохранив сам EOS.

    Args:
        ids: Python-последовательность ID токенов одной completion.
        eos_id: ID EOS конкретного tokenizer.

    Returns:
        Префикс до первого EOS включительно; исходная последовательность при
        отсутствии EOS. Последующие позиции больше не передаются в scoring.

    OWN G12: видимый EOS остаётся в контексте, но вызывающий код исключает его
    из eligible positions. Эта функция сама не создаёт scoring masks."""
    return ids[: ids.index(eos_id) + 1] if eos_id in ids else ids
