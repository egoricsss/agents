"""Ошибки предметной области. Текст исключения можно показать пользователю как есть."""

import math
from datetime import timedelta


class TelepathyError(Exception):
    pass


class NotPairedError(TelepathyError):
    def __init__(self) -> None:
        super().__init__("Сначала создайте пару: /pair — получить код, /join КОД — вступить.")


class AlreadyPairedError(TelepathyError):
    def __init__(self) -> None:
        super().__init__("Вы уже в паре. В паре строго два человека.")


class OwnInviteError(TelepathyError):
    def __init__(self) -> None:
        super().__init__("Это ваш собственный код — отправьте его партнёру.")


class InviteNotFoundError(TelepathyError):
    def __init__(self) -> None:
        super().__init__("Код не найден. Проверьте его или попросите партнёра сделать /pair.")


class PairFullError(TelepathyError):
    def __init__(self) -> None:
        super().__init__("В этой паре уже двое — третий лишний.")


class ThoughtLimitError(TelepathyError):
    def __init__(self, limit: int, retry_after: timedelta) -> None:
        self.limit = limit
        self.retry_after = retry_after
        minutes = max(1, math.ceil(retry_after.total_seconds() / 60))
        super().__init__(
            f"⏳ Не больше {limit} мыслей в час. Следующую можно отправить через {minutes} мин."
        )
