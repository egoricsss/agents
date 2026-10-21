"""Ошибки предметной области. Текст исключения можно показать пользователю как есть."""


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
