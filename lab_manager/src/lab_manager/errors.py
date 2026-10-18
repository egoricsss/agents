"""Исключения предметной области."""


class LabManagerError(Exception):
    """Базовая ошибка приложения."""


class NotFoundError(LabManagerError):
    """Запрошенная или связанная сущность не существует."""


class AlreadyExistsError(LabManagerError):
    """Нарушена уникальность (например, такой файл или номер лабы уже есть)."""
