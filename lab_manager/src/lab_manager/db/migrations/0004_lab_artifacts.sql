-- Ожидаемые артефакты сдачи (программа, отчёт, схема…), извлечённые из задания. JSON-массив строк.
ALTER TABLE labs ADD COLUMN artifacts TEXT NOT NULL DEFAULT '[]';
