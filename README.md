# Генератор задач линейного программирования с решениями

Данный курсовой проект разработан для автоматической генерации уникальных задач по линейному программированию (ЛП) с готовыми решениями. Цель проекта – создать множество вариантов контрольных работ по предмету «Исследование операций», чтобы каждая задача была по сложности примерно одинаковой, а проверка студенческих решений упрощалась за счёт автоматического сохранения оптимальных ответов и пошаговых логов симплекс-метода.

## Что делает проект?
- **Генерация задач ЛП**: случайным образом задаёт количество переменных, ограничений и типы ограничений (≤, ≥, =).
- **Решение задач**: использует двухфазный симплекс-метод для поиска оптимального решения, с подробным логированием каждого шага.
- **Формирование отчётов**: результаты сохраняются в pdf файле и в LaTeX-формате.

## Что нужно сделать для запуска?
1. **Установите зависимости.** Все нужные библиотеки перечислены в файле `requirements.txt`.
2. **Запустите основной скрипт** (например, `python main.py`). После запуска в корневой папке появятся:
   - Файл `tasks_solutions.pdf` (условия, решения, шаги симплекс-метода).
   - Файл `tasks_solutions.tex` (затеханный вид pdf).
3. **Примеры** уже находятся в папке `samples` для ознакомления с форматами вывода.

## Структура проекта
- **src/main.py** – основной файл, в котором собраны функции для генерации задач, их решения и формирования отчётов.
- **requirements.txt** – список зависимостей.
- **samples/** – примеры сгенерированных отчётов (текстовых и PDF-файлов).

## Автор
Диёр Султонов