#!/bin/bash
# Production-запуск SemanticUA через Gunicorn.
#
# Якщо моделі ще не скачані — перший запит чекатиме завантаження
# (Bi-Encoder + Cross-Encoder, ~850 MB), наступні воркери підхоплять кеш.
#
# Параметри:
#   -w 4        — 4 воркери (по числу ядер CPU)
#   --threads 2 — 2 потоки на воркер → 8 паралельних запитів
#   --preload   — модель вантажиться у parent process, fork шарить пам'ять
#   --timeout 120 — CE-запит може займати кілька секунд
#
# Налаштування:
#   SECRET_KEY=... ./gunicorn.sh   # фіксований key для production
#   SESSION_TTL=7200 ./gunicorn.sh # 2 години замість 1

cd "$(dirname "$0")"

exec gunicorn \
  -w "${WORKERS:-4}" \
  --threads "${THREADS:-2}" \
  --bind "${BIND:-0.0.0.0:5000}" \
  --timeout 120 \
  --preload \
  --access-logfile - \
  --error-logfile - \
  app:app
