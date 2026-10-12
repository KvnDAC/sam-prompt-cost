#!/bin/sh
# Ejecuta en orden las líneas de logs/queue.txt (se pueden añadir líneas mientras corre).
cd "$(dirname "$0")/.."
i=1
while true; do
  line=$(sed -n "${i}p" logs/queue.txt)
  if [ -z "$line" ]; then sleep 20; line=$(sed -n "${i}p" logs/queue.txt); [ -z "$line" ] && { echo "[queue] vacía, fin $(date)"; exit 0; }; fi
  echo "[queue] $(date +%H:%M:%S) START $i: $line"
  sh -c "$line" > "logs/job_$i.log" 2>&1
  echo "[queue] $(date +%H:%M:%S) END $i (exit $?)"
  i=$((i+1))
done
