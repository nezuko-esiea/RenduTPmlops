#!/bin/bash

API_URL="http://localhost:8000/predict"
IMAGE_PATH="geometrie/sample_images/test1.png"

DURATION_MINUTES=20
SLEEP_SECONDS=2

END_TIME=$((SECONDS + DURATION_MINUTES * 60))

echo "Démarrage du test de charge pendant $DURATION_MINUTES minutes..."
echo "API : $API_URL"
echo "Image : $IMAGE_PATH"
echo "Intervalle : $SLEEP_SECONDS secondes"
echo "--------------------------------------------"

REQUEST_COUNT=0

while [ $SECONDS -lt $END_TIME ]; do
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST \
    -F "file=@${IMAGE_PATH}" \
    $API_URL)

  REQUEST_COUNT=$((REQUEST_COUNT + 1))

  echo "[$(date '+%H:%M:%S')] Requête #$REQUEST_COUNT → HTTP $HTTP_CODE"

  sleep $SLEEP_SECONDS
done

echo "--------------------------------------------"
echo "Test terminé"
echo "Requêtes envoyées : $REQUEST_COUNT"
