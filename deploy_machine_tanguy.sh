#!/bin/bash

# Config
REMOTE_USER=tanguy
REMOTE_HOST=192.168.20.20
REMOTE_PATH=/home/tanguy/Bureau/python_projects/deep_sql
LOCAL_PATH=$(dirname "$0")

echo "Déploiement en cours depuis : $LOCAL_PATH"
echo "Vers : $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

# Transfert avec rsync, en forçant l’usage du SSH
rsync -avz -e "ssh" --delete "$LOCAL_PATH/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

echo "✅ Déploiement terminé."
