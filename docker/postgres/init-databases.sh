#!/bin/sh
set -eu

: "${POSTGRES_DB:?POSTGRES_DB is required}"
: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${CONTEXTGATE_MLFLOW_DATABASE_NAME:?CONTEXTGATE_MLFLOW_DATABASE_NAME is required}"

if [ "$CONTEXTGATE_MLFLOW_DATABASE_NAME" = "$POSTGRES_DB" ]; then
  echo "MLflow database is the primary application database; skipping extra database creation."
  exit 0
fi

database_exists="$(
  psql \
    -v ON_ERROR_STOP=1 \
    -v mlflow_db="$CONTEXTGATE_MLFLOW_DATABASE_NAME" \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    --tuples-only \
    --no-align <<-'EOSQL'
SELECT 1 FROM pg_database WHERE datname = :'mlflow_db';
EOSQL
)"

if [ "$database_exists" != "1" ]; then
  psql \
    -v ON_ERROR_STOP=1 \
    -v mlflow_db="$CONTEXTGATE_MLFLOW_DATABASE_NAME" \
    -v postgres_user="$POSTGRES_USER" \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" <<-'EOSQL'
CREATE DATABASE :"mlflow_db" OWNER :"postgres_user";
EOSQL
  echo "Created MLflow database: $CONTEXTGATE_MLFLOW_DATABASE_NAME"
else
  echo "MLflow database already exists: $CONTEXTGATE_MLFLOW_DATABASE_NAME"
fi

psql \
  -v ON_ERROR_STOP=1 \
  -v mlflow_db="$CONTEXTGATE_MLFLOW_DATABASE_NAME" \
  -v postgres_user="$POSTGRES_USER" \
  --username "$POSTGRES_USER" \
  --dbname "$POSTGRES_DB" <<-'EOSQL'
GRANT ALL PRIVILEGES ON DATABASE :"mlflow_db" TO :"postgres_user";
EOSQL
