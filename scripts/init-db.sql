-- LatentSpec database initialization
-- Enables TimescaleDB and pgvector extensions per §7 / §8.2

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
