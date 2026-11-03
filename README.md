# Tech Challenge - Fase 3: Triagem Automática de Laudos Médicos

## 1. Decisão Arquitetural de Nuvem
Optou-se por uma **arquitetura Serverless em Containers (Real-Time)** utilizando AWS ECS Fargate, Azure Container Apps ou Google Cloud Run.
- **Inferência em Tempo Real**: Atende aos requisitos médicos de classificação imediata de urgências.
- **Containers**: Facilita a migração direta do ecossistema de desenvolvimento local para o ambiente de produção.

## 2. Como Executar Localmente

### Usando Docker Compose
```powershell
docker compose up --build -d