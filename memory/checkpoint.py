"""
Módulo de Checkpoint de Estado
Salva e restaura o estado de trabalho em pontos específicos
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)

# Constantes
CHECKPOINT_INDEX = "claude_checkpoints"


def ensure_checkpoint_index(es_client: Elasticsearch):
    """Cria índice de checkpoints se não existir"""
    try:
        if not es_client.indices.exists(index=CHECKPOINT_INDEX):
            mapping = {
                "mappings": {
                    "properties": {
                        "checkpoint_id": {"type": "keyword"},
                        "conversation_id": {"type": "keyword"},
                        "timestamp": {"type": "date"},
                        "working_on": {"type": "text"},
                        "context": {
                            "type": "object",
                            "properties": {
                                "project": {"type": "keyword"},
                                "current_problem": {"type": "text"},
                                "solved_problems": {"type": "text"}
                            }
                        },
                        "next_steps": {"type": "text"},
                        "open_questions": {"type": "text"},
                        "decisions_made": {"type": "text"}
                    }
                }
            }
            es_client.indices.create(index=CHECKPOINT_INDEX, body=mapping)
            logger.info(f"Índice '{CHECKPOINT_INDEX}' criado")
    except Exception as e:
        logger.error(f"Erro ao criar índice de checkpoints: {e}")


async def create_checkpoint(
    es_client: Elasticsearch,
    working_on: str,
    context: Dict[str, Any] = None,
    next_steps: List[str] = None,
    open_questions: List[str] = None,
    decisions_made: List[str] = None,
    conversation_id: str = ""
) -> Dict[str, Any]:
    """
    Cria um checkpoint do estado atual

    Args:
        working_on: Descrição do que está sendo trabalhado
        context: Contexto adicional {"project": "...", "current_problem": "...", "solved_problems": [...]}
        next_steps: Lista de próximos passos
        open_questions: Perguntas em aberto
        decisions_made: Decisões tomadas
        conversation_id: ID da conversa relacionada

    Returns:
        {"success": True, "checkpoint_id": "...", "id": "..."}
    """
    try:
        ensure_checkpoint_index(es_client)

        checkpoint_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        document = {
            "checkpoint_id": checkpoint_id,
            "conversation_id": conversation_id,
            "timestamp": now,
            "working_on": working_on,
            "context": context or {},
            "next_steps": next_steps or [],
            "open_questions": open_questions or [],
            "decisions_made": decisions_made or []
        }

        response = es_client.index(
            index=CHECKPOINT_INDEX,
            document=document,
            refresh="wait_for"
        )

        logger.info(f"✅ Checkpoint criado: {checkpoint_id}")
        return {
            "success": True,
            "checkpoint_id": checkpoint_id,
            "id": response['_id'],
            "timestamp": now
        }

    except Exception as e:
        logger.error(f"Erro ao criar checkpoint: {e}")
        return {"success": False, "error": str(e)}


async def restore_from_checkpoint(
    es_client: Elasticsearch,
    checkpoint_id: str
) -> Optional[Dict[str, Any]]:
    """
    Restaura estado de um checkpoint específico

    Args:
        checkpoint_id: ID do checkpoint

    Returns:
        Dict com todos os dados do checkpoint ou None se não encontrado
    """
    try:
        response = es_client.search(
            index=CHECKPOINT_INDEX,
            body={
                "query": {"term": {"checkpoint_id": checkpoint_id}},
                "size": 1
            }
        )

        if response['hits']['hits']:
            hit = response['hits']['hits'][0]
            checkpoint_data = hit['_source']

            logger.info(f"✅ Checkpoint restaurado: {checkpoint_id}")
            return {
                "id": hit['_id'],
                **checkpoint_data
            }

        logger.warning(f"Checkpoint não encontrado: {checkpoint_id}")
        return None

    except Exception as e:
        logger.error(f"Erro ao restaurar checkpoint: {e}")
        return None


async def list_checkpoints(
    es_client: Elasticsearch,
    limit: int = 5,
    conversation_id: str = None
) -> List[Dict[str, Any]]:
    """
    Lista checkpoints mais recentes

    Args:
        limit: Número máximo de checkpoints
        conversation_id: Filtrar por conversa específica

    Returns:
        Lista de checkpoints ordenados por timestamp (mais recente primeiro)
    """
    try:
        ensure_checkpoint_index(es_client)

        query = {"match_all": {}}
        if conversation_id:
            query = {"term": {"conversation_id": conversation_id}}

        response = es_client.search(
            index=CHECKPOINT_INDEX,
            body={
                "query": query,
                "sort": [{"timestamp": {"order": "desc"}}],
                "size": limit
            }
        )

        checkpoints = []
        for hit in response['hits']['hits']:
            checkpoints.append({
                "id": hit['_id'],
                **hit['_source']
            })

        logger.info(f"📋 Listados {len(checkpoints)} checkpoints")
        return checkpoints

    except Exception as e:
        logger.error(f"Erro ao listar checkpoints: {e}")
        return []


async def get_latest_checkpoint(
    es_client: Elasticsearch,
    conversation_id: str = None
) -> Optional[Dict[str, Any]]:
    """
    Busca o checkpoint mais recente

    Args:
        conversation_id: Filtrar por conversa específica

    Returns:
        Checkpoint mais recente ou None
    """
    try:
        checkpoints = await list_checkpoints(es_client, limit=1, conversation_id=conversation_id)
        return checkpoints[0] if checkpoints else None

    except Exception as e:
        logger.error(f"Erro ao buscar último checkpoint: {e}")
        return None


async def delete_checkpoint(
    es_client: Elasticsearch,
    checkpoint_id: str
) -> Dict[str, Any]:
    """
    Deleta um checkpoint específico
    """
    try:
        response = es_client.delete_by_query(
            index=CHECKPOINT_INDEX,
            body={
                "query": {"term": {"checkpoint_id": checkpoint_id}}
            },
            refresh=True
        )

        logger.info(f"✅ Checkpoint deletado: {checkpoint_id}")
        return {"success": True, "deleted": response['deleted']}

    except Exception as e:
        logger.error(f"Erro ao deletar checkpoint: {e}")
        return {"success": False, "error": str(e)}
