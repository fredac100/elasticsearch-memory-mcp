"""
Módulo de Memória de Sessão/Conversa
Gerencia snapshots de conversas e continuidade de contexto
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)

# Constantes
SESSION_INDEX = "claude_conversations"


def ensure_session_index(es_client: Elasticsearch):
    """Cria índice de sessões se não existir"""
    try:
        if not es_client.indices.exists(index=SESSION_INDEX):
            mapping = {
                "mappings": {
                    "properties": {
                        "conversation_id": {"type": "keyword"},
                        "chat_platform_id": {"type": "keyword"},
                        "timestamp_start": {"type": "date"},
                        "timestamp_last": {"type": "date"},
                        "message_count": {"type": "integer"},
                        "last_messages": {
                            "type": "nested",
                            "properties": {
                                "role": {"type": "keyword"},
                                "content": {"type": "text"},
                                "timestamp": {"type": "date"}
                            }
                        },
                        "current_topic": {"type": "text"},
                        "conversation_flow": {"type": "keyword"},
                        "active": {"type": "boolean"}
                    }
                }
            }
            es_client.indices.create(index=SESSION_INDEX, body=mapping)
            logger.info(f"Índice '{SESSION_INDEX}' criado")
    except Exception as e:
        logger.error(f"Erro ao criar índice de sessões: {e}")


async def save_conversation_snapshot(
    es_client: Elasticsearch,
    conversation_id: str,
    last_messages: List[Dict[str, str]],
    current_topic: str = "",
    conversation_flow: List[str] = None,
    chat_platform_id: str = ""
) -> Dict[str, Any]:
    """
    Salva snapshot da conversa atual

    Args:
        conversation_id: ID único da conversa
        last_messages: Últimas N mensagens [{"role": "user|assistant", "content": "...", "timestamp": "ISO"}]
        current_topic: Tópico atual da conversa
        conversation_flow: Fluxo de tópicos ["intro", "debugging", "implementation"]
        chat_platform_id: ID do chat na plataforma (ex: claude_chat_id)
    """
    try:
        ensure_session_index(es_client)

        now = datetime.utcnow().isoformat()

        # Busca conversa existente
        try:
            response = es_client.search(
                index=SESSION_INDEX,
                body={
                    "query": {"term": {"conversation_id": conversation_id}},
                    "size": 1
                }
            )

            if response['hits']['hits']:
                # Atualiza conversa existente
                doc_id = response['hits']['hits'][0]['_id']
                existing = response['hits']['hits'][0]['_source']

                document = {
                    "conversation_id": conversation_id,
                    "chat_platform_id": chat_platform_id or existing.get('chat_platform_id', ''),
                    "timestamp_start": existing['timestamp_start'],
                    "timestamp_last": now,
                    "message_count": existing.get('message_count', 0) + len(last_messages),
                    "last_messages": last_messages,
                    "current_topic": current_topic or existing.get('current_topic', ''),
                    "conversation_flow": conversation_flow or existing.get('conversation_flow', []),
                    "active": True
                }

                es_client.update(
                    index=SESSION_INDEX,
                    id=doc_id,
                    body={"doc": document},
                    refresh="wait_for"
                )

                logger.info(f"✅ Conversa atualizada: {conversation_id}")
                return {"success": True, "id": doc_id, "updated": True}

        except Exception:
            pass

        # Cria nova conversa
        document = {
            "conversation_id": conversation_id,
            "chat_platform_id": chat_platform_id,
            "timestamp_start": now,
            "timestamp_last": now,
            "message_count": len(last_messages),
            "last_messages": last_messages,
            "current_topic": current_topic,
            "conversation_flow": conversation_flow or [],
            "active": True
        }

        response = es_client.index(
            index=SESSION_INDEX,
            document=document,
            refresh="wait_for"
        )

        logger.info(f"✅ Nova conversa salva: {conversation_id}")
        return {"success": True, "id": response['_id'], "created": True}

    except Exception as e:
        logger.error(f"Erro ao salvar snapshot: {e}")
        return {"success": False, "error": str(e)}


async def get_active_conversation(es_client: Elasticsearch) -> Optional[Dict[str, Any]]:
    """
    Busca a conversa ativa mais recente
    """
    try:
        ensure_session_index(es_client)

        response = es_client.search(
            index=SESSION_INDEX,
            body={
                "query": {"term": {"active": True}},
                "sort": [{"timestamp_last": {"order": "desc"}}],
                "size": 1
            }
        )

        if response['hits']['hits']:
            hit = response['hits']['hits'][0]
            return {
                "id": hit['_id'],
                **hit['_source']
            }

        return None

    except Exception as e:
        logger.error(f"Erro ao buscar conversa ativa: {e}")
        return None


async def close_conversation(
    es_client: Elasticsearch,
    conversation_id: str
) -> Dict[str, Any]:
    """
    Marca uma conversa como inativa/encerrada
    """
    try:
        response = es_client.update_by_query(
            index=SESSION_INDEX,
            body={
                "script": {
                    "source": "ctx._source.active = false",
                    "lang": "painless"
                },
                "query": {"term": {"conversation_id": conversation_id}}
            },
            refresh=True
        )

        logger.info(f"✅ Conversa encerrada: {conversation_id}")
        return {"success": True, "updated": response['updated']}

    except Exception as e:
        logger.error(f"Erro ao encerrar conversa: {e}")
        return {"success": False, "error": str(e)}


async def list_conversations(
    es_client: Elasticsearch,
    limit: int = 10,
    active_only: bool = False
) -> List[Dict[str, Any]]:
    """
    Lista conversas com opção de filtrar apenas ativas
    """
    try:
        ensure_session_index(es_client)

        query = {"match_all": {}}
        if active_only:
            query = {"term": {"active": True}}

        response = es_client.search(
            index=SESSION_INDEX,
            body={
                "query": query,
                "sort": [{"timestamp_last": {"order": "desc"}}],
                "size": limit
            }
        )

        conversations = []
        for hit in response['hits']['hits']:
            conversations.append({
                "id": hit['_id'],
                **hit['_source']
            })

        return conversations

    except Exception as e:
        logger.error(f"Erro ao listar conversas: {e}")
        return []
