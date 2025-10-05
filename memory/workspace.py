"""
Módulo de Contexto de Trabalho
Gerencia estado do workspace (arquivos, terminal, navegador)
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)

# Constantes
WORKSPACE_INDEX = "claude_workspaces"


def ensure_workspace_index(es_client: Elasticsearch):
    """Cria índice de workspaces se não existir"""
    try:
        if not es_client.indices.exists(index=WORKSPACE_INDEX):
            mapping = {
                "mappings": {
                    "properties": {
                        "workspace_id": {"type": "keyword"},
                        "conversation_id": {"type": "keyword"},
                        "timestamp": {"type": "date"},
                        "active_files": {
                            "type": "nested",
                            "properties": {
                                "path": {"type": "text"},
                                "last_line_viewed": {"type": "integer"},
                                "modifications": {"type": "text"}
                            }
                        },
                        "current_directory": {"type": "keyword"},
                        "terminal_history": {"type": "text"},
                        "open_urls": {"type": "text"},
                        "clipboard_history": {"type": "text"}
                    }
                }
            }
            es_client.indices.create(index=WORKSPACE_INDEX, body=mapping)
            logger.info(f"Índice '{WORKSPACE_INDEX}' criado")
    except Exception as e:
        logger.error(f"Erro ao criar índice de workspaces: {e}")


async def save_workspace_context(
    es_client: Elasticsearch,
    active_files: List[Dict[str, Any]] = None,
    current_directory: str = "",
    terminal_history: List[str] = None,
    open_urls: List[str] = None,
    clipboard_history: List[str] = None,
    conversation_id: str = ""
) -> Dict[str, Any]:
    """
    Salva contexto do workspace atual

    Args:
        active_files: [{"path": "/path/to/file", "last_line_viewed": 123, "modifications": ["..."]}]
        current_directory: Diretório de trabalho atual
        terminal_history: Lista de comandos executados
        open_urls: URLs abertas no navegador
        clipboard_history: Histórico da área de transferência
        conversation_id: ID da conversa relacionada

    Returns:
        {"success": True, "workspace_id": "...", "id": "..."}
    """
    try:
        ensure_workspace_index(es_client)

        workspace_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        document = {
            "workspace_id": workspace_id,
            "conversation_id": conversation_id,
            "timestamp": now,
            "active_files": active_files or [],
            "current_directory": current_directory,
            "terminal_history": terminal_history or [],
            "open_urls": open_urls or [],
            "clipboard_history": clipboard_history or []
        }

        response = es_client.index(
            index=WORKSPACE_INDEX,
            document=document,
            refresh="wait_for"
        )

        logger.info(f"✅ Workspace salvo: {workspace_id}")
        return {
            "success": True,
            "workspace_id": workspace_id,
            "id": response['_id'],
            "timestamp": now
        }

    except Exception as e:
        logger.error(f"Erro ao salvar workspace: {e}")
        return {"success": False, "error": str(e)}


async def restore_workspace(
    es_client: Elasticsearch,
    workspace_id: str
) -> Optional[Dict[str, Any]]:
    """
    Restaura contexto de um workspace

    Args:
        workspace_id: ID do workspace

    Returns:
        Dict com todos os dados do workspace ou None
    """
    try:
        response = es_client.search(
            index=WORKSPACE_INDEX,
            body={
                "query": {"term": {"workspace_id": workspace_id}},
                "size": 1
            }
        )

        if response['hits']['hits']:
            hit = response['hits']['hits'][0]
            workspace_data = hit['_source']

            logger.info(f"✅ Workspace restaurado: {workspace_id}")
            return {
                "id": hit['_id'],
                **workspace_data
            }

        logger.warning(f"Workspace não encontrado: {workspace_id}")
        return None

    except Exception as e:
        logger.error(f"Erro ao restaurar workspace: {e}")
        return None


async def get_recent_workspaces(
    es_client: Elasticsearch,
    limit: int = 3,
    conversation_id: str = None
) -> List[Dict[str, Any]]:
    """
    Busca workspaces mais recentes

    Args:
        limit: Número máximo de workspaces
        conversation_id: Filtrar por conversa específica

    Returns:
        Lista de workspaces ordenados por timestamp (mais recente primeiro)
    """
    try:
        ensure_workspace_index(es_client)

        query = {"match_all": {}}
        if conversation_id:
            query = {"term": {"conversation_id": conversation_id}}

        response = es_client.search(
            index=WORKSPACE_INDEX,
            body={
                "query": query,
                "sort": [{"timestamp": {"order": "desc"}}],
                "size": limit
            }
        )

        workspaces = []
        for hit in response['hits']['hits']:
            workspaces.append({
                "id": hit['_id'],
                **hit['_source']
            })

        logger.info(f"💼 Listados {len(workspaces)} workspaces recentes")
        return workspaces

    except Exception as e:
        logger.error(f"Erro ao listar workspaces: {e}")
        return []


async def update_workspace_files(
    es_client: Elasticsearch,
    workspace_id: str,
    active_files: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Atualiza lista de arquivos ativos em um workspace

    Args:
        workspace_id: ID do workspace
        active_files: Nova lista de arquivos

    Returns:
        {"success": True, "id": "..."}
    """
    try:
        response = es_client.update_by_query(
            index=WORKSPACE_INDEX,
            body={
                "script": {
                    "source": "ctx._source.active_files = params.files",
                    "lang": "painless",
                    "params": {
                        "files": active_files
                    }
                },
                "query": {"term": {"workspace_id": workspace_id}}
            },
            refresh=True
        )

        logger.info(f"✅ Arquivos do workspace atualizados: {workspace_id}")
        return {"success": True, "updated": response['updated']}

    except Exception as e:
        logger.error(f"Erro ao atualizar arquivos: {e}")
        return {"success": False, "error": str(e)}


async def delete_workspace(
    es_client: Elasticsearch,
    workspace_id: str
) -> Dict[str, Any]:
    """
    Deleta um workspace
    """
    try:
        response = es_client.delete_by_query(
            index=WORKSPACE_INDEX,
            body={
                "query": {"term": {"workspace_id": workspace_id}}
            },
            refresh=True
        )

        logger.info(f"✅ Workspace deletado: {workspace_id}")
        return {"success": True, "deleted": response['deleted']}

    except Exception as e:
        logger.error(f"Erro ao deletar workspace: {e}")
        return {"success": False, "error": str(e)}
