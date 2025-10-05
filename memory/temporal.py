"""
Módulo de Relacionamento Temporal
Gerencia relações temporais e causais entre memórias
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)

# Constantes
MEMORY_INDEX = "claude_memory"


async def link_memories_timeline(
    es_client: Elasticsearch,
    memory_id: str,
    relations: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Adiciona relações temporais a uma memória

    Args:
        memory_id: ID da memória
        relations: {
            "happened_before": ["mem_id_1"],
            "happened_after": ["mem_id_2"],
            "led_to": ["mem_id_3"],
            "caused_by": ["mem_id_4"],
            "concurrent_with": ["mem_id_5"]
        }

    Returns:
        {"success": True, "id": "..."}
    """
    try:
        # Busca memória atual para pegar temporal_relations existentes
        try:
            existing = es_client.get(index=MEMORY_INDEX, id=memory_id)
            existing_relations = existing['_source'].get('temporal_relations', {})
        except Exception:
            existing_relations = {}

        # Merge com novas relações
        updated_relations = {**existing_relations}
        for relation_type, memory_ids in relations.items():
            if relation_type in updated_relations:
                # Adiciona sem duplicar
                existing_ids = set(updated_relations[relation_type])
                existing_ids.update(memory_ids)
                updated_relations[relation_type] = list(existing_ids)
            else:
                updated_relations[relation_type] = memory_ids

        # Atualiza memória
        es_client.update(
            index=MEMORY_INDEX,
            id=memory_id,
            body={
                "doc": {
                    "temporal_relations": updated_relations
                }
            },
            refresh="wait_for"
        )

        logger.info(f"✅ Relações temporais atualizadas: {memory_id}")
        return {"success": True, "id": memory_id, "relations": updated_relations}

    except Exception as e:
        logger.error(f"Erro ao adicionar relações temporais: {e}")
        return {"success": False, "error": str(e)}


async def get_memory_timeline(
    es_client: Elasticsearch,
    start_date: str,
    end_date: str,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Busca memórias em um período temporal

    Args:
        start_date: Data inicial ISO format (ex: "2025-01-01")
        end_date: Data final ISO format (ex: "2025-12-31")
        limit: Máximo de memórias

    Returns:
        Lista de memórias ordenadas por timestamp
    """
    try:
        response = es_client.search(
            index=MEMORY_INDEX,
            body={
                "query": {
                    "range": {
                        "timestamp": {
                            "gte": start_date,
                            "lte": end_date
                        }
                    }
                },
                "sort": [{"timestamp": {"order": "asc"}}],
                "size": limit,
                "_source": ["content", "timestamp", "type", "tags", "importance", "temporal_relations", "sequence_index"]
            }
        )

        memories = []
        for i, hit in enumerate(response['hits']['hits']):
            src = hit['_source']
            memories.append({
                "id": hit['_id'],
                "sequence_index": i,
                "content": src['content'],
                "timestamp": src['timestamp'],
                "type": src.get('type', 'general'),
                "tags": src.get('tags', []),
                "importance": src.get('importance', 5),
                "temporal_relations": src.get('temporal_relations', {})
            })

        logger.info(f"📅 Timeline: {len(memories)} memórias entre {start_date} e {end_date}")
        return memories

    except Exception as e:
        logger.error(f"Erro ao buscar timeline: {e}")
        return []


async def get_causal_chain(
    es_client: Elasticsearch,
    memory_id: str,
    max_depth: int = 5
) -> Dict[str, Any]:
    """
    Constrói cadeia causal a partir de uma memória

    Args:
        memory_id: ID da memória inicial
        max_depth: Profundidade máxima da busca

    Returns:
        {
            "root": {...},
            "causes": [...],  # O que causou esta memória
            "effects": [...],  # O que esta memória causou
            "concurrent": [...]  # Memórias concorrentes
        }
    """
    try:
        # Busca memória raiz
        root = es_client.get(index=MEMORY_INDEX, id=memory_id)
        root_data = {
            "id": memory_id,
            **root['_source']
        }

        relations = root['_source'].get('temporal_relations', {})

        # Busca causas (caused_by + happened_before)
        cause_ids = []
        cause_ids.extend(relations.get('caused_by', []))
        cause_ids.extend(relations.get('happened_before', []))

        causes = []
        if cause_ids:
            cause_response = es_client.mget(
                index=MEMORY_INDEX,
                body={"ids": cause_ids[:max_depth]}
            )
            for doc in cause_response['docs']:
                if doc['found']:
                    causes.append({
                        "id": doc['_id'],
                        **doc['_source']
                    })

        # Busca efeitos (led_to + happened_after)
        effect_ids = []
        effect_ids.extend(relations.get('led_to', []))
        effect_ids.extend(relations.get('happened_after', []))

        effects = []
        if effect_ids:
            effect_response = es_client.mget(
                index=MEMORY_INDEX,
                body={"ids": effect_ids[:max_depth]}
            )
            for doc in effect_response['docs']:
                if doc['found']:
                    effects.append({
                        "id": doc['_id'],
                        **doc['_source']
                    })

        # Busca concorrentes
        concurrent_ids = relations.get('concurrent_with', [])
        concurrent = []
        if concurrent_ids:
            concurrent_response = es_client.mget(
                index=MEMORY_INDEX,
                body={"ids": concurrent_ids[:max_depth]}
            )
            for doc in concurrent_response['docs']:
                if doc['found']:
                    concurrent.append({
                        "id": doc['_id'],
                        **doc['_source']
                    })

        logger.info(f"🔗 Cadeia causal: {len(causes)} causas, {len(effects)} efeitos, {len(concurrent)} concorrentes")

        return {
            "root": root_data,
            "causes": causes,
            "effects": effects,
            "concurrent": concurrent
        }

    except Exception as e:
        logger.error(f"Erro ao buscar cadeia causal: {e}")
        return {
            "root": None,
            "causes": [],
            "effects": [],
            "concurrent": [],
            "error": str(e)
        }


async def set_sequence_index(
    es_client: Elasticsearch,
    memory_id: str,
    sequence_index: int,
    conversation_id: str = ""
) -> Dict[str, Any]:
    """
    Define índice de sequência de uma memória em uma conversa

    Args:
        memory_id: ID da memória
        sequence_index: Posição na sequência
        conversation_id: ID da conversa

    Returns:
        {"success": True, "id": "..."}
    """
    try:
        es_client.update(
            index=MEMORY_INDEX,
            id=memory_id,
            body={
                "doc": {
                    "sequence_index": sequence_index,
                    "conversation_id": conversation_id
                }
            },
            refresh="wait_for"
        )

        logger.info(f"✅ Sequence index definido: {memory_id} = {sequence_index}")
        return {"success": True, "id": memory_id, "sequence_index": sequence_index}

    except Exception as e:
        logger.error(f"Erro ao definir sequence index: {e}")
        return {"success": False, "error": str(e)}
