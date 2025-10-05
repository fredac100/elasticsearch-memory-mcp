"""
Módulo de Memória Emocional/Tom
Rastreia tom, humor e preferências de comunicação
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)

# Constantes
TONE_INDEX = "claude_tone_profiles"


def ensure_tone_index(es_client: Elasticsearch):
    """Cria índice de perfis de tom se não existir"""
    try:
        if not es_client.indices.exists(index=TONE_INDEX):
            mapping = {
                "mappings": {
                    "properties": {
                        "tone_id": {"type": "keyword"},
                        "conversation_id": {"type": "keyword"},
                        "user_id": {"type": "keyword"},
                        "timestamp": {"type": "date"},
                        "mood": {"type": "text"},
                        "energy_level": {"type": "keyword"},
                        "rapport_level": {"type": "float"},
                        "communication_style": {
                            "type": "object",
                            "properties": {
                                "formality": {"type": "keyword"},
                                "technical_depth": {"type": "keyword"},
                                "humor_used": {"type": "boolean"}
                            }
                        },
                        "inside_jokes": {"type": "text"},
                        "user_preferences_observed": {"type": "object"}
                    }
                }
            }
            es_client.indices.create(index=TONE_INDEX, body=mapping)
            logger.info(f"Índice '{TONE_INDEX}' criado")
    except Exception as e:
        logger.error(f"Erro ao criar índice de tom: {e}")


async def save_conversation_tone(
    es_client: Elasticsearch,
    mood: str = "neutral",
    energy_level: str = "medium",
    rapport_level: float = 5.0,
    communication_style: Dict[str, Any] = None,
    inside_jokes: List[str] = None,
    user_preferences_observed: Dict[str, Any] = None,
    conversation_id: str = "",
    user_id: str = "default_user"
) -> Dict[str, Any]:
    """
    Salva perfil de tom da conversa

    Args:
        mood: Humor da conversa (ex: "debugging produtivo", "explorando ideias")
        energy_level: Nível de energia ("low", "medium", "high")
        rapport_level: Nível de rapport 0-10
        communication_style: {
            "formality": "informal|formal|professional",
            "technical_depth": "shallow|medium|deep",
            "humor_used": true|false,
            "cursing_comfort": "none|low|medium|high"
        }
        inside_jokes: Lista de piadas internas
        user_preferences_observed: {
            "likes_direct_solutions": true,
            "appreciates_technical_details": true,
            "prefers_pragmatism": true
        }
        conversation_id: ID da conversa
        user_id: ID do usuário

    Returns:
        {"success": True, "tone_id": "...", "id": "..."}
    """
    try:
        ensure_tone_index(es_client)

        tone_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        document = {
            "tone_id": tone_id,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "timestamp": now,
            "mood": mood,
            "energy_level": energy_level,
            "rapport_level": min(max(rapport_level, 0.0), 10.0),
            "communication_style": communication_style or {
                "formality": "informal",
                "technical_depth": "medium",
                "humor_used": False
            },
            "inside_jokes": inside_jokes or [],
            "user_preferences_observed": user_preferences_observed or {}
        }

        response = es_client.index(
            index=TONE_INDEX,
            document=document,
            refresh="wait_for"
        )

        logger.info(f"✅ Perfil de tom salvo: {tone_id}")
        return {
            "success": True,
            "tone_id": tone_id,
            "id": response['_id'],
            "timestamp": now
        }

    except Exception as e:
        logger.error(f"Erro ao salvar tom: {e}")
        return {"success": False, "error": str(e)}


async def get_tone_profile(
    es_client: Elasticsearch,
    user_id: str = "default_user",
    limit: int = 10
) -> Dict[str, Any]:
    """
    Busca perfil de tom agregado de um usuário

    Args:
        user_id: ID do usuário
        limit: Número de conversas recentes para analisar

    Returns:
        Perfil agregado com preferências e padrões observados
    """
    try:
        ensure_tone_index(es_client)

        # Busca registros mais recentes do usuário
        response = es_client.search(
            index=TONE_INDEX,
            body={
                "query": {"term": {"user_id": user_id}},
                "sort": [{"timestamp": {"order": "desc"}}],
                "size": limit
            }
        )

        if not response['hits']['hits']:
            logger.info(f"Nenhum perfil encontrado para: {user_id}")
            return {
                "user_id": user_id,
                "samples": 0,
                "average_rapport": 5.0,
                "common_moods": [],
                "communication_style": {},
                "inside_jokes": [],
                "preferences": {}
            }

        # Agrega dados
        samples = []
        all_moods = []
        all_jokes = []
        rapport_sum = 0.0
        style_counts = {
            "formality": {},
            "technical_depth": {},
            "humor_used": 0
        }
        preferences = {}

        for hit in response['hits']['hits']:
            src = hit['_source']
            samples.append(src)
            all_moods.append(src.get('mood', 'neutral'))
            rapport_sum += src.get('rapport_level', 5.0)
            all_jokes.extend(src.get('inside_jokes', []))

            # Agrega estilo de comunicação
            style = src.get('communication_style', {})
            if 'formality' in style:
                formality = style['formality']
                style_counts['formality'][formality] = style_counts['formality'].get(formality, 0) + 1

            if 'technical_depth' in style:
                depth = style['technical_depth']
                style_counts['technical_depth'][depth] = style_counts['technical_depth'].get(depth, 0) + 1

            if style.get('humor_used'):
                style_counts['humor_used'] += 1

            # Agrega preferências
            user_prefs = src.get('user_preferences_observed', {})
            for key, value in user_prefs.items():
                if key not in preferences:
                    preferences[key] = []
                preferences[key].append(value)

        # Calcula agregados
        avg_rapport = rapport_sum / len(samples)

        # Estilo mais comum
        most_common_formality = max(style_counts['formality'].items(), key=lambda x: x[1])[0] if style_counts['formality'] else "informal"
        most_common_depth = max(style_counts['technical_depth'].items(), key=lambda x: x[1])[0] if style_counts['technical_depth'] else "medium"
        humor_percentage = (style_counts['humor_used'] / len(samples)) * 100

        # Preferências consolidadas (maioria)
        consolidated_prefs = {}
        for key, values in preferences.items():
            if values:
                # Pega o valor mais comum
                most_common = max(set(values), key=values.count)
                consolidated_prefs[key] = most_common

        # Moods únicos (frequência)
        from collections import Counter
        mood_counter = Counter(all_moods)
        common_moods = [mood for mood, count in mood_counter.most_common(3)]

        # Piadas únicas
        unique_jokes = list(set(all_jokes))

        profile = {
            "user_id": user_id,
            "samples": len(samples),
            "average_rapport": round(avg_rapport, 2),
            "common_moods": common_moods,
            "communication_style": {
                "formality": most_common_formality,
                "technical_depth": most_common_depth,
                "humor_used": humor_percentage > 50,
                "humor_percentage": round(humor_percentage, 1)
            },
            "inside_jokes": unique_jokes[:10],  # Top 10
            "preferences": consolidated_prefs
        }

        logger.info(f"📊 Perfil de tom agregado: {user_id} ({len(samples)} amostras)")
        return profile

    except Exception as e:
        logger.error(f"Erro ao buscar perfil de tom: {e}")
        return {
            "user_id": user_id,
            "samples": 0,
            "error": str(e)
        }


async def adapt_to_tone(
    es_client: Elasticsearch,
    tone_id: str
) -> Optional[Dict[str, Any]]:
    """
    Busca um perfil de tom específico para adaptar comunicação

    Args:
        tone_id: ID do perfil de tom

    Returns:
        Dados do perfil ou None
    """
    try:
        response = es_client.search(
            index=TONE_INDEX,
            body={
                "query": {"term": {"tone_id": tone_id}},
                "size": 1
            }
        )

        if response['hits']['hits']:
            hit = response['hits']['hits'][0]
            tone_data = hit['_source']

            logger.info(f"✅ Perfil de tom carregado: {tone_id}")
            return {
                "id": hit['_id'],
                **tone_data
            }

        logger.warning(f"Perfil de tom não encontrado: {tone_id}")
        return None

    except Exception as e:
        logger.error(f"Erro ao carregar tom: {e}")
        return None


async def list_tone_profiles(
    es_client: Elasticsearch,
    user_id: str = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Lista perfis de tom

    Args:
        user_id: Filtrar por usuário específico
        limit: Número máximo de perfis

    Returns:
        Lista de perfis ordenados por timestamp
    """
    try:
        ensure_tone_index(es_client)

        query = {"match_all": {}}
        if user_id:
            query = {"term": {"user_id": user_id}}

        response = es_client.search(
            index=TONE_INDEX,
            body={
                "query": query,
                "sort": [{"timestamp": {"order": "desc"}}],
                "size": limit
            }
        )

        profiles = []
        for hit in response['hits']['hits']:
            profiles.append({
                "id": hit['_id'],
                **hit['_source']
            })

        logger.info(f"😊 Listados {len(profiles)} perfis de tom")
        return profiles

    except Exception as e:
        logger.error(f"Erro ao listar perfis: {e}")
        return []
