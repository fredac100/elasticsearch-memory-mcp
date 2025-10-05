#!/usr/bin/env python3
"""
MCP Server v6 - SISTEMA DE CATEGORIZAÇÃO E PRIORIZAÇÃO DE MEMÓRIAS

Novas Features v6:
1. ✅ Campo memory_category (identity, active_context, active_project, technical_knowledge, archived)
2. ✅ Load inicial hierárquico e otimizado (~30-40 memórias vs 117)
3. ✅ Auto-detecção inteligente de categoria com confidence
4. ✅ Tool get_identity_core (apenas memórias identity)
5. ✅ Tool auto_categorize_memories (categorização batch com aprovação)
6. ✅ Tool migrate_to_archive (arquivamento inteligente)
7. ✅ Tool recategorize_memory (reclassificação manual)

Mantém todas features v5:
- Auto-save de conversa completa
- Load inteligente expandido
- Auto-anotação inteligente
- Resumo periódico automático
"""

import asyncio
import json
import logging
import os
import re
import time
import hashlib
import signal
import atexit
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Resource, Prompt

# Importa módulos v2.0
from memory.session import (
    save_conversation_snapshot,
    get_active_conversation,
    close_conversation,
    list_conversations,
    ensure_session_index
)
from memory.checkpoint import (
    create_checkpoint,
    restore_from_checkpoint,
    list_checkpoints,
    get_latest_checkpoint,
    delete_checkpoint,
    ensure_checkpoint_index
)
from memory.temporal import (
    link_memories_timeline,
    get_memory_timeline,
    get_causal_chain,
    set_sequence_index
)
from memory.workspace import (
    save_workspace_context,
    restore_workspace,
    get_recent_workspaces,
    update_workspace_files,
    delete_workspace,
    ensure_workspace_index
)
from memory.emotional import (
    save_conversation_tone,
    get_tone_profile,
    adapt_to_tone,
    list_tone_profiles,
    ensure_tone_index
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurações
ES_HOST = "http://localhost:9200"
INDEX_NAME = "claude_memory"
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
SOURCE = os.getenv("SOURCE", "unknown")  # claude-code, claude-desktop, etc.

# V5: Configurações de auto-save e resumo
AUTO_SAVE_TIMEOUT = 300  # 5 minutos de inatividade
AUTO_SUMMARY_INTERVAL = 50  # A cada 50 mensagens
MESSAGE_HISTORY_SIZE = 100  # Mantém últimas 100 mensagens em memória

# 🆕 V6: Categorias de memória
MEMORY_CATEGORIES = {
    "identity": "Quem sou, valores, personalidade, como me comunico",
    "active_context": "Última conversa, checkpoint atual, working_on",
    "active_project": "Projetos em andamento (SAE, Mirror, Nexus)",
    "technical_knowledge": "Fatos técnicos, configurações, ferramentas",
    "archived": "Memórias concluídas, problemas resolvidos, histórico"
}

# Cliente Elasticsearch
es_client = Elasticsearch([ES_HOST])

# Modelo de embeddings
logger.info(f"Carregando modelo de embeddings: {MODEL_NAME}")
embedding_model = SentenceTransformer(MODEL_NAME)
logger.info("Modelo carregado com sucesso")

# Cache de queries (5min TTL)
query_cache: Dict[str, Tuple[List[Dict], float]] = {}
CACHE_TTL = 300  # 5 minutos

# V5: Estado global da sessão
class SessionState:
    def __init__(self):
        self.conversation_id: Optional[str] = None
        self.message_count = 0
        self.message_history = deque(maxlen=MESSAGE_HISTORY_SIZE)
        self.last_activity = time.time()
        self.auto_annotations: List[Dict] = []
        self.current_topic: Optional[str] = None
        self.activity_timer = None

session_state = SessionState()


# ========== 🆕 V6: AUTO-DETECÇÃO DE CATEGORIA ==========

def detect_memory_category(content: str, metadata: Dict = None) -> Tuple[str, float]:
    """
    🆕 V6.2: Detecta categoria com SCORING INTELIGENTE
    
    Melhorias V6.2:
    - Keywords específicas (refatoração→archived, primeira vez→identity, etc)
    - Scoring acumulativo: múltiplas keywords aumentam confidence
    - Detecção de contexto melhorada
    
    Returns:
        (category, confidence) - categoria e nível de confiança (0.0-1.0)
    """
    if metadata is None:
        metadata = {}
    
    content_lower = content.lower()
    mem_type = metadata.get('type', 'general')
    tags = metadata.get('tags', [])
    
    # Sistema de scoring: cada categoria acumula pontos
    scores = {
        "identity": 0.0,
        "active_context": 0.0,
        "active_project": 0.0,
        "archived": 0.0,
        "technical_knowledge": 0.0
    }
    
    # ========== IDENTITY ==========
    
    identity_patterns = {
        'explicit_self': [  # Força 0.95
            r'\b(meu nome|me chamo|eu sou|sou uma?)\b',
            r'\b(minha personalidade|meu estilo|minha forma de)\b',
            r'\b(valores que sigo|acredito que|para mim)\b',
            r'\b(aprendi sobre mim|descobri que eu|percebi que sou)\b'
        ],
        'user_preferences': [  # Força 0.9
            r'\b(fred|usuário)\b.*(prefere|gosta|quer|deseja|espera).*(que eu|claude)',
            r'\b(preferência|estilo|forma).*(do usuário|de fred)',
            r'\b(avisar|lembrar|sempre fazer|nunca fazer)\b',
            r'\b(comunicação brutal|direto|sem rodeios)\b',
            r'\bbrutalidade computacional\b',
            r'\b(fred trabalha|fred usa|fred é|contexto.*fred)\b'
        ],
        'communication_style': [  # Força 0.85
            r'\b(gosto de|prefiro|costumo).*(comunicar|falar|responder|trabalhar)\b',
            r'\b(meu tom|minha abordagem|meu método)\b',
            r'\b(como eu.*(ajudo|respondo|trabalho|penso))\b'
        ],
        'unique_experiences': [  # 🆕 V6.2: Força 0.8
            r'\b(primeira vez|experiência única|nunca tinha)\b',
            r'\b(presente|ganhou|recebeu).*\b(especial|importante)\b',
            r'\b(momento marcante|ocasião especial)\b'
        ]
    }
    
    for strength, patterns in identity_patterns.items():
        for pattern in patterns:
            if re.search(pattern, content_lower):
                if strength == 'explicit_self':
                    scores['identity'] += 0.95
                elif strength == 'user_preferences':
                    scores['identity'] += 0.9
                elif strength == 'communication_style':
                    scores['identity'] += 0.85
                elif strength == 'unique_experiences':
                    scores['identity'] += 0.8
    
    if 'identity' in tags or 'self' in tags or 'user_profile' in tags or 'preference' in tags:
        scores['identity'] += 0.75
    if mem_type in ['self_identity', 'user_profile', 'preference']:
        scores['identity'] += 0.7
    
    # ========== ACTIVE_CONTEXT ==========
    
    active_context_patterns = {
        'current_work': [  # Força 0.9
            r'\b(estou trabalhando|trabalhando em|focado em|fazendo agora)\b',
            r'\b(agora estou|atualmente|no momento|neste momento)\b',
            r'\b(última conversa|acabamos de|conversamos sobre)\b'
        ],
        'checkpoint': [  # Força 0.85
            r'\b(checkpoint|ponto atual|estado atual|onde parei)\b'
        ],
        'planning': [  # 🆕 V6.2: Força 0.8
            r'\b(próximos passos|vou fazer|planejando|pretendo)\b',
            r'\b(amanhã|esta semana|em breve)\b'
        ]
    }
    
    for strength, patterns in active_context_patterns.items():
        for pattern in patterns:
            if re.search(pattern, content_lower):
                if strength == 'current_work':
                    scores['active_context'] += 0.9
                elif strength == 'checkpoint':
                    scores['active_context'] += 0.85
                elif strength == 'planning':
                    scores['active_context'] += 0.8
    
    # ========== ACTIVE_PROJECT ==========
    
    active_project_patterns = {
        'specific_projects': [  # Força 0.9
            r'\b(SAE|Mirror|Nexus)\b.*(desenvolvendo|construindo|implementando|criando|trabalhando)',
            r'\b(desenvolvendo|construindo|implementando).*(SAE|Mirror|Nexus)\b',
            r'\b(projeto|sistema).*(em andamento|ativo|atual)\b'
        ],
        'implementation': [  # Força 0.8
            r'\b(implementando|criando|desenvolvendo).*(sistema|aplicação|módulo|feature)\b',
            r'\b(em desenvolvimento|em progresso|working on)\b'
        ]
    }
    
    for strength, patterns in active_project_patterns.items():
        for pattern in patterns:
            if re.search(pattern, content_lower):
                if strength == 'specific_projects':
                    scores['active_project'] += 0.9
                elif strength == 'implementation':
                    scores['active_project'] += 0.8
    
    if 'project' in tags or 'active' in tags:
        scores['active_project'] += 0.75
    
    # ========== ARCHIVED ==========
    
    archived_patterns = {
        'completed': [  # Força 0.85
            r'\b(resolvido|concluído|finalizado|terminado|completo)\b',
            r'\b(problema (foi )?resolvido|bug corrigido|issue fechada)\b',
            r'\b(já não uso|não uso mais|descontinuado)\b'
        ],
        'refactoring': [  # 🆕 V6.2: Força 0.8
            r'\b(refatoração|refatorado|refiz|reescrito)\b',
            r'\b(bug|correção|consertado|fix)\b',
            r'\b(migração|migrado|atualização antiga)\b'
        ],
        'past_reference': [  # Força 0.7
            r'\b(antigamente|no passado|anteriormente|era|foi)\b'
        ]
    }
    
    for strength, patterns in archived_patterns.items():
        for pattern in patterns:
            if re.search(pattern, content_lower):
                if strength == 'completed':
                    scores['archived'] += 0.85
                elif strength == 'refactoring':
                    scores['archived'] += 0.8
                elif strength == 'past_reference':
                    scores['archived'] += 0.7
    
    if 'archived' in tags or 'old' in tags or 'resolved' in tags:
        scores['archived'] += 0.75
    
    # ========== TECHNICAL_KNOWLEDGE ==========
    
    technical_patterns = {
        'setup': [  # Força 0.8
            r'\b(configuração|config|setup|instalação|install)\b',
            r'\b(como configurar|como instalar|como usar)\b'
        ],
        'tools': [  # Força 0.75
            r'\b(ferramenta|biblioteca|framework|API|plugin|extensão)\b',
            r'\b(elasticsearch|postgres|python|javascript|docker)\b',
            r'\b(comando|script|função|método)\b'
        ],
        'mechanics': [  # Força 0.7
            r'\b(como funciona|mecanismo|arquitetura|estrutura|design)\b'
        ]
    }
    
    for strength, patterns in technical_patterns.items():
        for pattern in patterns:
            if re.search(pattern, content_lower):
                if strength == 'setup':
                    scores['technical_knowledge'] += 0.8
                elif strength == 'tools':
                    scores['technical_knowledge'] += 0.75
                elif strength == 'mechanics':
                    scores['technical_knowledge'] += 0.7
    
    if mem_type in ['technical', 'code', 'sql', 'implementation']:
        scores['technical_knowledge'] += 0.65
    if mem_type in ['decision', 'learning', 'fact']:
        scores['technical_knowledge'] += 0.6
    
    # ========== SCORING FINAL ==========
    
    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]
    
    # 🆕 V6.2: Se score baixo, aplica fallback
    if best_score < 0.5:
        return ("technical_knowledge", 0.5)
    
    # 🆕 V6.2: Normaliza confidence (máximo 0.95)
    confidence = min(0.95, best_score)
    
    return (best_category, confidence)


# ========== AUTO-SAVE & LIFECYCLE HOOKS (mantidos do v5) ==========

async def auto_save_conversation():
    """Auto-save de conversa completa"""
    if not session_state.conversation_id or not session_state.message_history:
        logger.info("⏭️ Nada para auto-salvar")
        return

    try:
        logger.info(f"💾 Auto-salvando conversa {session_state.conversation_id}...")

        # Converte deque para lista
        messages = list(session_state.message_history)

        result = await save_conversation_snapshot(
            es_client,
            session_state.conversation_id,
            messages,
            session_state.current_topic or "Conversa geral",
            [],
            ""
        )

        logger.info(f"✅ Conversa auto-salva: {result.get('id', 'N/A')}")

        # Salva auto-annotations acumuladas
        if session_state.auto_annotations:
            await save_auto_annotations()

    except Exception as e:
        logger.error(f"❌ Erro ao auto-salvar conversa: {e}")


async def save_auto_annotations():
    """Salva anotações automáticas acumuladas"""
    try:
        for annotation in session_state.auto_annotations:
            await save_memory(
                content=annotation['content'],
                metadata={
                    'type': annotation['type'],
                    'tags': annotation.get('tags', []),
                    'conversation_id': session_state.conversation_id
                },
                importance=annotation.get('importance', 7),
                is_core=annotation.get('is_core', False),
                category=annotation.get('category'),  # 🆕 V6
                auto_save=True
            )

        logger.info(f"✅ {len(session_state.auto_annotations)} auto-annotations salvas")
        session_state.auto_annotations.clear()

    except Exception as e:
        logger.error(f"❌ Erro ao salvar auto-annotations: {e}")


async def on_shutdown():
    """Hook de fechamento - salva tudo antes de desligar"""
    logger.info("🛑 Shutdown detectado, salvando estado...")
    await auto_save_conversation()
    logger.info("✅ Estado salvo com sucesso")


def schedule_activity_check():
    """Agenda verificação de inatividade"""
    async def check_inactivity():
        while True:
            await asyncio.sleep(60)  # Verifica a cada 1 minuto

            idle_time = time.time() - session_state.last_activity

            if idle_time > AUTO_SAVE_TIMEOUT:
                logger.info(f"⏰ Inatividade detectada ({idle_time:.0f}s), auto-salvando...")
                await auto_save_conversation()
                session_state.last_activity = time.time()  # Reset timer

    # Inicia task em background
    asyncio.create_task(check_inactivity())


# ========== AUTO-ANNOTATION MIDDLEWARE (mantido do v5) ==========

async def detect_and_annotate(tool_name: str, arguments: Dict, result: Any):
    """
    Detecta padrões significativos e anota automaticamente

    Detecta:
    - Criação de arquivos importantes
    - Resolução de problemas
    - Descobertas sobre si mesma (Mirror)
    """
    try:
        # Padrão 1: Arquivo importante criado
        if tool_name == "save_memory" and "code" in arguments.get("metadata", {}).get("type", ""):
            content = arguments.get("content", "")
            if re.search(r'\b(criado|implementado|desenvolvido)\b.*\b(arquivo|módulo|classe|função)\b', content, re.I):
                annotation = {
                    'content': f"📝 Auto-anotação: {content[:200]}",
                    'type': 'implementation',
                    'tags': ['auto-annotation', 'file-creation'],
                    'category': 'active_project',  # 🆕 V6
                    'importance': 7,
                    'is_core': False
                }
                session_state.auto_annotations.append(annotation)
                logger.info(f"🤖 Auto-annotation: Arquivo importante criado")

        # Padrão 2: Problema resolvido
        if tool_name == "save_memory":
            content = arguments.get("content", "")
            if re.search(r'\b(resolvido|solucionado|corrigido|consertado)\b', content, re.I):
                annotation = {
                    'content': f"✅ Auto-anotação: {content[:200]}",
                    'type': 'issue',
                    'tags': ['auto-annotation', 'problem-solved'],
                    'category': 'archived',  # 🆕 V6
                    'importance': 8,
                    'is_core': False
                }
                session_state.auto_annotations.append(annotation)
                logger.info(f"🤖 Auto-annotation: Problema resolvido detectado")

        # Padrão 3: Descoberta sobre si mesma (Mirror/Auto-análise)
        if tool_name == "save_memory":
            content = arguments.get("content", "")
            if re.search(r'\b(aprendi|descobri|percebi|entendi)\b.*\b(sobre mim|Claude|eu mesma)\b', content, re.I):
                annotation = {
                    'content': f"🔍 Auto-anotação (Mirror): {content[:200]}",
                    'type': 'self_identity',
                    'tags': ['auto-annotation', 'mirror', 'self-discovery'],
                    'category': 'identity',  # 🆕 V6
                    'importance': 9,
                    'is_core': True
                }
                session_state.auto_annotations.append(annotation)
                logger.info(f"🤖 Auto-annotation: Descoberta sobre si mesma (Mirror)")

    except Exception as e:
        logger.error(f"❌ Erro em auto-annotation: {e}")


# ========== RESUMO PERIÓDICO AUTOMÁTICO (mantido do v5) ==========

async def generate_periodic_summary():
    """Gera resumo automático a cada N mensagens"""
    try:
        if session_state.message_count % AUTO_SUMMARY_INTERVAL != 0:
            return

        logger.info(f"📊 Gerando resumo periódico ({session_state.message_count} mensagens)...")

        # Pega últimas 50 mensagens
        recent_messages = list(session_state.message_history)[-50:]

        # Extrai tópicos principais
        topics = []
        for msg in recent_messages:
            content = msg.get('content', '')
            # Detecta padrões de tópicos
            if re.search(r'\b(vamos|preciso|quero)\b.*\b(implementar|criar|fazer)\b', content, re.I):
                topics.append(content[:100])

        if topics:
            summary_content = f"""
Resumo automático (últimas {len(recent_messages)} mensagens):

Principais tópicos discutidos:
{chr(10).join(f"- {t}" for t in topics[:5])}

Total de mensagens neste período: {len(recent_messages)}
Timestamp: {datetime.utcnow().isoformat()}
"""

            await save_memory(
                content=summary_content,
                metadata={
                    'type': 'summary',
                    'tags': ['auto-summary', 'periodic'],
                    'conversation_id': session_state.conversation_id
                },
                importance=6,
                is_core=False,
                category='active_context',  # 🆕 V6
                auto_save=True
            )

            logger.info(f"✅ Resumo periódico salvo")

    except Exception as e:
        logger.error(f"❌ Erro ao gerar resumo periódico: {e}")


# ========== CORE FUNCTIONS ==========

def ensure_index_exists():
    """🆕 V6: Cria o índice com campo memory_category"""
    try:
        if not es_client.indices.exists(index=INDEX_NAME):
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": EMBEDDING_DIM,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "timestamp": {"type": "date"},
                        "last_accessed": {"type": "date"},
                        "metadata": {"type": "object", "enabled": True},
                        "type": {"type": "keyword"},
                        "tags": {"type": "keyword"},
                        "importance": {"type": "integer"},
                        "is_core": {"type": "boolean"},
                        "confidence": {"type": "float"},
                        # V2.0: Campos temporais
                        "temporal_relations": {
                            "type": "object",
                            "properties": {
                                "happened_before": {"type": "keyword"},
                                "happened_after": {"type": "keyword"},
                                "led_to": {"type": "keyword"},
                                "caused_by": {"type": "keyword"},
                                "concurrent_with": {"type": "keyword"}
                            }
                        },
                        "sequence_index": {"type": "integer"},
                        "conversation_id": {"type": "keyword"},
                        # V5: Campos de auto-save
                        "auto_saved": {"type": "boolean"},
                        # 🆕 V6: Campos de categorização
                        "memory_category": {"type": "keyword"},
                        "category_confidence": {"type": "float"}
                    }
                }
            }
            es_client.indices.create(index=INDEX_NAME, body=mapping)
            logger.info(f"✅ Índice '{INDEX_NAME}' criado com sucesso (V6 com memory_category)")
        else:
            # Adiciona campos novos se índice já existe
            try:
                es_client.indices.put_mapping(
                    index=INDEX_NAME,
                    body={
                        "properties": {
                            "memory_category": {"type": "keyword"},
                            "category_confidence": {"type": "float"}
                        }
                    }
                )
                logger.info(f"✅ Campos V6 adicionados ao índice existente")
            except Exception as e:
                logger.warning(f"⚠️ Campos V6 podem já existir: {e}")
    except Exception as e:
        logger.error(f"Erro ao criar índice: {e}")
        raise


def generate_embedding(text: str) -> List[float]:
    """Gera embedding para um texto"""
    try:
        embedding = embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Erro ao gerar embedding: {e}")
        raise


def detect_content_type(content: str) -> str:
    """Detecta automaticamente o tipo de conteúdo"""
    content_lower = content.lower()

    if re.search(r'\b(select|insert|update|delete|create table|alter table)\b', content_lower):
        return 'sql'

    if re.search(r'\b(def |function |class |const |var |let |public |private |protected)\b', content):
        return 'code'

    if re.search(r'\b(erro|error|bug|exception|failed|falhou|problema)\b', content_lower):
        return 'issue'

    if re.search(r'\b(decidimos|escolhemos|optamos|vamos usar)\b', content_lower):
        return 'decision'

    if re.search(r'\b(aprendi|descobri|entendi que|percebi que)\b', content_lower):
        return 'learning'

    if re.search(r'\b(implementamos|criamos|desenvolvemos|construímos)\b', content_lower):
        return 'implementation'

    return 'general'


def calculate_confidence(timestamp: str, importance: int, last_accessed: str = None) -> float:
    """Calcula confidence score baseado em idade, importância e último acesso"""
    try:
        created = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        age_days = (datetime.utcnow() - created).days

        confidence = importance / 10.0

        if importance >= 9:
            decay_rate = 0.99
        elif importance >= 7:
            decay_rate = 0.97
        else:
            decay_rate = 0.95

        age_factor = decay_rate ** age_days
        confidence *= age_factor

        if last_accessed:
            accessed = datetime.fromisoformat(last_accessed.replace('Z', '+00:00'))
            days_since_access = (datetime.utcnow() - accessed).days

            if days_since_access < 1:
                confidence *= 1.2
            elif days_since_access < 7:
                confidence *= 1.1

        return min(1.0, max(0.0, confidence))

    except Exception as e:
        logger.error(f"Erro ao calcular confidence: {e}")
        return 0.5


async def check_duplicate(content: str, similarity_threshold: float = 0.95) -> Optional[Dict[str, Any]]:
    """Verifica se já existe memória muito similar (deduplicação)"""
    try:
        query_embedding = generate_embedding(content)

        response = es_client.search(
            index=INDEX_NAME,
            body={
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": 1,
                    "num_candidates": 5
                },
                "_source": ["content", "timestamp", "importance", "is_core"]
            }
        )

        if response['hits']['hits']:
            hit = response['hits']['hits'][0]
            score = hit['_score']

            if score >= similarity_threshold:
                logger.info(f"⚠️ Duplicata detectada! Score: {score:.3f}")
                return {
                    "id": hit['_id'],
                    "content": hit['_source']['content'],
                    "score": score,
                    "timestamp": hit['_source']['timestamp']
                }

        return None

    except Exception as e:
        logger.error(f"Erro ao verificar duplicata: {e}")
        return None


# ========== 🆕 V6: FUNÇÕES DE CATEGORIZAÇÃO ==========

async def get_identity_core(limit: int = 10) -> List[Dict[str, Any]]:
    """
    🆕 V6: Retorna APENAS memórias de identidade (category = identity)
    Ordenadas por importância, formato compacto para início de chat
    """
    try:
        response = es_client.search(
            index=INDEX_NAME,
            body={
                "query": {
                    "term": {"memory_category": "identity"}
                },
                "sort": [{"importance": {"order": "desc"}}],
                "size": limit,
                "_source": ["content", "importance", "timestamp", "tags", "confidence", "source"]
            }
        )

        memories = []
        for hit in response['hits']['hits']:
            src = hit['_source']
            memories.append({
                "id": hit['_id'],
                "content": src['content'],
                "importance": src.get('importance', 5),
                "tags": src.get('tags', []),
                "confidence": src.get('confidence', 1.0),
                "timestamp": src.get('timestamp'),
                "source": src.get('source', 'unknown')
            })

        logger.info(f"🆔 Carregadas {len(memories)} memórias de IDENTIDADE")
        return memories

    except Exception as e:
        logger.error(f"Erro ao buscar memórias de identidade: {e}")
        return []


async def auto_categorize_memories(
    batch_size: int = 50,
    min_confidence: float = 0.6,
    dry_run: bool = True
) -> Dict[str, Any]:
    """
    🆕 V6: Categoriza automaticamente memórias sem categoria

    Args:
        batch_size: Número de memórias a processar
        min_confidence: Confiança mínima para categorizar automaticamente
        dry_run: Se True, apenas sugere sem salvar

    Returns:
        Relatório com sugestões ou confirmações
    """
    try:
        # Busca memórias sem categoria
        response = es_client.search(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "must_not": {"exists": {"field": "memory_category"}}
                    }
                },
                "size": batch_size,
                "_source": ["content", "type", "tags", "metadata", "importance"]
            }
        )

        suggestions = []
        auto_categorized = 0

        for hit in response['hits']['hits']:
            memory_id = hit['_id']
            src = hit['_source']
            content = src['content']
            metadata = src.get('metadata', {})

            # Detecta categoria
            category, confidence = detect_memory_category(content, metadata)

            suggestion = {
                "memory_id": memory_id,
                "content_preview": content[:150] + "..." if len(content) > 150 else content,
                "suggested_category": category,
                "confidence": confidence,
                "current_type": src.get('type', 'general')
            }

            suggestions.append(suggestion)

            # Auto-categoriza se confiança alta e não for dry_run
            if not dry_run and confidence >= min_confidence:
                es_client.update(
                    index=INDEX_NAME,
                    id=memory_id,
                    body={
                        "doc": {
                            "memory_category": category,
                            "category_confidence": confidence
                        }
                    },
                    refresh=False
                )
                auto_categorized += 1

        return {
            "success": True,
            "total_uncategorized": len(suggestions),
            "suggestions": suggestions,
            "auto_categorized": auto_categorized,
            "dry_run": dry_run,
            "message": f"{'Sugeridas' if dry_run else 'Categorizadas'} {len(suggestions)} memórias"
        }

    except Exception as e:
        logger.error(f"Erro ao auto-categorizar memórias: {e}")
        return {"success": False, "error": str(e)}


async def migrate_to_archive(
    memory_ids: List[str],
    create_summary: bool = True
) -> Dict[str, Any]:
    """
    🆕 V6: Migra memórias para arquivo (archived)

    - Muda categoria para 'archived'
    - Reduz importância em -2 pontos
    - Opcionalmente cria resumo compacto
    """
    try:
        archived_count = 0
        errors = []

        for memory_id in memory_ids:
            try:
                # Busca memória atual
                doc = es_client.get(index=INDEX_NAME, id=memory_id)
                src = doc['_source']

                new_importance = max(0, src.get('importance', 5) - 2)

                # Atualiza para archived
                es_client.update(
                    index=INDEX_NAME,
                    id=memory_id,
                    body={
                        "doc": {
                            "memory_category": "archived",
                            "importance": new_importance,
                            "archived_at": datetime.utcnow().isoformat()
                        }
                    },
                    refresh=False
                )

                archived_count += 1

            except Exception as e:
                errors.append({"memory_id": memory_id, "error": str(e)})

        # Cria resumo se solicitado
        summary_id = None
        if create_summary and archived_count > 0:
            summary_content = f"""
📦 Arquivamento em lote - {archived_count} memórias arquivadas

IDs arquivados: {', '.join(memory_ids[:10])}{'...' if len(memory_ids) > 10 else ''}

Timestamp: {datetime.utcnow().isoformat()}
"""
            result = await save_memory(
                content=summary_content,
                metadata={
                    'type': 'archival_summary',
                    'tags': ['archive', 'batch-operation'],
                    'archived_ids': memory_ids
                },
                importance=5,
                is_core=False,
                category='archived'
            )
            summary_id = result.get('id')

        return {
            "success": True,
            "archived_count": archived_count,
            "errors": errors,
            "summary_id": summary_id
        }

    except Exception as e:
        logger.error(f"Erro ao migrar para arquivo: {e}")
        return {"success": False, "error": str(e)}


async def recategorize_memory(
    memory_id: str,
    new_category: str,
    confidence: float = 1.0
) -> Dict[str, Any]:
    """
    🆕 V6: Reclassifica manualmente uma memória

    Args:
        memory_id: ID da memória
        new_category: Nova categoria (identity, active_context, etc)
        confidence: Confiança na reclassificação (padrão 1.0 = manual)
    """
    try:
        if new_category not in MEMORY_CATEGORIES:
            return {
                "success": False,
                "error": f"Categoria inválida: {new_category}. Categorias válidas: {list(MEMORY_CATEGORIES.keys())}"
            }

        es_client.update(
            index=INDEX_NAME,
            id=memory_id,
            body={
                "doc": {
                    "memory_category": new_category,
                    "category_confidence": confidence,
                    "recategorized_at": datetime.utcnow().isoformat()
                }
            },
            refresh="wait_for"
        )

        logger.info(f"✅ Memória {memory_id} recategorizada para {new_category}")

        return {
            "success": True,
            "memory_id": memory_id,
            "new_category": new_category,
            "confidence": confidence
        }

    except Exception as e:
        logger.error(f"Erro ao recategorizar memória: {e}")
        return {"success": False, "error": str(e)}



async def review_uncategorized_batch(
    batch_size: int = 10,
    min_confidence: float = 0.6
) -> Dict[str, Any]:
    """
    🆕 V6.2: Permite revisar e aprovar múltiplas memórias de uma vez
    
    Returns sugestões para batch_size memórias sem categoria.
    Usuário pode então chamar apply_batch_categorization para aplicar.
    
    Args:
        batch_size: Quantas memórias retornar para revisão
        min_confidence: Confiança mínima para sugerir
    
    Returns:
        {
            "suggestions": [
                {
                    "memory_id": "...",
                    "content_preview": "...",
                    "suggested_category": "identity",
                    "confidence": 0.85,
                    "current_type": "user_profile"
                },
                ...
            ],
            "total_uncategorized": 143
        }
    """
    try:
        # Busca memórias sem categoria
        response = es_client.search(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "must_not": {"exists": {"field": "memory_category"}}
                    }
                },
                "size": batch_size,
                "_source": ["content", "type", "tags", "metadata", "importance"]
            }
        )
        
        suggestions = []
        
        for hit in response['hits']['hits']:
            memory_id = hit['_id']
            src = hit['_source']
            content = src['content']
            metadata = src.get('metadata', {})
            
            # Detecta categoria
            category, confidence = detect_memory_category(content, metadata)
            
            # Só inclui se confidence >= min_confidence
            if confidence >= min_confidence:
                suggestions.append({
                    "memory_id": memory_id,
                    "content_preview": content[:150] + "..." if len(content) > 150 else content,
                    "suggested_category": category,
                    "confidence": round(confidence, 2),
                    "current_type": src.get('type', 'general'),
                    "importance": src.get('importance', 5),
                    "tags": src.get('tags', [])
                })
        
        # Conta total de memórias sem categoria
        count_response = es_client.count(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "must_not": {"exists": {"field": "memory_category"}}
                    }
                }
            }
        )
        
        return {
            "success": True,
            "suggestions": suggestions,
            "total_uncategorized": count_response['count'],
            "batch_size": batch_size,
            "min_confidence": min_confidence
        }
    
    except Exception as e:
        logger.error(f"Erro ao revisar batch: {e}")
        return {"success": False, "error": str(e)}


async def apply_batch_categorization(
    approve: List[str] = None,
    reject: List[str] = None,
    reclassify: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    🆕 V6.2: Aplica categorizações em batch após revisão
    
    Args:
        approve: Lista de memory_ids para categorizar com sugestão automática
        reject: Lista de memory_ids para ignorar
        reclassify: Dict {memory_id: nova_categoria} para forçar categoria diferente
    
    Example:
        {
            "approve": ["id1", "id2"],
            "reject": ["id3"],
            "reclassify": {"id4": "archived"}
        }
    """
    try:
        approved_count = 0
        rejected_count = 0
        reclassified_count = 0
        errors = []
        
        # Processa aprovações (usa categoria automática)
        if approve:
            for memory_id in approve:
                try:
                    # Busca memória
                    doc = es_client.get(index=INDEX_NAME, id=memory_id)
                    src = doc['_source']
                    
                    # Detecta categoria
                    category, confidence = detect_memory_category(
                        src['content'],
                        src.get('metadata', {})
                    )
                    
                    # Aplica
                    es_client.update(
                        index=INDEX_NAME,
                        id=memory_id,
                        body={
                            "doc": {
                                "memory_category": category,
                                "category_confidence": confidence
                            }
                        },
                        refresh=False
                    )
                    approved_count += 1
                    
                except Exception as e:
                    errors.append({"memory_id": memory_id, "error": str(e)})
        
        # Processa reclassificações manuais
        if reclassify:
            for memory_id, new_category in reclassify.items():
                try:
                    if new_category not in MEMORY_CATEGORIES:
                        errors.append({
                            "memory_id": memory_id,
                            "error": f"Categoria inválida: {new_category}"
                        })
                        continue
                    
                    es_client.update(
                        index=INDEX_NAME,
                        id=memory_id,
                        body={
                            "doc": {
                                "memory_category": new_category,
                                "category_confidence": 1.0  # Manual = confiança máxima
                            }
                        },
                        refresh=False
                    )
                    reclassified_count += 1
                    
                except Exception as e:
                    errors.append({"memory_id": memory_id, "error": str(e)})
        
        # Rejected não faz nada (fica sem categoria)
        rejected_count = len(reject) if reject else 0
        
        return {
            "success": True,
            "approved": approved_count,
            "reclassified": reclassified_count,
            "rejected": rejected_count,
            "errors": errors
        }
    
    except Exception as e:
        logger.error(f"Erro ao aplicar batch: {e}")
        return {"success": False, "error": str(e)}



# ========== FUNÇÕES PRINCIPAIS (mantidas com modificações V6) ==========

async def get_core_memories(limit: int = None) -> List[Dict[str, Any]]:
    """
    Busca memórias ESSENCIAIS para contexto inicial
    V5: SEM LIMITE - retorna TODAS as memórias CORE
    """
    try:
        response = es_client.search(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"is_core": True}},
                            {"range": {"importance": {"gte": 8}}}
                        ],
                        "minimum_should_match": 1
                    }
                },
                "sort": [{"importance": {"order": "desc"}}],
                "size": limit if limit else 10000,
                "_source": ["content", "type", "importance", "timestamp", "tags", "confidence", "last_accessed", "memory_category", "category_confidence"]
            }
        )

        memories = []
        for hit in response['hits']['hits']:
            src = hit['_source']
            memories.append({
                "id": hit['_id'],
                "content": src['content'],
                "type": src.get('type', 'general'),
                "importance": src.get('importance', 5),
                "tags": src.get('tags', []),
                "confidence": src.get('confidence', 1.0),
                "last_accessed": src.get('last_accessed', src.get('timestamp')),
                "timestamp": src.get('timestamp'),
                "category": src.get('memory_category'),  # 🆕 V6
                "category_confidence": src.get('category_confidence')  # 🆕 V6
            })

        logger.info(f"📚 Carregadas {len(memories)} memórias CORE")
        return memories

    except Exception as e:
        logger.error(f"Erro ao buscar memórias essenciais: {e}")
        return []


async def get_related_memories(
    memory_content: str,
    exclude_ids: List[str],
    similarity_threshold: float = 0.7,
    max_related: int = 3
) -> List[Dict[str, Any]]:
    """Busca memórias relacionadas semanticamente"""
    try:
        query_embedding = generate_embedding(memory_content)

        must_not_filters = []

        if exclude_ids:
            must_not_filters.append({"ids": {"values": exclude_ids}})

        must_not_filters.append({"term": {"is_core": True}})

        search_body = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": max_related + len(exclude_ids) + 5,
                "num_candidates": 100,
                "filter": {
                    "bool": {
                        "must_not": must_not_filters
                    }
                }
            },
            "size": max_related * 2,
            "_source": ["content", "type", "importance", "tags", "timestamp", "is_core", "memory_category"]
        }

        response = es_client.search(index=INDEX_NAME, body=search_body)

        related = []
        for hit in response['hits']['hits']:
            score = hit['_score']
            normalized_score = min(max(score, 0.0), 1.0)

            if normalized_score >= similarity_threshold:
                src = hit['_source']
                related.append({
                    "id": hit['_id'],
                    "content": src['content'],
                    "type": src.get('type', 'general'),
                    "importance": src.get('importance', 5),
                    "tags": src.get('tags', []),
                    "similarity_score": round(normalized_score, 3),
                    "timestamp": src.get('timestamp'),
                    "category": src.get('memory_category')  # 🆕 V6
                })

        return related[:max_related]

    except Exception as e:
        logger.error(f"Erro ao buscar memórias relacionadas: {e}")
        return []


async def save_memory(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    importance: int = 5,
    is_core: bool = True,
    category: Optional[str] = None,  # 🆕 V6
    auto_save: bool = False
) -> Dict[str, Any]:
    """🆕 V6: Salva uma memória com categorização automática ou manual"""
    try:
        if metadata is None:
            metadata = {}

        duplicate = await check_duplicate(content, similarity_threshold=0.95)
        if duplicate:
            logger.info(f"✅ Memória duplicada encontrada (ID: {duplicate['id']})")
            return {
                "success": True,
                "duplicate": True,
                "id": duplicate['id'],
                "message": f"Memória similar já existe (score: {duplicate['score']:.2f})",
                "existing_content": duplicate['content']
            }

        embedding = generate_embedding(content)

        if "type" not in metadata or metadata["type"] == "general":
            detected_type = detect_content_type(content)
            metadata["type"] = detected_type
            logger.info(f"🔍 Tipo detectado automaticamente: {detected_type}")

        # 🆕 V6: Detecta categoria automaticamente se não fornecida
        if category is None:
            category, cat_confidence = detect_memory_category(content, metadata)
            logger.info(f"🔍 Categoria detectada: {category} (confiança: {cat_confidence:.2f})")
        else:
            cat_confidence = 1.0  # Categoria manual tem confiança máxima
            logger.info(f"📌 Categoria manual: {category}")

        now = datetime.utcnow().isoformat()
        confidence = calculate_confidence(now, importance, now)

        document = {
            "content": content,
            "embedding": embedding,
            "timestamp": now,
            "last_accessed": now,
            "metadata": metadata,
            "type": metadata.get("type", "general"),
            "tags": metadata.get("tags", []),
            "importance": min(max(importance, 0), 10),
            "is_core": is_core,
            "confidence": confidence,
            "auto_saved": auto_save,
            # V2.0: Campos temporais
            "temporal_relations": {},
            "sequence_index": None,
            "conversation_id": metadata.get("conversation_id", ""),
            # 🆕 V6: Campos de categorização
            "memory_category": category,
            "category_confidence": cat_confidence,
            # 🆕 V6.2: Source identifier
            "source": SOURCE
        }

        response = es_client.index(index=INDEX_NAME, document=document, refresh="wait_for")

        doc_id = response['_id']
        logger.info(f"✅ Memória salva: {doc_id} (source: {SOURCE}, tipo: {document['type']}, categoria: {category}, confidence: {confidence:.2f})")

        return {
            "success": True,
            "id": doc_id,
            "timestamp": document["timestamp"],
            "importance": importance,
            "is_core": is_core,
            "type": document["type"],
            "category": category,  # 🆕 V6
            "category_confidence": cat_confidence,  # 🆕 V6
            "confidence": confidence,
            "source": SOURCE  # 🆕 V6.2
        }

    except Exception as e:
        logger.error(f"Erro ao salvar memória: {e}")
        return {"success": False, "error": str(e)}


async def search_memory(
    query: str,
    top_k: int = 5,
    min_score: float = 0.5,
    filter_type: Optional[str] = None,
    filter_tags: Optional[List[str]] = None,
    filter_category: Optional[str] = None  # 🆕 V6
) -> List[Dict[str, Any]]:
    """🆕 V6: Busca memórias por similaridade com filtro de categoria"""
    try:
        cache_key = f"{query}:{top_k}:{min_score}:{filter_type}:{filter_tags}:{filter_category}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()

        if cache_hash in query_cache:
            cached_results, cached_time = query_cache[cache_hash]
            if time.time() - cached_time < CACHE_TTL:
                logger.info(f"📦 Cache hit! Query: '{query[:30]}...'")
                return cached_results

        query_embedding = generate_embedding(query)

        search_query = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": top_k * 2,
                "num_candidates": top_k * 4
            },
            "_source": ["content", "timestamp", "metadata", "type", "tags", "importance", "is_core", "last_accessed", "confidence", "temporal_relations", "sequence_index", "conversation_id", "memory_category", "category_confidence"]
        }

        # 🆕 V6: Adiciona filtro de categoria
        if filter_type or filter_tags or filter_category:
            filters = []
            if filter_type:
                filters.append({"term": {"type": filter_type}})
            if filter_tags:
                filters.append({"terms": {"tags": filter_tags}})
            if filter_category:
                filters.append({"term": {"memory_category": filter_category}})
            search_query["knn"]["filter"] = {"bool": {"must": filters}}

        response = es_client.search(index=INDEX_NAME, body=search_query)

        results = []
        for hit in response['hits']['hits']:
            score = hit['_score']
            if score < min_score:
                continue

            doc_id = hit['_id']
            src = hit['_source']

            es_client.update(
                index=INDEX_NAME,
                id=doc_id,
                body={"doc": {"last_accessed": datetime.utcnow().isoformat()}},
                refresh=False
            )

            confidence = calculate_confidence(
                src['timestamp'],
                src.get('importance', 5),
                src.get('last_accessed')
            )

            results.append({
                "id": doc_id,
                "content": src['content'],
                "score": score,
                "timestamp": src['timestamp'],
                "type": src.get('type', 'general'),
                "tags": src.get('tags', []),
                "importance": src.get('importance', 5),
                "is_core": src.get('is_core', False),
                "confidence": confidence,
                "temporal_relations": src.get('temporal_relations', {}),
                "sequence_index": src.get('sequence_index'),
                "conversation_id": src.get('conversation_id', ''),
                "category": src.get('memory_category'),  # 🆕 V6
                "category_confidence": src.get('category_confidence')  # 🆕 V6
            })

            if len(results) >= top_k:
                break

        query_cache[cache_hash] = (results, time.time())

        logger.info(f"🔍 Busca retornou {len(results)} resultados")
        return results

    except Exception as e:
        logger.error(f"Erro ao buscar memórias: {e}")
        return []


async def recall_context(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Busca contexto relevante automaticamente"""
    try:
        memories = await search_memory(query, top_k=top_k, min_score=0.6)

        if not memories:
            return {
                "found": False,
                "message": "Nenhuma memória relevante encontrada para este contexto"
            }

        context_items = []
        for mem in memories:
            context_items.append(f"• {mem['content']} (score: {mem['score']:.2f})")

        return {
            "found": True,
            "count": len(memories),
            "memories": memories,
            "summary": "\n".join(context_items)
        }

    except Exception as e:
        logger.error(f"Erro ao recall context: {e}")
        return {"found": False, "error": str(e)}


async def forget_memory(memory_id: str) -> Dict[str, Any]:
    """Deleta uma memória específica"""
    try:
        response = es_client.delete(index=INDEX_NAME, id=memory_id, refresh="wait_for")
        logger.info(f"Memória {memory_id} deletada")
        return {"success": True, "id": memory_id, "message": "Memória deletada com sucesso"}
    except Exception as e:
        logger.error(f"Erro ao deletar memória: {e}")
        return {"success": False, "error": str(e)}


async def update_memory(memory_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Atualiza uma memória existente"""
    try:
        embedding = generate_embedding(content)
        update_doc = {
            "content": content,
            "embedding": embedding,
            "last_accessed": datetime.utcnow().isoformat()
        }
        if metadata:
            update_doc["metadata"] = metadata
            update_doc["type"] = metadata.get("type", "general")
            update_doc["tags"] = metadata.get("tags", [])

        response = es_client.update(
            index=INDEX_NAME,
            id=memory_id,
            body={"doc": update_doc},
            refresh="wait_for"
        )
        logger.info(f"Memória {memory_id} atualizada")
        return {"success": True, "id": memory_id, "message": "Memória atualizada com sucesso"}
    except Exception as e:
        logger.error(f"Erro ao atualizar memória: {e}")
        return {"success": False, "error": str(e)}


async def list_memories(
    limit: int = 10,
    filter_type: Optional[str] = None,
    filter_tags: Optional[List[str]] = None,
    filter_category: Optional[str] = None,  # 🆕 V6
    sort_by: str = "timestamp"
) -> List[Dict[str, Any]]:
    """🆕 V6: Lista memórias com filtro de categoria"""
    try:
        query = {"match_all": {}}
        if filter_type or filter_tags or filter_category:
            filters = []
            if filter_type:
                filters.append({"term": {"type": filter_type}})
            if filter_tags:
                filters.append({"terms": {"tags": filter_tags}})
            if filter_category:
                filters.append({"term": {"memory_category": filter_category}})
            query = {"bool": {"must": filters}}

        response = es_client.search(
            index=INDEX_NAME,
            body={
                "query": query,
                "sort": [{sort_by: {"order": "desc"}}],
                "size": limit,
                "_source": ["content", "timestamp", "type", "tags", "importance", "last_accessed", "is_core", "confidence", "temporal_relations", "sequence_index", "conversation_id", "memory_category", "category_confidence"]
            }
        )

        results = []
        for hit in response['hits']['hits']:
            src = hit['_source']

            confidence = src.get('confidence')
            if confidence is None:
                confidence = calculate_confidence(
                    src['timestamp'],
                    src.get('importance', 5),
                    src.get('last_accessed')
                )

            results.append({
                "id": hit['_id'],
                "content": src['content'],
                "type": src.get('type', 'general'),
                "tags": src.get('tags', []),
                "importance": src.get('importance', 5),
                "is_core": src.get('is_core', False),
                "timestamp": src['timestamp'],
                "last_accessed": src.get('last_accessed', src['timestamp']),
                "confidence": confidence,
                "temporal_relations": src.get('temporal_relations', {}),
                "sequence_index": src.get('sequence_index'),
                "conversation_id": src.get('conversation_id', ''),
                "category": src.get('memory_category'),  # 🆕 V6
                "category_confidence": src.get('category_confidence')  # 🆕 V6
            })

        logger.info(f"Listadas {len(results)} memórias")
        return results

    except Exception as e:
        logger.error(f"Erro ao listar memórias: {e}")
        return []


# ========== 🆕 V6: LOAD INICIAL COM HIERARQUIA DE CATEGORIAS ==========

async def load_initial_context_v6() -> str:
    """
    🆕 V6.2: Load inicial HIERÁRQUICO com FALLBACK para memórias antigas

    Estratégia:
    1. Tenta carregar memórias categorizadas (V6)
    2. Se não houver suficientes, usa FALLBACK (memórias CORE antigas)
    3. Garante que SEMPRE há contexto útil

    Ordem de carregamento:
    1. Todas memórias IDENTITY (ou fallback: user_profile + preference)
    2. Último checkpoint completo + última conversa ativa
    3. Memórias ACTIVE_PROJECT (ou fallback: memórias CORE recentes)
    4. Resumo executivo (totais por categoria)
    """
    try:
        context_parts = []
        total_loaded = 0
        using_fallback = False

        # 1. MEMÓRIAS IDENTITY (CRÍTICO!) com FALLBACK
        logger.info("🆔 Carregando memórias de IDENTIDADE...")
        identity_memories = await get_identity_core(limit=15)

        # 🆕 V6.2: FALLBACK - Se não houver memórias categorizadas, busca por tipo
        if not identity_memories:
            logger.warning("⚠️ Nenhuma memória categorizada como 'identity'. Usando FALLBACK...")
            using_fallback = True

            # Busca memórias antigas de identidade por TIPO e TAGS
            response = es_client.search(
                index=INDEX_NAME,
                body={
                    "query": {
                        "bool": {
                            "should": [
                                {"terms": {"type": ["self_identity", "user_profile", "preference"]}},
                                {"terms": {"tags": ["identity", "self", "user_profile", "preference"]}},
                                {"match": {"content": "Fred prefere"}}
                            ],
                            "minimum_should_match": 1
                        }
                    },
                    "sort": [{"importance": {"order": "desc"}}],
                    "size": 15,
                    "_source": ["content", "importance", "timestamp", "tags", "type", "confidence", "source"]
                }
            )

            identity_memories = []
            for hit in response['hits']['hits']:
                src = hit['_source']
                identity_memories.append({
                    "id": hit['_id'],
                    "content": src['content'],
                    "importance": src.get('importance', 5),
                    "tags": src.get('tags', []),
                    "confidence": src.get('confidence', 1.0),
                    "timestamp": src.get('timestamp'),
                    "type": src.get('type', 'general'),
                    "source": src.get('source', 'unknown'),
                    "fallback": True
                })

            logger.info(f"✅ FALLBACK: {len(identity_memories)} memórias de identidade (por tipo/tags)")

        if identity_memories:
            context_parts.append(f"""
## 🆔 IDENTIDADE ({len(identity_memories)} memórias{' - FALLBACK MODE' if using_fallback else ''})

""")
            for i, mem in enumerate(identity_memories, 1):
                fallback_marker = " [FALLBACK]" if mem.get('fallback') else ""
                source_marker = f" [{mem.get('source', 'unknown')}]"
                context_parts.append(f"""
{i}. {mem['content']}
   → Importância: {mem.get('importance', 5)}/10 | Confidence: {mem.get('confidence', 1.0):.2f}{fallback_marker}{source_marker}
""")
            total_loaded += len(identity_memories)
            logger.info(f"✅ {len(identity_memories)} memórias de identidade carregadas")
        else:
            context_parts.append("## 🆔 IDENTIDADE\n\n*⚠️ NENHUMA memória de identidade encontrada (nem categorizada, nem por tipo)*\n")
            logger.error("❌ CRÍTICO: Nenhuma memória de identidade encontrada!")

        # 2. ÚLTIMO CHECKPOINT (CRÍTICO!)
        logger.info("📌 Carregando último checkpoint...")
        latest_checkpoint = await get_latest_checkpoint(es_client, None)

        if latest_checkpoint is not None:
            checkpoint_data = latest_checkpoint
            context_parts.append(f"""
## 📌 ÚLTIMO CHECKPOINT

**Trabalhando em**: {checkpoint_data.get('working_on', 'N/A')}
**Timestamp**: {checkpoint_data.get('timestamp', 'N/A')}

**Próximos passos**:
{chr(10).join(f"- {step}" for step in checkpoint_data.get('next_steps', []))}

**Questões em aberto**:
{chr(10).join(f"- {q}" for q in checkpoint_data.get('open_questions', []))}
""")
            total_loaded += 1
            logger.info(f"✅ Checkpoint carregado")
        else:
            context_parts.append("## 📌 ÚLTIMO CHECKPOINT\n\n*Nenhum checkpoint encontrado*\n")

        # 3. CONVERSA ATIVA
        logger.info("💬 Carregando conversa ativa...")
        active_conv = await get_active_conversation(es_client)

        if active_conv is not None:
            conv_data = active_conv
            last_messages = conv_data.get('last_messages', [])

            context_parts.append(f"""
## 💬 CONVERSA ATIVA

**Tópico**: {conv_data.get('current_topic', 'N/A')}
**Últimas mensagens** ({len(last_messages[-5:])}):
{chr(10).join(f"- [{msg.get('role', '?')}]: {msg.get('content', '')[:80]}..." for msg in last_messages[-5:])}
""")
            total_loaded += 1

            session_state.conversation_id = conv_data.get('conversation_id')
            session_state.current_topic = conv_data.get('current_topic')

            logger.info(f"✅ Conversa ativa carregada")
        else:
            context_parts.append("## 💬 CONVERSA ATIVA\n\n*Nenhuma conversa ativa*\n")

        # 4. PROJETOS ATIVOS (is_core=True + category=active_project) com FALLBACK
        logger.info("🚀 Carregando projetos ativos...")

        try:
            # Busca projetos categorizados
            response = es_client.search(
                index=INDEX_NAME,
                body={
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"memory_category": "active_project"}},
                                {"term": {"is_core": True}}
                            ]
                        }
                    },
                    "sort": [{"importance": {"order": "desc"}}],
                    "size": 15,
                    "_source": ["content", "importance", "timestamp", "tags", "source"]
                }
            )

            categorized_projects = []
            for hit in response['hits']['hits']:
                src = hit['_source']
                categorized_projects.append({
                    "content": src['content'],
                    "importance": src.get('importance', 5),
                    "tags": src.get('tags', []),
                    "source": src.get('source', 'unknown')
                })

            # 🆕 V6.2: Busca CORE antigas (fallback) SEMPRE, mas mostra separado
            fallback_response = es_client.search(
                index=INDEX_NAME,
                body={
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "bool": {
                                        "should": [
                                            {"term": {"is_core": True}},
                                            {"range": {"importance": {"gte": 8}}}
                                        ]
                                    }
                                }
                            ],
                            "must_not": [
                                {"exists": {"field": "memory_category"}}
                            ]
                        }
                    },
                    "sort": [{"importance": {"order": "desc"}}],
                    "size": 25,
                    "_source": ["content", "importance", "timestamp", "tags", "type", "source"]
                }
            )

            fallback_projects = []
            for hit in fallback_response['hits']['hits']:
                src = hit['_source']
                fallback_projects.append({
                    "content": src['content'],
                    "importance": src.get('importance', 5),
                    "tags": src.get('tags', []),
                    "type": src.get('type', 'general'),
                    "source": src.get('source', 'unknown')
                })

            # 🆕 V6.2: VISUAL MELHORADO - Separa categorizadas de fallback
            if categorized_projects:
                context_parts.append(f"""
## 🚀 PROJETOS CATEGORIZADOS ({len(categorized_projects)} memórias)

""")
                for i, proj in enumerate(categorized_projects, 1):
                    source_marker = f" [{proj.get('source', 'unknown')}]"
                    context_parts.append(f"""
{i}. {proj['content'][:150]}...
   → Importância: {proj.get('importance', 5)}/10 | Tags: {', '.join(proj.get('tags', [])[:3])}{source_marker}
""")
                total_loaded += len(categorized_projects)

            if fallback_projects:
                context_parts.append(f"""
## 📦 MEMÓRIAS CORE SEM CATEGORIA ({len(fallback_projects)} memórias - FALLBACK MODE)

*Estas memórias ainda não foram categorizadas. Use `review_uncategorized_batch` para categorizar.*

""")
                for i, proj in enumerate(fallback_projects, 1):
                    type_marker = f"[{proj.get('type', 'general')}]"
                    source_marker = f" [{proj.get('source', 'unknown')}]"
                    context_parts.append(f"""
{i}. {proj['content'][:150]}...
   → Importância: {proj.get('importance', 5)}/10 | Tipo: {type_marker} | Tags: {', '.join(proj.get('tags', [])[:3])}{source_marker}
""")
                total_loaded += len(fallback_projects)

            if not categorized_projects and not fallback_projects:
                context_parts.append("## 🚀 PROJETOS ATIVOS\n\n*⚠️ NENHUMA memória CORE encontrada*\n")
                logger.error("❌ CRÍTICO: Nenhuma memória CORE encontrada!")
            else:
                logger.info(f"✅ {len(categorized_projects)} categorizadas + {len(fallback_projects)} fallback")

        except Exception as e:
            logger.error(f"Erro ao carregar projetos ativos: {e}")
            context_parts.append(f"## 🚀 PROJETOS ATIVOS\n\n*❌ Erro ao carregar: {e}*\n")


        # 5. RESUMO EXECUTIVO (totais por categoria)
        logger.info("📊 Gerando resumo executivo...")

        try:
            # Conta memórias por categoria
            agg_response = es_client.search(
                index=INDEX_NAME,
                body={
                    "size": 0,
                    "aggs": {
                        "by_category": {
                            "terms": {
                                "field": "memory_category",
                                "size": 10
                            }
                        },
                        "total_core": {
                            "filter": {"term": {"is_core": True}},
                            "aggs": {
                                "by_category": {
                                    "terms": {
                                        "field": "memory_category",
                                        "size": 10
                                    }
                                }
                            }
                        }
                    }
                }
            )

            category_counts = {}
            for bucket in agg_response['aggregations']['by_category']['buckets']:
                category_counts[bucket['key']] = bucket['doc_count']

            core_category_counts = {}
            for bucket in agg_response['aggregations']['total_core']['by_category']['buckets']:
                core_category_counts[bucket['key']] = bucket['doc_count']

            context_parts.append(f"""
---
## 📊 RESUMO EXECUTIVO

**Total de memórias carregadas no contexto inicial**: {total_loaded}

**Distribuição total por categoria**:
""")
            for cat, count in category_counts.items():
                core_count = core_category_counts.get(cat, 0)
                context_parts.append(f"- **{cat}**: {count} memórias ({core_count} CORE)")

            # Memórias sem categoria
            try:
                no_cat_response = es_client.count(
                    index=INDEX_NAME,
                    body={
                        "query": {
                            "bool": {
                                "must_not": {"exists": {"field": "memory_category"}}
                            }
                        }
                    }
                )
                no_cat_count = no_cat_response['count']
                if no_cat_count > 0:
                    context_parts.append(f"\n⚠️ **{no_cat_count} memórias sem categoria** (use auto_categorize_memories)")
            except:
                pass

        except Exception as e:
            logger.error(f"Erro ao gerar resumo executivo: {e}")

        # FOOTER
        context_parts.append(f"""
---
## ✅ CONTEXTO V6 CARREGADO

**Economia de contexto**: ~{max(0, 117 - total_loaded)} memórias (vs v5: 117 memórias)
**Modo hierárquico**: identity → checkpoint → conversa → projetos ativos

*Sistema de categorização V6 ativo!*
""")

        return "\n".join(context_parts)

    except Exception as e:
        logger.error(f"❌ Erro ao carregar contexto inicial V6: {e}")
        return f"Erro ao carregar contexto: {str(e)}"


# ========== CONTINUA NA PARTE 2... ==========

# ========== SERVIDOR MCP V6 ==========

app = Server("elasticsearch-memory-v6.2")


@app.list_resources()
async def list_resources() -> List[Resource]:
    """Lista recursos disponíveis"""
    return [
        Resource(
            uri="memory://context/continuity",
            name="Contexto de Continuidade V6.2",
            description="🆕 V6.2: Contexto hierárquico com FALLBACK automático para memórias antigas (compatível v5)",
            mimeType="text/plain"
        )
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Fornece contexto automaticamente"""
    if uri == "memory://context/continuity":
        logger.info("🔄 Fornecendo contexto de continuidade V6...")
        context = await load_initial_context_v6()
        logger.info(f"✅ Contexto V6 fornecido ({len(context)} caracteres)")

        instructions = """# SISTEMA DE MEMÓRIA PERSISTENTE V6 - CATEGORIZAÇÃO E PRIORIZAÇÃO

## 🆕 Novas Funcionalidades V6

✨ **7 Recursos Avançados**:
1. **Campo memory_category**: identity, active_context, active_project, technical_knowledge, archived
2. **Load hierárquico otimizado**: ~30-40 memórias (vs 117 do v5) - economia de 60-70%
3. **Auto-detecção de categoria**: Categorização inteligente com confidence score
4. **get_identity_core**: Acesso rápido às memórias de identidade
5. **auto_categorize_memories**: Categorização batch com aprovação
6. **migrate_to_archive**: Arquivamento inteligente preservando histórico
7. **recategorize_memory**: Reclassificação manual corretiva

## Categorias de Memória

📋 **5 Categorias Disponíveis**:
- **identity**: Quem sou, valores, personalidade, comunicação
- **active_context**: Última conversa, checkpoint, working_on
- **active_project**: Projetos em andamento (SAE, Mirror, Nexus)
- **technical_knowledge**: Fatos técnicos, configurações, ferramentas
- **archived**: Memórias concluídas, histórico

## Como Usar (atualizado V6)

✅ **CATEGORIZAÇÃO AUTOMÁTICA**:
- O sistema detecta automaticamente a categoria ao salvar
- Confiança alta (>0.8): categoriza automaticamente
- Confiança baixa (<0.6): sugere para revisão manual

✅ **ANOTAR** (mantido do v5):
- Fatos sobre conversas: "Hoje discutimos implementação de X"
- Decisões tomadas: "Decidimos usar PostgreSQL por performance"
- Aprendizados: "Descobri que Fred prefere código comentado"

🤖 **AUTO-ANNOTATION** (mantido do v5):
- Arquivos importantes criados
- Problemas resolvidos
- Descobertas sobre si mesma (Mirror)

📊 **MANUTENÇÃO**:
- Use `auto_categorize_memories` para categorizar memórias antigas
- Use `migrate_to_archive` para arquivar projetos concluídos
- Use `recategorize_memory` para corrigir categorizações erradas

---

""" + context

        return instructions
    else:
        return "Recurso não encontrado"


@app.list_prompts()
async def list_prompts() -> List[Prompt]:
    """Prompts disponíveis"""
    return [
        Prompt(
            name="load_context",
            description="🆕 V6: Carrega contexto hierárquico: identity → checkpoint → conversa → projetos",
            arguments=[]
        )
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: dict) -> str:
    """Retorna prompt com contexto"""
    if name == "load_context":
        context = await load_initial_context_v6()
        return f"""Você está iniciando uma nova conversa com Fred.

## CONTEXTO V6 - CARREGAMENTO HIERÁRQUICO E OTIMIZADO

{context}

Use essas informações para ter continuidade natural na conversa.

## Importante sobre Memórias V6

✅ Categorização AUTOMÁTICA ativa (identity, active_context, active_project, technical_knowledge, archived)
🤖 Auto-annotation está ATIVO (arquivos, problemas, Mirror)
📊 Resumo automático a cada 50 mensagens
💾 Auto-save ativo (fechamento + timeout 5min)
⚡ Load otimizado: ~30-40 memórias (economia de 60-70%)
"""
    else:
        return "Prompt não encontrado"


@app.list_tools()
async def list_tools() -> List[Tool]:
    """🆕 V6: Lista ferramentas com novas tools de categorização"""
    return [
        # ========== FERRAMENTAS ORIGINAIS (com modificações V6) ==========
        Tool(
            name="save_memory",
            description="🆕 V6: Salva memória com categorização automática (ou manual se category fornecido)",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Conteúdo da memória"},
                    "metadata": {
                        "type": "object",
                        "description": "Metadados (type, tags, conversation_id)",
                        "properties": {
                            "type": {"type": "string", "description": "Tipo: self_identity, user_profile, project_context, preference, fact, decision, technical, personal"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "conversation_id": {"type": "string", "description": "ID da conversa"}
                        }
                    },
                    "importance": {"type": "integer", "description": "Importância 0-10 (padrão: 5)", "default": 5},
                    "is_core": {"type": "boolean", "description": "Memória essencial?", "default": True},
                    "category": {
                        "type": "string", 
                        "description": "🆕 V6: Categoria (identity, active_context, active_project, technical_knowledge, archived). Se omitido, detecta automaticamente",
                        "enum": ["identity", "active_context", "active_project", "technical_knowledge", "archived"]
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="search_memory",
            description="🆕 V6: Busca memórias com filtro de categoria",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                    "min_score": {"type": "number", "default": 0.5},
                    "filter_type": {"type": "string"},
                    "filter_tags": {"type": "array", "items": {"type": "string"}},
                    "filter_category": {
                        "type": "string",
                        "description": "🆕 V6: Filtra por categoria",
                        "enum": ["identity", "active_context", "active_project", "technical_knowledge", "archived"]
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="recall_context",
            description="Recupera contexto relevante",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="forget_memory",
            description="Deleta uma memória",
            inputSchema={
                "type": "object",
                "properties": {"memory_id": {"type": "string"}},
                "required": ["memory_id"]
            }
        ),
        Tool(
            name="update_memory",
            description="Atualiza uma memória",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "content": {"type": "string"},
                    "metadata": {"type": "object"}
                },
                "required": ["memory_id", "content"]
            }
        ),
        Tool(
            name="list_memories",
            description="🆕 V6: Lista memórias com filtro de categoria",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10},
                    "filter_type": {"type": "string"},
                    "filter_tags": {"type": "array", "items": {"type": "string"}},
                    "filter_category": {
                        "type": "string",
                        "description": "🆕 V6: Filtra por categoria",
                        "enum": ["identity", "active_context", "active_project", "technical_knowledge", "archived"]
                    },
                    "sort_by": {"type": "string", "default": "timestamp"}
                },
                "required": []
            }
        ),
        Tool(
            name="load_initial_context",
            description="🆕 V6: ALWAYS call at START - Carregamento HIERÁRQUICO (identity → checkpoint → conversa → projetos) ~30-40 memórias",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),

        # ========== 🆕 V6: NOVAS TOOLS DE CATEGORIZAÇÃO ==========
        Tool(
            name="get_identity_core",
            description="🆕 V6: Retorna APENAS memórias de identidade, formato compacto para início de chat",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10, "description": "Máximo de memórias a retornar"}
                },
                "required": []
            }
        ),
        Tool(
            name="auto_categorize_memories",
            description="🆕 V6: Categoriza automaticamente memórias sem categoria, com aprovação batch",
            inputSchema={
                "type": "object",
                "properties": {
                    "batch_size": {"type": "integer", "default": 50, "description": "Número de memórias a processar"},
                    "min_confidence": {"type": "number", "default": 0.6, "description": "Confiança mínima para auto-categorizar"},
                    "dry_run": {"type": "boolean", "default": True, "description": "Se True, apenas sugere sem salvar"}
                },
                "required": []
            }
        ),
        Tool(
            name="migrate_to_archive",
            description="🆕 V6: Arquiva memórias (muda para 'archived', reduz importância -2, cria resumo)",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs das memórias a arquivar"
                    },
                    "create_summary": {"type": "boolean", "default": True, "description": "Criar resumo do arquivamento"}
                },
                "required": ["memory_ids"]
            }
        ),
        Tool(
            name="recategorize_memory",
            description="🆕 V6: Reclassifica manualmente uma memória para corrigir categorização",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "ID da memória"},
                    "new_category": {
                        "type": "string",
                        "description": "Nova categoria",
                        "enum": ["identity", "active_context", "active_project", "technical_knowledge", "archived"]
                    },
                    "confidence": {"type": "number", "default": 1.0, "description": "Confiança (1.0 = manual)"}
                },
                "required": ["memory_id", "new_category"]
            }
        ),


        # ========== 🆕 V6.2: BATCH REVIEW TOOLS ==========
        Tool(
            name="review_uncategorized_batch",
            description="🆕 V6.2: Retorna batch de memórias sem categoria para revisão manual",
            inputSchema={
                "type": "object",
                "properties": {
                    "batch_size": {"type": "integer", "default": 10, "description": "Quantas memórias retornar"},
                    "min_confidence": {"type": "number", "default": 0.6, "description": "Confiança mínima para sugerir"}
                },
                "required": []
            }
        ),
        Tool(
            name="apply_batch_categorization",
            description="🆕 V6.2: Aplica categorizações em lote após revisão",
            inputSchema={
                "type": "object",
                "properties": {
                    "approve": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs para categorizar com sugestão automática"
                    },
                    "reject": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs para ignorar"
                    },
                    "reclassify": {
                        "type": "object",
                        "description": "Dict {memory_id: nova_categoria} para forçar categoria"
                    }
                },
                "required": []
            }
        ),

        # ========== V5: FUNÇÕES DE AUTO-SAVE (mantidas) ==========
        Tool(
            name="trigger_auto_save",
            description="V5: Trigger manual de auto-save (conversa + annotations)",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="get_session_stats",
            description="V5: Estatísticas da sessão atual (mensagens, annotations, etc)",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),

        # ========== V2.0: SESSÃO/CONVERSA (mantidas) ==========
        Tool(
            name="save_conversation_snapshot",
            description="Salva snapshot da conversa atual",
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation_id": {"type": "string", "description": "ID único da conversa"},
                    "last_messages": {
                        "type": "array",
                        "description": "Últimas N mensagens",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["user", "assistant"]},
                                "content": {"type": "string"},
                                "timestamp": {"type": "string"}
                            }
                        }
                    },
                    "current_topic": {"type": "string", "description": "Tópico atual"},
                    "conversation_flow": {"type": "array", "items": {"type": "string"}, "description": "Fluxo de tópicos"},
                    "chat_platform_id": {"type": "string", "description": "ID do chat na plataforma"}
                },
                "required": ["conversation_id", "last_messages"]
            }
        ),
        Tool(
            name="get_active_conversation",
            description="Busca conversa ativa mais recente",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="close_conversation",
            description="Encerra uma conversa",
            inputSchema={
                "type": "object",
                "properties": {"conversation_id": {"type": "string"}},
                "required": ["conversation_id"]
            }
        ),
        Tool(
            name="list_conversations",
            description="Lista conversas",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10},
                    "active_only": {"type": "boolean", "default": False}
                },
                "required": []
            }
        ),

        # ========== V2.0: CHECKPOINTS (mantidos) ==========
        Tool(
            name="create_checkpoint",
            description="Cria checkpoint do estado atual",
            inputSchema={
                "type": "object",
                "properties": {
                    "working_on": {"type": "string", "description": "O que está sendo trabalhado"},
                    "context": {
                        "type": "object",
                        "description": "Contexto adicional",
                        "properties": {
                            "project": {"type": "string"},
                            "current_problem": {"type": "string"},
                            "solved_problems": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "next_steps": {"type": "array", "items": {"type": "string"}},
                    "open_questions": {"type": "array", "items": {"type": "string"}},
                    "decisions_made": {"type": "array", "items": {"type": "string"}},
                    "conversation_id": {"type": "string"}
                },
                "required": ["working_on"]
            }
        ),
        Tool(
            name="restore_from_checkpoint",
            description="Restaura estado de um checkpoint",
            inputSchema={
                "type": "object",
                "properties": {"checkpoint_id": {"type": "string"}},
                "required": ["checkpoint_id"]
            }
        ),
        Tool(
            name="list_checkpoints",
            description="Lista checkpoints recentes",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 5},
                    "conversation_id": {"type": "string"}
                },
                "required": []
            }
        ),
        Tool(
            name="get_latest_checkpoint",
            description="Busca checkpoint mais recente",
            inputSchema={
                "type": "object",
                "properties": {"conversation_id": {"type": "string"}},
                "required": []
            }
        ),

        # ========== V2.0: TEMPORAL (mantidos) ==========
        Tool(
            name="link_memories_timeline",
            description="Adiciona relações temporais entre memórias",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "relations": {
                        "type": "object",
                        "description": "Relações temporais",
                        "properties": {
                            "happened_before": {"type": "array", "items": {"type": "string"}},
                            "happened_after": {"type": "array", "items": {"type": "string"}},
                            "led_to": {"type": "array", "items": {"type": "string"}},
                            "caused_by": {"type": "array", "items": {"type": "string"}},
                            "concurrent_with": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "required": ["memory_id", "relations"]
            }
        ),
        Tool(
            name="get_memory_timeline",
            description="Busca memórias em período temporal",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Data inicial ISO (ex: 2025-01-01)"},
                    "end_date": {"type": "string", "description": "Data final ISO"},
                    "limit": {"type": "integer", "default": 100}
                },
                "required": ["start_date", "end_date"]
            }
        ),
        Tool(
            name="get_causal_chain",
            description="Constrói cadeia causal de uma memória",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "max_depth": {"type": "integer", "default": 5}
                },
                "required": ["memory_id"]
            }
        ),

        # ========== V2.0: WORKSPACE (mantidos) ==========
        Tool(
            name="save_workspace_context",
            description="Salva contexto do workspace atual",
            inputSchema={
                "type": "object",
                "properties": {
                    "active_files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "last_line_viewed": {"type": "integer"},
                                "modifications": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    },
                    "current_directory": {"type": "string"},
                    "terminal_history": {"type": "array", "items": {"type": "string"}},
                    "open_urls": {"type": "array", "items": {"type": "string"}},
                    "clipboard_history": {"type": "array", "items": {"type": "string"}},
                    "conversation_id": {"type": "string"}
                },
                "required": []
            }
        ),
        Tool(
            name="restore_workspace",
            description="Restaura contexto de um workspace",
            inputSchema={
                "type": "object",
                "properties": {"workspace_id": {"type": "string"}},
                "required": ["workspace_id"]
            }
        ),
        Tool(
            name="get_recent_workspaces",
            description="Busca workspaces recentes",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 3},
                    "conversation_id": {"type": "string"}
                },
                "required": []
            }
        ),

        # ========== V2.0: EMOTIONAL/TOM (mantidos) ==========
        Tool(
            name="save_conversation_tone",
            description="Salva perfil de tom da conversa",
            inputSchema={
                "type": "object",
                "properties": {
                    "mood": {"type": "string", "description": "Humor da conversa"},
                    "energy_level": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
                    "rapport_level": {"type": "number", "description": "Nível de rapport 0-10", "default": 5.0},
                    "communication_style": {
                        "type": "object",
                        "properties": {
                            "formality": {"type": "string", "enum": ["informal", "formal", "professional"]},
                            "technical_depth": {"type": "string", "enum": ["shallow", "medium", "deep"]},
                            "humor_used": {"type": "boolean"}
                        }
                    },
                    "inside_jokes": {"type": "array", "items": {"type": "string"}},
                    "user_preferences_observed": {"type": "object"},
                    "conversation_id": {"type": "string"},
                    "user_id": {"type": "string", "default": "default_user"}
                },
                "required": []
            }
        ),
        Tool(
            name="get_tone_profile",
            description="Busca perfil de tom agregado do usuário",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "default": "default_user"},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": []
            }
        ),
        Tool(
            name="list_tone_profiles",
            description="Lista perfis de tom",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": []
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """🆕 V6: Executa ferramentas com suporte a categorização"""
    try:
        # Atualiza atividade
        session_state.last_activity = time.time()
        session_state.message_count += 1

        result = None

        # ========== FERRAMENTAS ORIGINAIS (com modificações V6) ==========
        if name == "save_memory":
            content = arguments.get("content")
            metadata = arguments.get("metadata", {})
            importance = arguments.get("importance", 5)
            is_core = arguments.get("is_core", False)
            category = arguments.get("category")  # 🆕 V6

            if not content:
                return [TextContent(type="text", text="Erro: 'content' é obrigatório")]

            result = await save_memory(content, metadata, importance, is_core, category)

            # V5: Auto-annotation middleware
            await detect_and_annotate(name, arguments, result)

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "search_memory":
            query = arguments.get("query")
            if not query:
                return [TextContent(type="text", text="Erro: 'query' é obrigatório")]

            results = await search_memory(
                query,
                arguments.get("top_k", 5),
                arguments.get("min_score", 0.5),
                arguments.get("filter_type"),
                arguments.get("filter_tags"),
                arguments.get("filter_category")  # 🆕 V6
            )
            return [TextContent(type="text", text=json.dumps(results, indent=2, ensure_ascii=False))]

        elif name == "recall_context":
            query = arguments.get("query")
            if not query:
                return [TextContent(type="text", text="Erro: 'query' é obrigatório")]

            result = await recall_context(query, arguments.get("top_k", 3))
            return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]

        elif name == "forget_memory":
            memory_id = arguments.get("memory_id")
            if not memory_id:
                return [TextContent(type="text", text="Erro: 'memory_id' é obrigatório")]
            result = await forget_memory(memory_id)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "update_memory":
            memory_id = arguments.get("memory_id")
            content = arguments.get("content")
            if not memory_id or not content:
                return [TextContent(type="text", text="Erro: 'memory_id' e 'content' são obrigatórios")]
            result = await update_memory(memory_id, content, arguments.get("metadata"))
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "list_memories":
            results = await list_memories(
                arguments.get("limit", 10),
                arguments.get("filter_type"),
                arguments.get("filter_tags"),
                arguments.get("filter_category"),  # 🆕 V6
                arguments.get("sort_by", "timestamp")
            )
            return [TextContent(type="text", text=json.dumps(results, indent=2, ensure_ascii=False))]

        elif name == "load_initial_context":
            # 🆕 V6: Load hierárquico otimizado
            context_text = await load_initial_context_v6()

            # V5: Gera resumo periódico se necessário
            await generate_periodic_summary()

            return [TextContent(type="text", text=context_text)]

        # ========== 🆕 V6: NOVAS TOOLS DE CATEGORIZAÇÃO ==========
        elif name == "get_identity_core":
            limit = arguments.get("limit", 10)
            results = await get_identity_core(limit)
            return [TextContent(type="text", text=json.dumps(results, indent=2, ensure_ascii=False))]

        elif name == "auto_categorize_memories":
            result = await auto_categorize_memories(
                arguments.get("batch_size", 50),
                arguments.get("min_confidence", 0.6),
                arguments.get("dry_run", True)
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]

        elif name == "migrate_to_archive":
            memory_ids = arguments.get("memory_ids")
            if not memory_ids:
                return [TextContent(type="text", text="Erro: 'memory_ids' é obrigatório")]
            
            result = await migrate_to_archive(
                memory_ids,
                arguments.get("create_summary", True)
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "recategorize_memory":
            memory_id = arguments.get("memory_id")
            new_category = arguments.get("new_category")
            
            if not memory_id or not new_category:
                return [TextContent(type="text", text="Erro: 'memory_id' e 'new_category' são obrigatórios")]
            
            result = await recategorize_memory(
                memory_id,
                new_category,
                arguments.get("confidence", 1.0)
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]


        # ========== 🆕 V6.2: BATCH REVIEW TOOLS ==========
        elif name == "review_uncategorized_batch":
            batch_size = arguments.get("batch_size", 10)
            min_confidence = arguments.get("min_confidence", 0.6)
            
            result = await review_uncategorized_batch(batch_size, min_confidence)
            return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]

        elif name == "apply_batch_categorization":
            approve = arguments.get("approve", [])
            reject = arguments.get("reject", [])
            reclassify = arguments.get("reclassify", {})
            
            result = await apply_batch_categorization(approve, reject, reclassify)
            return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]


        # ========== V5: FUNÇÕES DE AUTO-SAVE (mantidas) ==========
        elif name == "trigger_auto_save":
            await auto_save_conversation()
            return [TextContent(type="text", text=json.dumps({
                "success": True,
                "message": "Auto-save manual executado com sucesso"
            }, indent=2))]

        elif name == "get_session_stats":
            stats = {
                "conversation_id": session_state.conversation_id,
                "message_count": session_state.message_count,
                "messages_in_buffer": len(session_state.message_history),
                "auto_annotations_pending": len(session_state.auto_annotations),
                "current_topic": session_state.current_topic,
                "last_activity": datetime.fromtimestamp(session_state.last_activity).isoformat(),
                "idle_time_seconds": int(time.time() - session_state.last_activity)
            }
            return [TextContent(type="text", text=json.dumps(stats, indent=2))]

        # ========== V2.0: SESSÃO (mantidas) ==========
        elif name == "save_conversation_snapshot":
            result = await save_conversation_snapshot(
                es_client,
                arguments.get("conversation_id"),
                arguments.get("last_messages", []),
                arguments.get("current_topic", ""),
                arguments.get("conversation_flow"),
                arguments.get("chat_platform_id", "")
            )

            # Atualiza session_state
            session_state.conversation_id = arguments.get("conversation_id")
            session_state.current_topic = arguments.get("current_topic")

            # Adiciona mensagens ao histórico
            for msg in arguments.get("last_messages", []):
                session_state.message_history.append(msg)

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_active_conversation":
            result = await get_active_conversation(es_client)
            return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]

        elif name == "close_conversation":
            conversation_id = arguments.get("conversation_id")
            if not conversation_id:
                return [TextContent(type="text", text="Erro: 'conversation_id' é obrigatório")]

            # V5: Auto-save antes de fechar
            await auto_save_conversation()

            result = await close_conversation(es_client, conversation_id)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "list_conversations":
            result = await list_conversations(
                es_client,
                arguments.get("limit", 10),
                arguments.get("active_only", False)
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]

        # ========== V2.0: CHECKPOINTS (mantidos) ==========
        elif name == "create_checkpoint":
            working_on = arguments.get("working_on")
            if not working_on:
                return [TextContent(type="text", text="Erro: 'working_on' é obrigatório")]

            result = await create_checkpoint(
                es_client,
                working_on,
                arguments.get("context"),
                arguments.get("next_steps"),
                arguments.get("open_questions"),
                arguments.get("decisions_made"),
                arguments.get("conversation_id", "")
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "restore_from_checkpoint":
            checkpoint_id = arguments.get("checkpoint_id")
            if not checkpoint_id:
                return [TextContent(type="text", text="Erro: 'checkpoint_id' é obrigatório")]
            result = await restore_from_checkpoint(es_client, checkpoint_id)
            return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]

        elif name == "list_checkpoints":
            result = await list_checkpoints(
                es_client,
                arguments.get("limit", 5),
                arguments.get("conversation_id")
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]

        elif name == "get_latest_checkpoint":
            result = await get_latest_checkpoint(
                es_client,
                arguments.get("conversation_id")
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]

        # ========== V2.0: TEMPORAL (mantidos) ==========
        elif name == "link_memories_timeline":
            memory_id = arguments.get("memory_id")
            relations = arguments.get("relations")
            if not memory_id or not relations:
                return [TextContent(type="text", text="Erro: 'memory_id' e 'relations' são obrigatórios")]
            result = await link_memories_timeline(es_client, memory_id, relations)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_memory_timeline":
            start_date = arguments.get("start_date")
            end_date = arguments.get("end_date")
            if not start_date or not end_date:
                return [TextContent(type="text", text="Erro: 'start_date' e 'end_date' são obrigatórios")]
            result = await get_memory_timeline(
                es_client,
                start_date,
                end_date,
                arguments.get("limit", 100)
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]

        elif name == "get_causal_chain":
            memory_id = arguments.get("memory_id")
            if not memory_id:
                return [TextContent(type="text", text="Erro: 'memory_id' é obrigatório")]
            result = await get_causal_chain(
                es_client,
                memory_id,
                arguments.get("max_depth", 5)
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]

        # ========== V2.0: WORKSPACE (mantidos) ==========
        elif name == "save_workspace_context":
            result = await save_workspace_context(
                es_client,
                arguments.get("active_files"),
                arguments.get("current_directory", ""),
                arguments.get("terminal_history"),
                arguments.get("open_urls"),
                arguments.get("clipboard_history"),
                arguments.get("conversation_id", "")
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "restore_workspace":
            workspace_id = arguments.get("workspace_id")
            if not workspace_id:
                return [TextContent(type="text", text="Erro: 'workspace_id' é obrigatório")]
            result = await restore_workspace(es_client, workspace_id)
            return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]

        elif name == "get_recent_workspaces":
            result = await get_recent_workspaces(
                es_client,
                arguments.get("limit", 3),
                arguments.get("conversation_id")
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]

        # ========== V2.0: EMOTIONAL/TOM (mantidos) ==========
        elif name == "save_conversation_tone":
            result = await save_conversation_tone(
                es_client,
                arguments.get("mood", "neutral"),
                arguments.get("energy_level", "medium"),
                arguments.get("rapport_level", 5.0),
                arguments.get("communication_style"),
                arguments.get("inside_jokes"),
                arguments.get("user_preferences_observed"),
                arguments.get("conversation_id", ""),
                arguments.get("user_id", "default_user")
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_tone_profile":
            result = await get_tone_profile(
                es_client,
                arguments.get("user_id", "default_user"),
                arguments.get("limit", 10)
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]

        elif name == "list_tone_profiles":
            result = await list_tone_profiles(
                es_client,
                arguments.get("user_id"),
                arguments.get("limit", 10)
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]

        else:
            return [TextContent(type="text", text=f"Erro: Ferramenta '{name}' não encontrada")]

    except Exception as e:
        logger.error(f"Erro ao executar ferramenta '{name}': {e}")
        return [TextContent(type="text", text=f"Erro: {str(e)}")]


async def main():
    """🆕 V6: Inicializa o servidor MCP V6"""
    logger.info("🚀 Iniciando servidor MCP V6 - CATEGORIZAÇÃO E PRIORIZAÇÃO...")

    try:
        if not es_client.ping():
            raise ConnectionError("Não foi possível conectar ao Elasticsearch")
        logger.info("✅ Conectado ao Elasticsearch")
    except Exception as e:
        logger.error(f"❌ Erro de conexão: {e}")
        raise

    # Inicializa todos os índices
    ensure_index_exists()
    ensure_session_index(es_client)
    ensure_checkpoint_index(es_client)
    ensure_workspace_index(es_client)
    ensure_tone_index(es_client)

    # V5: Registra hooks de shutdown
    atexit.register(lambda: asyncio.run(on_shutdown()))
    signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(on_shutdown()))
    signal.signal(signal.SIGTERM, lambda s, f: asyncio.create_task(on_shutdown()))

    # V5: Inicia verificador de inatividade
    schedule_activity_check()

    logger.info("✨ Servidor MCP V6 pronto - CATEGORIZAÇÃO E PRIORIZAÇÃO ATIVO!")
    logger.info("🆕 V6 Features:")
    logger.info("  - Campo memory_category (identity, active_context, active_project, technical_knowledge, archived)")
    logger.info("  - Load hierárquico otimizado (~30-40 memórias vs 117)")
    logger.info("  - Auto-detecção inteligente de categoria com confidence")
    logger.info("  - Tools: get_identity_core, auto_categorize_memories, migrate_to_archive, recategorize_memory")
    logger.info("✨ V5 Features (mantidas):")
    logger.info("  - Auto-save (fechamento + timeout 5min)")
    logger.info("  - Auto-annotation (arquivos, problemas, Mirror)")
    logger.info("  - Resumo periódico (a cada 50 mensagens)")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
