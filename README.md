[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/fredac100-elasticsearch-memory-mcp-badge.png)](https://mseep.ai/app/fredac100-elasticsearch-memory-mcp)

# 🧠 Elasticsearch Memory MCP

[![PyPI](https://img.shields.io/pypi/v/elasticsearch-memory-mcp)](https://pypi.org/project/elasticsearch-memory-mcp/)
[![MCP](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/pypi/pyversions/elasticsearch-memory-mcp)](https://pypi.org/project/elasticsearch-memory-mcp/)

A powerful **Model Context Protocol (MCP)** server that provides persistent, intelligent memory using Elasticsearch with hierarchical categorization and semantic search capabilities.

## ✨ Features

### 🎯 V6.2 - Latest Release

- **🏷️ Hierarchical Memory Categorization**
  - 5 category types: `identity`, `active_context`, `active_project`, `technical_knowledge`, `archived`
  - Automatic category detection with confidence scoring
  - Manual reclassification support

- **🤖 Intelligent Auto-Detection**
  - Accumulative scoring system (0.7-0.95 confidence range)
  - 23+ specialized keyword patterns
  - Context-aware categorization

- **📦 Batch Review System**
  - Review uncategorized memories in batches
  - Approve/reject/reclassify workflows
  - 10x faster than individual categorization

- **🔄 Backward Compatible Fallback**
  - Seamlessly loads v5 uncategorized memories
  - No data loss during upgrades
  - Graceful degradation

- **🚀 Optimized Context Loading**
  - Hierarchical priority loading (~30-40 memories vs 117)
  - 60-70% token reduction
  - Smart relevance ranking

- **💾 Persistent Memory**
  - Vector embeddings for semantic search
  - Session management with checkpoints
  - Conversation snapshots

## 🛠️ Installation

### Quick Start (Recommended)

Install directly from PyPI:

```bash
pip install elasticsearch-memory-mcp
```

### Prerequisites

- Python 3.8+
- Elasticsearch 8.0+

### Step 1: Start Elasticsearch

```bash
# Using Docker (recommended)
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.0.0

# Or install locally
# https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html
```

### Step 2: Configure MCP

#### For Claude Desktop

Add to `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "elasticsearch-memory": {
      "command": "uvx",
      "args": ["elasticsearch-memory-mcp"],
      "env": {
        "ELASTICSEARCH_URL": "http://localhost:9200"
      }
    }
  }
}
```

> **Note**: If you don't have `uvx`, install with `pip install uvx` or use `python -m elasticsearch_memory_mcp` instead.

#### For Claude Code CLI

```bash
claude mcp add elasticsearch-memory uvx elasticsearch-memory-mcp \
  -e ELASTICSEARCH_URL=http://localhost:9200
```

### Alternative: Install from Source

If you want to contribute or modify the code:

```bash
# Clone repository
git clone https://github.com/fredac100/elasticsearch-memory-mcp.git
cd elasticsearch-memory-mcp

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .
```

Then configure MCP pointing to your local installation:

```json
{
  "mcpServers": {
    "elasticsearch-memory": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "mcp_server"],
      "env": {
        "ELASTICSEARCH_URL": "http://localhost:9200"
      }
    }
  }
}
```

## 📚 Usage

### Available Tools

#### 1. **save_memory**
Save a new memory with automatic categorization.

```json
{
  "content": "Fred prefers direct, brutal communication style",
  "type": "user_profile",
  "importance": 9,
  "tags": ["communication", "preference"]
}
```

#### 2. **load_initial_context** (Resource)
Loads hierarchical context with:
- Identity memories (who you are)
- Active context (current work)
- Active projects (ongoing)
- Technical knowledge (relevant facts)

#### 3. **review_uncategorized_batch** 🆕 V6.2
Review uncategorized memories in batches.

```json
{
  "batch_size": 10,
  "min_confidence": 0.6
}
```

Returns suggestions with auto-detected categories and confidence scores.

#### 4. **apply_batch_categorization** 🆕 V6.2
Apply categorizations in batch after review.

```json
{
  "approve": ["id1", "id2"],           // Auto-categorize
  "reject": ["id3"],                    // Skip
  "reclassify": {"id4": "archived"}    // Force category
}
```

#### 5. **search_memory**
Semantic search with filters.

```json
{
  "query": "SAE project details",
  "limit": 5,
  "category": "active_project"
}
```

#### 6. **auto_categorize_memories**
Batch auto-categorize uncategorized memories.

```json
{
  "max_to_process": 50,
  "min_confidence": 0.75
}
```

## 🏗️ Architecture

```
┌─────────────────┐
│  Claude (MCP)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  MCP Server (v6.2)          │
│  ┌─────────────────────┐    │
│  │ Auto-Detection      │    │
│  │ - Keyword matching  │    │
│  │ - Confidence score  │    │
│  └─────────────────────┘    │
│                              │
│  ┌─────────────────────┐    │
│  │ Batch Review        │    │
│  │ - Review workflow   │    │
│  │ - Bulk operations   │    │
│  └─────────────────────┘    │
└──────────┬──────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Elasticsearch               │
│  ┌────────────────────────┐  │
│  │ memories (index)       │  │
│  │ - embeddings (vector)  │  │
│  │ - memory_category      │  │
│  │ - category_confidence  │  │
│  └────────────────────────┘  │
└──────────────────────────────┘
```

## 📊 Category System

| Category | Description | Examples |
|----------|-------------|----------|
| **identity** | Core identity, values, preferences | "Fred prefers brutal honesty" |
| **active_context** | Current work, recent conversations | "Working on SAE implementation" |
| **active_project** | Ongoing projects | "Mirror architecture design" |
| **technical_knowledge** | Facts, configs, tools | "Elasticsearch index settings" |
| **archived** | Completed, deprecated, old migrations | "Refactored old auth system" |

## 🎯 Auto-Detection Examples

### High Confidence (0.8-0.95)

```
"Fred prefere comunicação brutal" → identity (0.9)
"Refatoração do sistema SAE concluída" → archived (0.85)
"Próximos passos: implementar dashboard" → active_context (0.8)
```

### Multiple Keywords (Accumulative Scoring)

```
"Fred prefere comunicação brutal. Primeira vez usando este estilo."
  → Match 1: "Fred prefere" (+0.9)
  → Match 2: "primeira vez" (+0.8)
  → Total: 0.95 (normalized)
```

## 🔄 Migration from V5

The v6.2 system includes automatic fallback for v5 memories:

1. **Uncategorized memories** → Loaded via type/tags fallback
2. **Visual separation** → Categorized vs. fallback sections
3. **Batch review** → Categorize old memories efficiently

```bash
# Review and categorize v5 memories
review_uncategorized_batch(batch_size=20)
apply_batch_categorization(approve=[...])
```

## 🚀 Performance

- **Load initial context**: ~10-15s (includes embedding model load)
- **Save memory**: <1s
- **Search**: <500ms
- **Batch review (10 items)**: ~2s
- **Auto-categorize (50 items)**: ~5s

## 🧪 Testing

```bash
# Run quick test
python test_quick.py

# Expected output:
# ✅ Elasticsearch connected
# ✅ Context loaded
# ✅ Identity memories found
# ✅ Projects separated from fallback
```

## 📝 Changelog

### V6.2 (Latest)
- ✅ Improved auto-detection (0.4 → 0.9 confidence)
- ✅ 23 new specialized keywords
- ✅ Batch review tools (review_uncategorized_batch, apply_batch_categorization)
- ✅ Visual separation (categorized vs fallback)
- ✅ Accumulative confidence scoring

### V6.1
- ✅ Fallback mechanism for uncategorized memories
- ✅ Backward compatibility with v5

### V6.0
- ✅ Memory categorization system
- ✅ Hierarchical context loading
- ✅ Auto-detection with confidence

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Model Context Protocol (MCP)](https://modelcontextprotocol.io)
- Powered by [Elasticsearch](https://www.elastic.co)
- Embeddings by [Sentence Transformers](https://www.sbert.net)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/fredac100/elasticsearch-memory-mcp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/fredac100/elasticsearch-memory-mcp/discussions)

---

Made with ❤️ for the Claude ecosystem
