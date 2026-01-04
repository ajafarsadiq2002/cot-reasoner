# CoT Reasoner

A Python Chain of Thought (CoT) reasoning library with multi-provider LLM support.

## What is Chain of Thought Reasoning?

Chain of Thought is a prompting technique that encourages LLMs to break down complex problems into intermediate reasoning steps before arriving at a final answer. This improves accuracy on complex tasks like math, logic, and multi-step reasoning.

## Features

- **Multiple LLM Providers**: OpenAI (GPT-4) and Anthropic (Claude) with extensible architecture
- **Multiple CoT Strategies**: Standard, Zero-Shot, and Self-Consistency
- **Three Interfaces**: Python module, CLI application, and REST API
- **Conversation Memory**: Maintain context across multiple questions for follow-up reasoning
- **SQLite Persistence**: Automatically saves reasoning results to a local database
- **Streaming Support**: Real-time streaming of reasoning process
- **Structured Output**: Access individual reasoning steps and confidence scores

## Installation

```bash
# Clone the repository
cd cot_reasoner

# Install with pip
pip install -e .

# Or with uv
uv pip install -e .
```

## Configuration

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
```

## Usage

### Python Module

```python
from cot_reasoner import Reasoner

# Create reasoner with defaults (OpenAI, standard CoT)
reasoner = Reasoner()

# Or customize
reasoner = Reasoner(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    strategy="self_consistency"
)

# Perform reasoning
result = reasoner.reason("What is 15% of 240?")

# Access results
print(f"Query: {result.query}")
for step in result.steps:
    print(f"Step {step.number}: {step.content}")
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence:.0%}")
```

### Conversation Memory

Enable memory to maintain context across multiple questions:

```python
from cot_reasoner import Reasoner

# Enable memory for contextual conversations
reasoner = Reasoner(memory=True)

# First question
result1 = reasoner.reason("What is 15% of 240?")
print(result1.answer)  # "15% of 240 is 36."

# Follow-up question (references previous answer)
result2 = reasoner.reason("Double that")
print(result2.answer)  # "Double that is 72."

# Check conversation history
print(reasoner.memory.get_context())

# Clear memory when needed
reasoner.clear_memory()
```

### Async Usage

```python
import asyncio
from cot_reasoner import Reasoner

async def main():
    reasoner = Reasoner()
    result = await reasoner.reason_async("Complex problem here...")
    print(result.answer)

asyncio.run(main())
```

### CLI Application

```bash
# Single query
cot-reasoner "What is the sum of the first 10 prime numbers?"

# With options
cot-reasoner -p anthropic -s self_consistency "Your question"

# Interactive mode (memory enabled by default)
cot-reasoner

# Interactive mode without memory
cot-reasoner --no-memory

# Stream output
cot-reasoner --stream "Explain quantum computing"

# JSON output
cot-reasoner --json "Math problem"

# List providers and strategies
cot-reasoner providers
cot-reasoner strategies
```

**Interactive Mode Commands:**
- `history` - View conversation history
- `clear` - Clear conversation memory
- `config` - Show current configuration
- `debug` - Show context being sent to LLM
- `quit` / `exit` - Exit the application

### REST API

Start the server:

```bash
cot-reasoner-api --port 8000
```

Endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Perform reasoning
curl -X POST http://localhost:8000/reason \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 15% of 240?", "provider": "openai"}'

# Async reasoning
curl -X POST http://localhost:8000/reason/async \
  -H "Content-Type: application/json" \
  -d '{"query": "Complex problem"}'

# Get result by ID
curl http://localhost:8000/reason/{task_id}

# List providers/strategies
curl http://localhost:8000/providers
curl http://localhost:8000/strategies
```

API documentation: http://localhost:8000/docs

## Reasoning Strategies

### Standard CoT
Explicit step-by-step prompting with numbered reasoning steps.
```python
reasoner = Reasoner(strategy="standard")
```

### Zero-Shot CoT
Minimal prompting with just "Let's think step by step".
```python
reasoner = Reasoner(strategy="zero_shot")
```

### Self-Consistency
Generates multiple reasoning paths and uses majority voting.
```python
reasoner = Reasoner(strategy="self_consistency", num_samples=5)
```

## Extending

### Custom Provider

```python
from cot_reasoner import Reasoner
from cot_reasoner.providers.base import BaseLLMProvider

class OllamaProvider(BaseLLMProvider):
    @property
    def provider_name(self) -> str:
        return "ollama"

    def generate(self, prompt, system_prompt=None, **kwargs):
        # Implementation here
        pass

    # Implement other abstract methods...

# Register and use
Reasoner.register_provider("ollama", OllamaProvider)
reasoner = Reasoner(provider="ollama", model="llama2")
```

### Custom Strategy

```python
from cot_reasoner import Reasoner
from cot_reasoner.strategies.base import BaseStrategy

class TreeOfThoughtStrategy(BaseStrategy):
    @property
    def strategy_name(self) -> str:
        return "tree_of_thought"

    def reason(self, query, **kwargs):
        # Implementation here
        pass

    # Implement other abstract methods...

# Register and use
Reasoner.register_strategy("tree_of_thought", TreeOfThoughtStrategy)
reasoner = Reasoner(strategy="tree_of_thought")
```

## Project Structure

```
cot_reasoner/
├── src/cot_reasoner/
│   ├── __init__.py          # Package exports
│   ├── core/
│   │   ├── chain.py         # ReasoningStep, ReasoningChain
│   │   ├── memory.py        # ConversationMemory for context
│   │   ├── prompts.py       # CoT prompt templates
│   │   └── reasoner.py      # Main Reasoner class
│   ├── providers/
│   │   ├── base.py          # BaseLLMProvider
│   │   ├── openai.py        # OpenAI implementation
│   │   └── anthropic.py     # Anthropic implementation
│   ├── strategies/
│   │   ├── base.py          # BaseStrategy
│   │   ├── standard.py      # Standard CoT
│   │   ├── zero_shot.py     # Zero-shot CoT
│   │   └── self_consistency.py
│   ├── cli.py               # CLI application
│   ├── api.py               # REST API
│   └── db.py                # SQLite database
├── data/                    # SQLite database storage
└── tests/
```

## License

MIT License
