# Chain of Thought Strategies Explained

> A simple guide to understanding CoT prompting techniques - explained like you're teaching a curious 10-year-old!

---

## Table of Contents

1. [What is Chain of Thought?](#what-is-chain-of-thought)
2. [Strategy 1: Standard CoT](#strategy-1-standard-cot)
3. [Strategy 2: Zero-Shot CoT](#strategy-2-zero-shot-cot)
4. [Strategy 3: Self-Consistency CoT](#strategy-3-self-consistency-cot)
5. [Comparison Table](#comparison-table)
6. [When to Use Which Strategy](#when-to-use-which-strategy)

---

## What is Chain of Thought?

### The Problem

Imagine you ask a calculator: "What is 15% of 240?"

The calculator just says: **36**

But if you ask your math teacher, they might say:

> "Well, 15% means 15 out of 100. So we write it as 0.15. Then we multiply 0.15 × 240 = 36"

**Which answer helps you learn better?** The teacher's, right?

### The Solution

**Chain of Thought (CoT)** is a way of asking AI to "show its work" - just like your teacher does!

Instead of jumping straight to the answer, the AI:
1. Breaks down the problem
2. Thinks through each step
3. Then gives the final answer

### Why Does This Matter?

```
Without CoT:  "The answer is 36"
With CoT:     "Let me think... 15% = 0.15, then 0.15 × 240 = 36"
```

Research shows that when AI "thinks out loud," it makes **fewer mistakes** on hard problems!

---

## Strategy 1: Standard CoT

### Definition (Simple)

**Standard CoT** is like giving the AI a detailed instruction manual:

> "Hey AI, please solve this step by step. Number each step. Show your work. Then give me the answer."

### Definition (Technical)

Standard Chain of Thought uses explicit prompting with detailed instructions that guide the model to:
- Break down problems into numbered steps
- Show reasoning for each step
- Conclude with a clearly marked final answer

### Real-World Analogy

Think of it like a **recipe book**:

```
Recipe: Make a Sandwich

Step 1: Get two slices of bread
Step 2: Spread peanut butter on one slice
Step 3: Spread jelly on the other slice
Step 4: Put the slices together

Result: A delicious PB&J sandwich!
```

We're telling the AI: "Give me a recipe for solving this problem!"

### Example

**Without Standard CoT:**
```
User: What is 23 × 17?
AI: 391
```

**With Standard CoT:**
```
User: What is 23 × 17? Let's think through this step by step.

AI:
Step 1: I'll break this into smaller multiplications.
Step 2: 23 × 17 = 23 × (10 + 7)
Step 3: 23 × 10 = 230
Step 4: 23 × 7 = 161
Step 5: 230 + 161 = 391

Answer: 391
```

### How It's Implemented in Our Code

**File:** `src/cot_reasoner/strategies/standard.py`

```python
class StandardCoTStrategy(BaseStrategy):
    """Standard Chain of Thought reasoning strategy."""
```

**The Prompt Template** (`src/cot_reasoner/core/prompts.py`):

```python
STANDARD_COT_PROMPT = """Please solve the following problem step by step.

Problem: {query}

Let's think through this step by step:"""
```

**The System Instructions:**

```python
STANDARD_COT_SYSTEM = """You are a reasoning assistant. When solving problems:
1. Break down the problem into clear steps
2. Show your work for each step
3. Number each reasoning step
4. End with a clear final answer

Format your response as:
Step 1: [First reasoning step]
Step 2: [Second reasoning step]
...
Answer: [Your final answer]"""
```

**How the Code Works:**

```python
def reason(self, query: str, context: Optional[str] = None, **kwargs):
    # 1. Create an empty chain to store results
    chain = self._create_chain(query)

    # 2. Build the prompt with our CoT template
    prompt = self._build_prompt(query, context)

    # 3. Send to LLM with our system instructions
    response = self.provider.generate(
        prompt=prompt,
        system_prompt=STANDARD_COT_SYSTEM,
    )

    # 4. Parse the response into structured steps
    chain = self.parse_response(response.content, chain)

    return chain
```

### Visual Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    STANDARD CoT                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   User Question                                         │
│        ↓                                                │
│   "Please solve step by step..."  ← Detailed Prompt     │
│        ↓                                                │
│   ┌─────────────────────────────┐                       │
│   │         LLM                 │                       │
│   │  (follows instructions)     │                       │
│   └─────────────────────────────┘                       │
│        ↓                                                │
│   Step 1: ...                                           │
│   Step 2: ...                                           │
│   Step 3: ...                                           │
│   Answer: ...                                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Strategy 2: Zero-Shot CoT

### Definition (Simple)

**Zero-Shot CoT** is like a magic phrase. You just add these words to any question:

> "Let's think step by step."

That's it! No examples. No detailed instructions. Just that one magic phrase!

### Definition (Technical)

Zero-Shot Chain of Thought uses minimal prompting by simply appending the phrase "Let's think step by step" to the query. The model relies on its inherent reasoning capabilities without explicit formatting instructions.

**Paper:** "Large Language Models are Zero-Shot Reasoners" (Kojima et al., 2022)

### Real-World Analogy

Imagine you have a smart friend. You don't need to give them a recipe book. You just say:

> "Hey, can you figure out how to make a sandwich? Think through it step by step."

And they naturally break it down because they're smart!

**Zero-Shot** means "zero examples" - we don't show the AI any examples of how to solve problems. We just trust it knows how!

### Example

**The Magic Phrase in Action:**

```
User: A bat and a ball cost $1.10 in total. The bat costs $1.00 more
than the ball. How much does the ball cost?

Let's think step by step.

AI:
Let me work through this carefully.

If the ball costs X dollars, then the bat costs X + $1.00

Together: X + (X + $1.00) = $1.10
         2X + $1.00 = $1.10
         2X = $0.10
         X = $0.05

The ball costs $0.05 (5 cents).

Let me verify: Ball ($0.05) + Bat ($1.05) = $1.10 ✓
And the bat costs exactly $1.00 more than the ball ✓
```

Without "Let's think step by step," many people (and AIs!) incorrectly say "$0.10"!

### How It's Implemented in Our Code

**File:** `src/cot_reasoner/strategies/zero_shot.py`

```python
class ZeroShotCoTStrategy(BaseStrategy):
    """Zero-shot Chain of Thought reasoning strategy.

    This strategy uses minimal prompting with just "Let's think step by step"
    appended to the query. No examples or detailed instructions are provided.
    """
```

**The Prompt Template** (`src/cot_reasoner/core/prompts.py`):

```python
ZERO_SHOT_COT_PROMPT = """{query}

Let's think step by step."""
```

That's it! Super simple!

**The System Instructions:**

```python
ZERO_SHOT_COT_SYSTEM = """You are a logical reasoning assistant. When presented
with any problem, think through it step by step before providing your answer.
Always show your reasoning process."""
```

**How the Code Works:**

```python
def reason(self, query: str, context: Optional[str] = None, **kwargs):
    chain = self._create_chain(query)

    # The prompt is simply: "{query}\n\nLet's think step by step."
    prompt = self._build_prompt(query, context)

    response = self.provider.generate(
        prompt=prompt,
        system_prompt=ZERO_SHOT_COT_SYSTEM,
    )

    chain = self.parse_response(response.content, chain)
    return chain
```

### Visual Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    ZERO-SHOT CoT                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   User Question                                         │
│        +                                                │
│   "Let's think step by step."  ← Just this phrase!     │
│        ↓                                                │
│   ┌─────────────────────────────┐                       │
│   │         LLM                 │                       │
│   │  (figures it out itself)    │                       │
│   └─────────────────────────────┘                       │
│        ↓                                                │
│   Natural reasoning output                              │
│   (format may vary)                                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Standard vs Zero-Shot: What's the Difference?

| Aspect | Standard CoT | Zero-Shot CoT |
|--------|--------------|---------------|
| Prompt length | Long, detailed | Short, minimal |
| Instructions | "Number steps, show work, format like this..." | "Let's think step by step." |
| Output format | Predictable (Step 1, Step 2...) | Varies |
| Token usage | More tokens | Fewer tokens |
| Best for | Complex problems needing structure | Quick reasoning tasks |

---

## Strategy 3: Self-Consistency CoT

### Definition (Simple)

**Self-Consistency** is like asking the same question to 5 different smart friends, then going with the answer that most of them agree on!

If 4 out of 5 friends say "The answer is 36" and 1 says "The answer is 42," you trust the 36!

### Definition (Technical)

Self-Consistency generates multiple independent reasoning paths for the same problem, then uses **majority voting** to select the most common answer. This improves reliability by reducing the impact of any single flawed reasoning chain.

**Paper:** "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (Wang et al., 2022)

### Real-World Analogy

Imagine you're on a game show and you can "poll the audience":

```
Question: What's the capital of France?

Audience Member 1: "I think it's Paris because..."  → Paris
Audience Member 2: "Let me see... it's Paris..."    → Paris
Audience Member 3: "Could be Lyon? No, Paris..."   → Paris
Audience Member 4: "I'll guess Paris..."           → Paris
Audience Member 5: "Maybe Nice? Actually Paris..." → Paris

Final Answer: Paris (100% agreement, high confidence!)
```

Even if one person makes a mistake, the majority wins!

### Example

**Problem:** "If a train travels at 60 mph for 2.5 hours, how far does it go?"

**Path 1:**
```
Distance = Speed × Time
Distance = 60 × 2.5
Let me calculate: 60 × 2.5 = 150
Answer: 150 miles
```

**Path 2:**
```
Speed is 60 miles per hour
Time is 2.5 hours
In 1 hour: 60 miles
In 2 hours: 120 miles
In 0.5 hours: 30 miles
Total: 120 + 30 = 150 miles
Answer: 150 miles
```

**Path 3:**
```
Using d = rt formula
d = 60 mph × 2.5 h
d = 150 miles
Answer: 150 miles
```

**Majority Vote:**
- "150 miles": 3 votes (100%)
- Confidence: Very high!

### How It's Implemented in Our Code

**File:** `src/cot_reasoner/strategies/self_consistency.py`

```python
class SelfConsistencyStrategy(BaseStrategy):
    """Self-consistency Chain of Thought reasoning strategy.

    This strategy generates multiple reasoning paths and selects the most
    consistent answer through majority voting.
    """

    def __init__(self, provider, num_samples: int = 3, temperature: float = 0.7):
        super().__init__(provider)
        self.num_samples = num_samples      # How many "friends" to ask
        self.sample_temperature = temperature  # Add randomness for diversity
```

**The Core Logic:**

```python
def reason(self, query: str, context: Optional[str] = None, **kwargs):
    chain = self._create_chain(query)
    prompt = self._build_prompt(query, context)

    answers = []
    all_chains = []

    # Step 1: Generate multiple reasoning paths
    for i in range(self.num_samples):  # Default: 3 times
        response = self.provider.generate(
            prompt=prompt,
            system_prompt=SELF_CONSISTENCY_SYSTEM,
            temperature=self.sample_temperature,  # Adds variety
        )

        # Parse each response
        temp_chain = self.parse_response(response.content, temp_chain)

        if temp_chain.answer:
            normalized = self._normalize_answer(temp_chain.answer)
            answers.append((normalized, temp_chain.answer, temp_chain))

    # Step 2: Majority voting
    if answers:
        final_answer, confidence = self._majority_vote(answers)
        chain.set_answer(final_answer, confidence)

    return chain
```

**The Majority Voting Function:**

```python
def _majority_vote(self, answers):
    """Pick the most common answer."""

    # Count how many times each answer appears
    vote_counts = Counter(a[0] for a in answers)

    # Get the winner
    most_common = vote_counts.most_common(1)[0]
    winning_answer = most_common[0]
    winning_count = most_common[1]

    # Confidence = what percentage agreed
    confidence = winning_count / len(answers)

    return winning_answer, confidence
```

**Answer Normalization** (so "36" and "The answer is 36" count as the same):

```python
def _normalize_answer(self, answer: str) -> str:
    """Make answers comparable."""

    normalized = answer.lower().strip()

    # Remove common prefixes
    prefixes = ["the answer is", "therefore", "so", "thus"]
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()

    # Remove punctuation
    normalized = normalized.rstrip(".,!?")

    return normalized
```

### Visual Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    SELF-CONSISTENCY CoT                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User Question                                                 │
│        ↓                                                        │
│   ┌─────────┬─────────┬─────────┐                               │
│   │ Path 1  │ Path 2  │ Path 3  │  ← Same question, 3 times    │
│   │         │         │         │                               │
│   │ Step 1  │ Step 1  │ Step 1  │     (with temperature for    │
│   │ Step 2  │ Step 2  │ Step 2  │      variety)                │
│   │ Step 3  │ Step 3  │ Step 3  │                               │
│   │   ↓     │   ↓     │   ↓     │                               │
│   │ Ans: 36 │ Ans: 36 │ Ans: 42 │                               │
│   └────┬────┴────┬────┴────┬────┘                               │
│        │         │         │                                    │
│        └─────────┼─────────┘                                    │
│                  ↓                                              │
│        ┌─────────────────┐                                      │
│        │  MAJORITY VOTE  │                                      │
│        │                 │                                      │
│        │  36: 2 votes    │                                      │
│        │  42: 1 vote     │                                      │
│        │                 │                                      │
│        │  Winner: 36     │                                      │
│        │  Confidence: 67%│                                      │
│        └─────────────────┘                                      │
│                  ↓                                              │
│           Final Answer: 36                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why Use Temperature?

**Temperature** controls randomness:

- `temperature = 0`: Always gives the same answer (deterministic)
- `temperature = 0.7`: Some variety (good for self-consistency)
- `temperature = 1.0+`: Very creative/random

We use `0.7` so each path might reason differently, giving us diverse perspectives!

---

## Comparison Table

| Feature | Standard CoT | Zero-Shot CoT | Self-Consistency |
|---------|--------------|---------------|------------------|
| **Complexity** | Medium | Simple | Complex |
| **Prompt** | Detailed instructions | "Let's think step by step" | Same as Standard |
| **API Calls** | 1 | 1 | N (default: 3) |
| **Cost** | $ | $ | $$$ |
| **Speed** | Fast | Fast | Slow (N× calls) |
| **Reliability** | Good | Good | Best |
| **Best For** | Most problems | Quick queries | Critical decisions |
| **Output** | Structured | Variable | Structured + Voting |

---

## When to Use Which Strategy

### Use Standard CoT When:

- You want **predictable, formatted output**
- The problem needs **clear step-by-step breakdown**
- You're building a **production system** that parses responses
- Example: Math tutoring app, technical documentation

```python
reasoner = Reasoner(strategy="standard")
```

### Use Zero-Shot CoT When:

- You want **quick answers** with minimal tokens
- The problem is **relatively simple**
- You don't need **strict formatting**
- Example: Quick Q&A, chatbots, simple reasoning

```python
reasoner = Reasoner(strategy="zero_shot")
```

### Use Self-Consistency When:

- The answer **really matters** (high stakes)
- You can afford **more API calls** (cost/time)
- You want **higher confidence** in the result
- Example: Medical advice, financial decisions, exams

```python
reasoner = Reasoner(strategy="self_consistency", num_samples=5)
```

---

## Summary for Kids

| Strategy | Like... |
|----------|---------|
| **Standard CoT** | Following a recipe book with numbered steps |
| **Zero-Shot CoT** | Saying "think out loud please!" |
| **Self-Consistency** | Asking 5 friends and trusting the majority |

---

## Try It Yourself!

```python
from cot_reasoner import Reasoner

# Try each strategy on the same problem
question = "If I have 3 boxes with 4 apples each, and I eat 2 apples, how many are left?"

# Standard CoT
reasoner1 = Reasoner(strategy="standard")
result1 = reasoner1.reason(question)
print("Standard:", result1.answer)

# Zero-Shot CoT
reasoner2 = Reasoner(strategy="zero_shot")
result2 = reasoner2.reason(question)
print("Zero-Shot:", result2.answer)

# Self-Consistency CoT
reasoner3 = Reasoner(strategy="self_consistency", num_samples=3)
result3 = reasoner3.reason(question)
print("Self-Consistency:", result3.answer, f"(Confidence: {result3.confidence:.0%})")
```

---

## References

1. **Chain of Thought Prompting**: Wei et al., 2022 - "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"

2. **Zero-Shot CoT**: Kojima et al., 2022 - "Large Language Models are Zero-Shot Reasoners"

3. **Self-Consistency**: Wang et al., 2022 - "Self-Consistency Improves Chain of Thought Reasoning in Language Models"

---

*Created for the [cot-reasoner](https://github.com/ajafarsadiq2002/cot-reasoner) project*
