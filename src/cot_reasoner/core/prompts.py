"""Prompt templates for Chain of Thought reasoning strategies."""

# System prompts
REASONER_SYSTEM_PROMPT = """You are an expert reasoning assistant that breaks down complex problems into clear, logical steps.

When given a problem:
1. Analyze the problem carefully
2. Break it down into smaller, manageable steps
3. Work through each step systematically
4. Arrive at a well-reasoned conclusion

Always show your reasoning process clearly. Be thorough but concise."""

# Standard Chain of Thought prompts
STANDARD_COT_PROMPT = """Please solve the following problem step by step.

Problem: {query}

Let's think through this step by step:"""

STANDARD_COT_SYSTEM = """You are a reasoning assistant. When solving problems:
1. Break down the problem into clear steps
2. Show your work for each step
3. Number each reasoning step
4. End with a clear final answer

IMPORTANT: If previous conversation context is provided, use it to understand references like "that", "this", "the result", etc. The user may be referring to values or results from earlier questions.

Format your response as:
Step 1: [First reasoning step]
Step 2: [Second reasoning step]
...
Answer: [Your final answer]"""

# Zero-shot Chain of Thought prompts
ZERO_SHOT_COT_PROMPT = """{query}

Let's think step by step."""

ZERO_SHOT_COT_SYSTEM = """You are a logical reasoning assistant. When presented with any problem, think through it step by step before providing your answer. Always show your reasoning process.

IMPORTANT: If previous conversation context is provided, use it to understand references like "that", "this", "the result", etc. The user may be referring to values or results from earlier questions."""

# Self-consistency prompts
SELF_CONSISTENCY_PROMPT = """Problem: {query}

Please solve this problem using careful reasoning. Show your complete thought process step by step, then provide your final answer.

Think step by step:"""

SELF_CONSISTENCY_SYSTEM = """You are an expert problem solver. For each problem:
1. Consider the problem from multiple angles
2. Work through the solution step by step
3. Double-check your reasoning
4. Provide a confident final answer

IMPORTANT: If previous conversation context is provided, use it to understand references like "that", "this", "the result", etc. The user may be referring to values or results from earlier questions.

Always show clear, numbered reasoning steps."""

# Parsing prompts - for extracting structured responses
EXTRACT_STEPS_PROMPT = """Given the following reasoning response, extract the individual reasoning steps and the final answer.

Response:
{response}

Please output in this exact JSON format:
{{
    "steps": [
        {{"number": 1, "content": "First reasoning step"}},
        {{"number": 2, "content": "Second reasoning step"}}
    ],
    "answer": "The final answer",
    "confidence": 0.95
}}

Extract the steps and answer:"""

# Advanced reasoning prompts
STRUCTURED_REASONING_PROMPT = """Problem: {query}

Analyze this problem using structured reasoning:

1. UNDERSTAND: What is being asked?
2. IDENTIFY: What information do we have?
3. PLAN: How should we approach this?
4. EXECUTE: Work through the solution
5. VERIFY: Check the answer makes sense

Let's begin:"""

MATH_REASONING_PROMPT = """Mathematical Problem: {query}

Solve this step by step, showing all calculations:

Given:
- Identify all given values and variables

Solution:
- Show each calculation step
- Explain the mathematical operations used

Answer:
- State the final numerical answer with units if applicable

Let's solve this:"""

CODE_REASONING_PROMPT = """Programming Problem: {query}

Think through this coding problem systematically:

1. Understanding: What does the problem ask for?
2. Input/Output: What are the inputs and expected outputs?
3. Approach: What algorithm or method should we use?
4. Edge Cases: What special cases should we consider?
5. Solution: Implement the solution

Let's work through this:"""


def get_prompt_template(strategy: str, prompt_type: str = "user") -> str:
    """Get the appropriate prompt template for a strategy.

    Args:
        strategy: The reasoning strategy (standard, zero_shot, self_consistency)
        prompt_type: Either 'user' for the user prompt or 'system' for system prompt

    Returns:
        The prompt template string
    """
    templates = {
        "standard": {
            "user": STANDARD_COT_PROMPT,
            "system": STANDARD_COT_SYSTEM,
        },
        "zero_shot": {
            "user": ZERO_SHOT_COT_PROMPT,
            "system": ZERO_SHOT_COT_SYSTEM,
        },
        "self_consistency": {
            "user": SELF_CONSISTENCY_PROMPT,
            "system": SELF_CONSISTENCY_SYSTEM,
        },
    }

    strategy_templates = templates.get(strategy, templates["standard"])
    return strategy_templates.get(prompt_type, strategy_templates["user"])


def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with provided values.

    Args:
        template: The prompt template string
        **kwargs: Values to substitute in the template

    Returns:
        Formatted prompt string
    """
    return template.format(**kwargs)
