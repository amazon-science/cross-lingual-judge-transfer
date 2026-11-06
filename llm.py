# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""AWS Bedrock LLM interface with model registry."""

import json
import time
from typing import Optional, Callable, Dict, Any
import boto3
from botocore.config import Config


# Response extractors - reusable functions for different response formats
def openai_extract(r: Dict) -> str:
    """Extract from OpenAI-style response: choices[0].message.content"""
    return r["choices"][0]["message"]["content"]

def anthropic_extract(r: Dict) -> str:
    """Extract from Anthropic response: content[0].text"""
    return r["content"][0]["text"]

def converse_extract(r: Dict) -> str:
    """Extract from Bedrock Converse API response: output.message.content[0].text"""
    return r["output"]["message"]["content"][0]["text"]

def llama_extract(r: Dict) -> str:
    """Extract from Llama response: generation"""
    return r["generation"]


# Model Registry - each model defines its Bedrock ID and response extractor
MODELS: Dict[str, Dict[str, Any]] = {
    # Llama models
    "llama90b": {"id": "us.meta.llama3-2-90b-instruct-v1:0", "extract": llama_extract},
    "llama11b": {"id": "us.meta.llama3-2-11b-instruct-v1:0", "extract": llama_extract},
    "llama70b": {"id": "us.meta.llama3-3-70b-instruct-v1:0", "extract": llama_extract},
    "llama8b":  {"id": "us.meta.llama3-1-8b-instruct-v1:0",  "extract": llama_extract},
    
    # Claude models
    "claude3.7":  {"id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0", "extract": anthropic_extract},
    "claude4.1":  {"id": "us.anthropic.claude-opus-4-1-20250805-v1:0",   "extract": anthropic_extract},
    "claude4.5":  {"id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "extract": anthropic_extract},
    "claude4.5h": {"id": "us.anthropic.claude-haiku-4-5-20251001-v1:0",  "extract": anthropic_extract},
    "claude4.5o": {"id": "global.anthropic.claude-opus-4-5-20251101-v1:0", "extract": anthropic_extract},
    
    # Nova models
    "novamicro": {"id": "us.amazon.nova-micro-v1:0",   "extract": converse_extract},
    "novap":     {"id": "us.amazon.nova-premier-v1:0", "extract": converse_extract},
    "novapro":   {"id": "us.amazon.nova-pro-v1:0",    "extract": converse_extract},
    
    # Qwen models
    "qwen32b":  {"id": "qwen.qwen3-32b-v1:0",            "extract": openai_extract},
    "qwen235b": {"id": "qwen.qwen3-235b-a22b-2507-v1:0", "extract": openai_extract},
    
    # OpenAI model
    "oss":   {"id": "openai.gpt-oss-120b-1:0", "extract": openai_extract},
    "oss20": {"id": "openai.gpt-oss-20b-1:0",  "extract": openai_extract},
    
    # DeepSeek
    "deepseek": {"id": "us.deepseek.r1-v1:0", "extract": openai_extract},
}


class BedrockLLM:
    """Bedrock LLM client with model registry and retry logic."""
    
    def __init__(
        self,
        model: str = "qwen235b",
        region: str = "us-east-1",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        max_retries: int = 5,
        profile: Optional[str] = None,
    ):
        """Initialize Bedrock client.
        
        Args:
            model: Model short name (must be in MODELS registry)
            region: AWS region
            temperature: Sampling temperature (0.0 = deterministic)
            top_p: Top-p (nucleus) sampling parameter
            max_tokens: Maximum tokens to generate
            max_retries: Number of retries on throttling
            profile: AWS profile name (optional)
        """
        if model not in MODELS:
            raise ValueError(f"Unknown model '{model}'. Available: {list(MODELS.keys())}")

        self.model_name = model
        self.model_config = MODELS[model]
        self.model_id = self.model_config["id"]
        self.extract_fn: Callable = self.model_config["extract"]

        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        config = Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
            read_timeout=120,
            connect_timeout=30,
        )
        
        session_kwargs = {}
        if profile:
            session_kwargs["profile_name"] = profile
        
        session = boto3.Session(**session_kwargs)
        self.client = session.client(
            "bedrock-runtime",
            region_name=region,
            config=config,
        )
    
    def _format_request(self, system: str, prompt: str) -> dict:
        """Format request body using unified OpenAI-style format."""
        return {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
    
    def _parse_response(self, response: dict) -> str:
        """Parse response using model's registered extractor."""
        body = json.loads(response["body"].read())
        return self.extract_fn(body)
    
    def generate(self, system: str, prompt: str) -> str:
        """Generate a response from the LLM.
        
        Args:
            system: System prompt
            prompt: User prompt
            
        Returns:
            Generated text response
        """
        request_body = self._format_request(system, prompt)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body),
                    contentType="application/json",
                    accept="application/json",
                )
                return self._parse_response(response)
                
            except self.client.exceptions.ThrottlingException:
                wait_time = 2 ** attempt
                print(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"Error (attempt {attempt + 1}): {e}")
                time.sleep(2)
        
        raise RuntimeError(f"Failed after {self.max_retries} attempts")
    
    def __repr__(self) -> str:
        return f"BedrockLLM(model={self.model_name}, id={self.model_id})"
