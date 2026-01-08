"""
Base Agent class for the Multi-Agent Blog Writing System.
Provides common functionality for all specialized agents.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import os

from config import Config, LLMProvider
from models import AgentRole, AgentMessage


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(
        self,
        role: AgentRole,
        system_prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ):
        self.role = role
        self.system_prompt = system_prompt
        self.temperature = temperature or Config.LLM_TEMPERATURE
        self.max_tokens = max_tokens or Config.LLM_MAX_TOKENS
        self.conversation_history: List[Dict[str, str]] = []
        self._client = None
    
    @property
    def client(self):
        """Lazy-load the LLM client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    def _create_client(self):
        """Create the LLM client based on configuration."""
        api_key = Config.get_api_key()
        
        if Config.LLM_PROVIDER == LLMProvider.GOOGLE:
            from google import genai
            return genai.Client(api_key=api_key)
        elif Config.LLM_PROVIDER == LLMProvider.OPENAI:
            from openai import OpenAI
            return OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {Config.LLM_PROVIDER}")
    
    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt to send
            context: Optional additional context
        
        Returns:
            The generated response text
        """
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\n{prompt}"
        
        if Config.LLM_PROVIDER == LLMProvider.GOOGLE:
            return self._generate_google(full_prompt)
        elif Config.LLM_PROVIDER == LLMProvider.OPENAI:
            return self._generate_openai(full_prompt)
        else:
            raise ValueError(f"Unsupported provider: {Config.LLM_PROVIDER}")
    
    def _generate_google(self, prompt: str) -> str:
        """Generate using Google Generative AI."""
        from google.genai import types
        
        response = self.client.models.generate_content(
            model=Config.LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
        )
        return response.text
    
    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content
    
    def create_message(self, content: str, **metadata) -> AgentMessage:
        """Create a message from this agent."""
        return AgentMessage(
            role=self.role,
            content=content,
            metadata=metadata
        )
    
    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> Any:
        """
        Process input and generate output.
        Must be implemented by subclasses.
        """
        pass
    
    def reset(self):
        """Reset the agent's conversation history."""
        self.conversation_history = []
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(role={self.role.value})"
