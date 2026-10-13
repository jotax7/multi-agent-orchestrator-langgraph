"""
llm.py — Fábrica de modelos de chat.

El grafo NO depende de un proveedor: todo se pide a `get_llm()`.
Se cambia de proveedor con la variable de entorno LLM_PROVIDER, sin tocar código:

    LLM_PROVIDER=gemini     (default)  -> GOOGLE_API_KEY
    LLM_PROVIDER=groq                  -> GROQ_API_KEY
    LLM_PROVIDER=openai                -> OPENAI_API_KEY
    LLM_PROVIDER=anthropic             -> ANTHROPIC_API_KEY

Opcional: LLM_MODEL para forzar un modelo puntual.
"""
import os

# Modelo por defecto de cada proveedor (editables por LLM_MODEL)
DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5",
}


def get_llm(temperature: float = 0.0):
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = os.getenv("LLM_MODEL", DEFAULT_MODELS.get(provider, ""))

    # Los imports son perezosos: solo necesitás instalar el paquete del proveedor que uses.
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature)

    raise ValueError(f"LLM_PROVIDER desconocido: {provider!r}")
