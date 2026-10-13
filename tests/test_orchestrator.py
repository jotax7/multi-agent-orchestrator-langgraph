"""
Tests OFFLINE: no necesitan API key ni red.

- Las tools se prueban directo (son funciones determinísticas).
- El grafo se prueba con un FakeLLM guionado: ejercita el supervisor, el validador
  y las aristas REALES; solo las respuestas del "cerebro" están pre-armadas.
"""
import json

from langchain_core.messages import AIMessage

from agents.analyst_agent import calculate, sentiment_analysis, validate_schema
from agents.research_agent import search_knowledge_base
from graph import build_graph, initial_state
from supervisor import RouteDecision
from validator import ValidationResult


# ---------------- Doble de prueba del LLM ----------------

class _Structured:
    """Imita llm.with_structured_output(Schema): devuelve el próximo valor del guion."""
    def __init__(self, fake, schema):
        self.fake, self.schema = fake, schema

    def invoke(self, _messages):
        script = self.fake.route if self.schema is RouteDecision else self.fake.validation
        item = script.pop(0) if len(script) > 1 else script[0]  # el último se repite
        return item


class FakeLLM:
    def __init__(self, route, validation):
        self.route, self.validation = list(route), list(validation)

    def with_structured_output(self, schema):
        return _Structured(self, schema)

    def invoke(self, _messages):  # lo usa el sintetizador
        return AIMessage(content="RESPUESTA FINAL (fake)")


def route(next_, instr="ok"):
    return RouteDecision(next=next_, instruction=instr, reasoning="test")


def verdict(approved, feedback=""):
    return ValidationResult(approved=approved, feedback=feedback)


# Especialistas de mentira (devuelven un aporte fijo, sin LLM)
def fake_node(name, text):
    def node(state):
        return {"contributions": [{"agent": name, "step": state["step_count"], "content": text}]}
    return node


def run(llm, max_steps=8):
    nodes = {"researcher": fake_node("researcher", "DATOS"), "analyst": fake_node("analyst", "ANALISIS")}
    graph = build_graph(llm=llm, nodes=nodes)
    return graph.invoke(initial_state("consulta de prueba", max_steps=max_steps))


# ---------------- Tools ----------------

def test_kb_search_finds_reviews():
    out = json.loads(search_knowledge_base.invoke({"query": "reseñas zenbuds"}))
    assert len(out) >= 5 and all("zenbuds" in d["topic"] for d in out)


def test_sentiment_and_calculate_and_schema():
    s = json.loads(sentiment_analysis.invoke({"texts": ["Excelente, lo recomiendo", "Terrible, defectuoso", "ok"]}))
    assert s["positivo"] == 1 and s["negativo"] == 1 and s["neutro"] == 1
    assert calculate.invoke({"expression": "(120 + 105 + 90) / 3"}) == "105.0"
    assert calculate.invoke({"expression": "__import__('os')"}).startswith("Error")  # no ejecuta código
    assert validate_schema.invoke({"json_str": '{"a": 1}', "required_keys": ["a", "b"]}).startswith("Faltan")


# ---------------- Flujo del grafo ----------------

def test_happy_path_delegation_order():
    """researcher -> analyst -> FINISH -> validator OK -> synthesizer."""
    llm = FakeLLM(route=[route("researcher"), route("analyst"), route("FINISH")], validation=[verdict(True)])
    state = run(llm)
    assert [c["agent"] for c in state["contributions"]] == ["researcher", "analyst"]
    assert state["task_completed"] and state["final_answer"] == "RESPUESTA FINAL (fake)"


def test_guard_blocks_early_finish():
    """Si el LLM dice FINISH sin datos, la guarda lo manda primero al investigador."""
    llm = FakeLLM(route=[route("FINISH"), route("analyst"), route("FINISH")], validation=[verdict(True)])
    state = run(llm)
    assert state["contributions"][0]["agent"] == "researcher"


def test_validator_rejection_triggers_refinement():
    """El validador rechaza una vez -> vuelve al supervisor -> el analista refina -> aprueba."""
    llm = FakeLLM(
        route=[route("researcher"), route("analyst"), route("FINISH"), route("analyst", "corregí"), route("FINISH")],
        validation=[verdict(False, "falta el porcentaje"), verdict(True)],
    )
    state = run(llm)
    assert state["revisions"] == 1
    assert [c["agent"] for c in state["contributions"]].count("analyst") == 2


def test_no_infinite_loop_step_limit():
    """Un supervisor que siempre pide 'researcher' igual termina por max_steps."""
    llm = FakeLLM(route=[route("researcher")], validation=[verdict(True)])
    state = run(llm, max_steps=3)
    assert state["step_count"] <= 4          # 3 pasos + el que fuerza FINISH
    assert state["final_answer"]             # llegó al END


def test_forced_approval_after_max_revisions():
    """Un validador que nunca aprueba no genera bucle: tras MAX_REVISIONS se aprueba forzado."""
    llm = FakeLLM(route=[route("researcher"), route("analyst"), route("FINISH")],
                  validation=[verdict(False, "nunca conforme")])
    state = run(llm, max_steps=30)
    assert state["task_completed"] and "FORZADO" in state["validation_feedback"]
