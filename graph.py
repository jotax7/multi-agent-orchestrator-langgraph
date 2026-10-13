"""
graph.py — Construcción del StateGraph.

Topología jerárquica:

    START -> supervisor -(cond)-> researcher | analyst | validator
             researcher -> supervisor
             analyst    -> supervisor
             validator  -(cond)-> synthesizer | supervisor
             synthesizer -> END

Todos los nodos son inyectables (parámetro `nodes`) para poder testear el
flujo SIN API key ni red (ver tests/).
"""
from typing import Callable, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from state import OrchestratorState
from supervisor import make_supervisor, user_request
from validator import make_synthesizer, make_validator, to_text


# ---------- Nodos de especialistas (con aislamiento de contexto) ----------

def make_specialist_node(agent, name: str, context_builder: Callable[[OrchestratorState], str]):
    """
    Envuelve un agente ReAct como nodo del grafo.
    El especialista NO recibe el historial completo: solo lo que arma `context_builder`
    (instrucción puntual + el contexto mínimo necesario) -> evita contaminación de contexto.
    """
    def node(state: OrchestratorState) -> dict:
        task = context_builder(state)
        result = agent.invoke({"messages": [HumanMessage(content=task)]})
        content = to_text(result["messages"][-1].content)
        return {
            # reducer operator.add: se SUMA a la lista, no pisa lo anterior
            "contributions": [{"agent": name, "step": state.get("step_count", 0), "content": content}],
            "messages": [AIMessage(content=content, name=name)],
        }
    return node


def researcher_context(state: OrchestratorState) -> str:
    """El investigador ve: la orden del supervisor + el feedback del validador (si hay)."""
    msg = state.get("instruction") or user_request(state)
    fb = state.get("validation_feedback")
    return msg + (f"\n\nCorrección pedida por el validador: {fb}" if fb else "")


def analyst_context(state: OrchestratorState) -> str:
    """El analista ve: la orden del supervisor + SOLO los aportes del investigador."""
    data = "\n\n".join(c["content"] for c in state.get("contributions", []) if c["agent"] == "researcher")
    fb = state.get("validation_feedback")
    return (f"{state.get('instruction') or 'Analizá los datos.'}\n\n"
            f"DATOS INVESTIGADOS:\n{data}" + (f"\n\nCorrección pedida por el validador: {fb}" if fb else ""))


# ---------- Aristas condicionales (Literal = destinos válidos) ----------

def route_after_supervisor(state: OrchestratorState) -> Literal["researcher", "analyst", "validator"]:
    nxt = state["next_agent"]
    return "validator" if nxt == "FINISH" else nxt  # FINISH => validación formal antes del END


def route_after_validator(state: OrchestratorState) -> Literal["synthesizer", "supervisor"]:
    return "synthesizer" if state.get("task_completed") else "supervisor"


# ---------- Armado ----------

def build_graph(llm=None, nodes: dict | None = None):
    """
    `llm`: modelo de chat (si es None se crea con llm.get_llm()).
    `nodes`: overrides opcionales {"supervisor":..., "researcher":..., "analyst":..., "validator":..., "synthesizer":...}
    """
    nodes = dict(nodes or {})
    if llm is None and len(nodes) < 5:
        from llm import get_llm
        llm = get_llm()

    # Los imports de agentes son perezosos para que los tests no necesiten LLM real
    if "researcher" not in nodes:
        from agents.research_agent import build_research_agent
        nodes["researcher"] = make_specialist_node(build_research_agent(llm), "researcher", researcher_context)
    if "analyst" not in nodes:
        from agents.analyst_agent import build_analyst_agent
        nodes["analyst"] = make_specialist_node(build_analyst_agent(llm), "analyst", analyst_context)
    nodes.setdefault("supervisor", make_supervisor(llm) if llm else None)
    nodes.setdefault("validator", make_validator(llm) if llm else None)
    nodes.setdefault("synthesizer", make_synthesizer(llm) if llm else None)

    g = StateGraph(OrchestratorState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_after_supervisor)
    g.add_edge("researcher", "supervisor")   # los especialistas siempre vuelven al supervisor
    g.add_edge("analyst", "supervisor")
    g.add_conditional_edges("validator", route_after_validator)
    g.add_edge("synthesizer", END)
    return g.compile()


def initial_state(question: str, max_steps: int = 8) -> dict:
    """Estado inicial con los contadores en cero."""
    return {
        "messages": [HumanMessage(content=question)],
        "contributions": [], "step_count": 0, "max_steps": max_steps,
        "revisions": 0, "validation_feedback": "", "task_completed": False, "final_answer": "",
    }
