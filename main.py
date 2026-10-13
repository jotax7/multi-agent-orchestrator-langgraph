"""
main.py — Demo del orquestador.

    python main.py                      # corre la consulta de ejemplo
    python main.py "tu pregunta"        # consulta propia
    python main.py --diagram            # solo genera el diagrama Mermaid (graph.mmd / graph.png)

Imprime la TRAZA de delegación: quién intervino en cada paso.
"""
import sys

from dotenv import load_dotenv

from graph import build_graph, initial_state

DEFAULT_QUESTION = (
    "Buscá las reseñas del producto Zenbuds, analizá el sentimiento y decime qué porcentaje es positivo. "
    "Además calculá el precio promedio entre tienda oficial, marketplace y mayorista, "
    "y el precio de tienda oficial con el descuento promocional aplicado."
)


def save_diagram(graph) -> None:
    """Guarda el diagrama del grafo (Mermaid texto siempre; PNG si hay red disponible)."""
    drawable = graph.get_graph()
    with open("graph.mmd", "w", encoding="utf-8") as f:
        f.write(drawable.draw_mermaid())
    try:
        with open("graph.png", "wb") as f:
            f.write(drawable.draw_mermaid_png())
    except Exception as e:  # draw_mermaid_png usa un servicio web; si falla alcanza con el .mmd
        print(f"(PNG no generado: {e}) -> usá graph.mmd")


def main() -> None:
    load_dotenv()
    args = sys.argv[1:]
    graph = build_graph()

    if "--diagram" in args:
        save_diagram(graph)
        return

    question = args[0] if args else DEFAULT_QUESTION
    print(f"\n🧑 Consulta: {question}\n")

    # stream_mode="updates" -> vemos cada nodo apenas termina (esto es la traza de delegación)
    final = None
    for update in graph.stream(initial_state(question), stream_mode="updates"):
        for node, delta in update.items():
            if node == "supervisor":
                print(f"🧭 supervisor -> {delta['next_agent']}  | orden: {delta['instruction'][:90]}")
            elif node in ("researcher", "analyst"):
                c = delta["contributions"][0]
                print(f"🔧 {node} (paso {c['step']}): {c['content'][:160].replace(chr(10), ' ')}...")
            elif node == "validator":
                ok = delta["task_completed"]
                print(f"✅ validator: {'aprobado' if ok else 'RECHAZADO -> ' + delta['validation_feedback']}")
            elif node == "synthesizer":
                final = delta["final_answer"]

    print(f"\n🏁 Respuesta final:\n{final}")


if __name__ == "__main__":
    main()
