import os
import sys

# Permite `import state`, `import graph`, etc. desde tests/ sin instalar el paquete
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
