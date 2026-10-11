#!/usr/bin/env python3
"""Parches de compatibilidad para InfiniteTalk.

El repositorio original asume Python 3.10, diffusers viejo y la libreria
flash-attn instalada. Corriendo en Python 3.12 con diffusers 0.31 y sin
flash-attn falla en cuatro lugares. Este script los arregla y ademas deja
un stub de xfuser (el paralelismo multi-GPU, que con una sola placa no se usa).

Es idempotente: si ya esta parcheado, no hace nada.
Uso: apply_patches.py [/opt/italk]
"""
import pathlib
import subprocess
import sys


def edit(path, old, new, label, marca):
    p = pathlib.Path(path)
    if not p.exists():
        print(f"[!!] {label}: no existe {path}")
        return
    s = p.read_text()
    if marca in s and old not in s:
        print(f"[ok] {label}: ya estaba")
        return
    if old not in s:
        print(f"[!!] {label}: no encontre el texto esperado en {path}")
        return
    p.write_text(s.replace(old, new, 1))
    print(f"[ok] {label}: parcheado")


repo = sys.argv[1] if len(sys.argv) > 1 else "/opt/italk"

# 1) ArgSpec: eliminado de inspect en Python 3.11+
edit(
    f"{repo}/wan/multitalk.py",
    "from inspect import ArgSpec",
    "from collections import namedtuple\n"
    "ArgSpec = namedtuple('ArgSpec', ['args', 'varargs', 'keywords', 'defaults'])",
    "ArgSpec (python 3.12)",
    "ArgSpec = namedtuple",
)

# 2) no_init_weights / ContextManagers: diffusers >= 0.31 ya no los exporta.
edit(
    f"{repo}/wan/multitalk.py",
    "from diffusers.models.modeling_utils import no_init_weights, ContextManagers",
    '''try:
    from diffusers.models.modeling_utils import no_init_weights, ContextManagers
except ImportError:
    from contextlib import contextmanager, ExitStack

    class ContextManagers:
        def __init__(self, managers):
            self.managers = managers

        def __enter__(self):
            self.stack = ExitStack()
            for m in self.managers:
                self.stack.enter_context(m)
            return self

        def __exit__(self, *exc):
            return self.stack.__exit__(*exc)

    @contextmanager
    def no_init_weights(*a, **k):
        yield''',
    "no_init_weights/ContextManagers (diffusers 0.31)",
    "class ContextManagers",
)

# 3) kokoro: solo lo usa el modo texto-a-voz del demo; sin la libreria no importa el CLI.
edit(
    f"{repo}/generate_infinitetalk.py",
    "from kokoro import KPipeline",
    "try:\n    from kokoro import KPipeline\nexcept Exception:\n    KPipeline = None",
    "import de kokoro",
    "    KPipeline = None",
)

# 4) flash-attn: si la libreria no esta, usar la atencion nativa de PyTorch.
#    El codigo llama a flash_attention() de forma directa, asi que se redefine al final
#    del modulo para que los demas modulos importen esta version.
p = pathlib.Path(f"{repo}/wan/modules/attention.py")
marca = "def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None,"
if not p.exists():
    print(f"[!!] flash-attn: no existe {p}")
elif marca in p.read_text():
    print("[ok] flash-attn (atencion nativa): ya estaba")
else:
    p.write_text(p.read_text().rstrip() + '''


# Fallback: sin la libreria flash-attn, usamos la atencion nativa de PyTorch.
if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
    def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None,
                        q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
                        dtype=torch.bfloat16, version=None):
        _d = dtype if dtype is not None else q.dtype
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).to(_d), k.transpose(1, 2).to(_d), v.transpose(1, 2).to(_d),
            attn_mask=None, is_causal=causal, dropout_p=dropout_p)
        return out.transpose(1, 2).contiguous()
''')
    print("[ok] flash-attn (atencion nativa): agregado")

# 5) xfuser: se usa solo para repartir el modelo entre varias placas.
sp = subprocess.run(
    [sys.executable, "-c", "import site;print(site.getsitepackages()[0])"],
    capture_output=True, text=True,
).stdout.strip()
base = pathlib.Path(sp) / "xfuser"
(base / "core").mkdir(parents=True, exist_ok=True)
(base / "__init__.py").write_text("__version__ = '0.0.0-stub'\n")
# El __init__ tiene que quedar VACIO: un "from .distributed import *" aca dispara
# la iteracion de __all__, y con el __getattr__ del stub eso rompe el import.
(base / "core" / "__init__.py").write_text(
    "# Vacio a proposito: el stub vive en distributed.py.\n"
)
(base / "core" / "distributed.py").write_text('''"""Stub de xfuser: con una sola placa no hay nada que repartir.

InfiniteTalk importa estas funciones para el modo multi-GPU.
Aca devolvemos los valores de una sola GPU: rango 0, tamano de grupo 1, sin grupo.
"""

rank = 0
world_size = 1
_local_rank = 0


def __getattr__(name):
    # OJO: los nombres internos de Python tienen que fallar con AttributeError.
    # Si devolvemos None para "__all__", 'from ... import *' intenta iterar None
    # y muere con "TypeError: 'NoneType' object is not iterable".
    if name.startswith("__"):
        raise AttributeError(name)
    if "rank" in name:
        return lambda *a, **k: 0
    if "world_size" in name or "degree" in name:
        return lambda *a, **k: 1
    return lambda *a, **k: None
''')
(base / "core" / "long_ctx_attention.py").write_text('''"""Stub de xfuser: la atencion de contexto largo solo se usa con multi-GPU."""


class xFUserLongContextAttention:
    def __init__(self, *a, **k):
        raise RuntimeError(
            "El stub de xfuser no implementa xFUserLongContextAttention: "
            "la libreria real solo hace falta para repartir en varias placas."
        )
''')
print("[ok] stub de xfuser listo")
print("PARCHES_OK")
