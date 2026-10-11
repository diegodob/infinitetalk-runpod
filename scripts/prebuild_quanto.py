"""Compila por adelantado los nucleos CUDA de optimum-quanto.

Sin esto, el primer arranque de InfiniteTalk dentro de un pod compila la extension
'quanto_cuda' al vuelo (unos 5 minutos) y con el gcc 13 de la base ese build se cae con
un error interno del compilador:

    RuntimeError: Error building extension 'quanto_cuda'  ->  ninja: build stopped

Con g++-12 compila bien (nvcc lo toma de CUDAHOSTCXX). Aca queda hecho una sola vez,
para las arquitecturas de las placas mas comunes (A100 = 8.0, A4000/3090 = 8.6,
4090/L40S = 8.9). Si la placa es otra, el pod lo recompila solo la primera vez.
"""

from optimum.quanto.library.extensions.cuda import load_cuda_extension

if __name__ == "__main__":
    load_cuda_extension()
    print("quanto_cuda precompilado ok")
