"""Terminal de Trading Cuantitativa.

Paquete modular para ingesta de datos multi-fuente, procesamiento cuantitativo,
agregación de señales por confluencia, gestión de riesgo y alertas.

El núcleo computacional depende solo de numpy/pandas/scipy/ta. Las capas de red,
NLP y ML importan sus dependencias de forma perezosa.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
