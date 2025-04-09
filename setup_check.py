import os
import sys
from dotenv import load_dotenv


def check_environment():
    """Verifica que el entorno esté configurado correctamente."""
    print("Verificando entorno de desarrollo...")

    # Verificar Python
    python_version = sys.version_info
    print(f"- Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
        print("  ❌ Se requiere Python 3.9+")
    else:
        print("  ✅ Versión de Python correcta")

    # Verificar variables de entorno
    load_dotenv()
    required_env_vars = ["OPENAI_API_KEY"]
    for var in required_env_vars:
        if os.getenv(var):
            print(f"- {var}: ✅ Configurado")
        else:
            print(f"- {var}: ❌ No configurado")

    # Verificar dependencias principales
    libraries = [
        "llama-index", "langchain", "openai", "chromadb",
        "fastapi", "streamlit", "pandas", "ragas"
    ]

    for lib in libraries:
        try:
            __import__(lib)
            print(f"- {lib}: ✅ Instalado")
        except ImportError:
            print(f"- {lib}: ❌ No instalado")


if __name__ == "__main__":
    check_environment()
    print("\nPara más información sobre la configuración, consulta el README.md")