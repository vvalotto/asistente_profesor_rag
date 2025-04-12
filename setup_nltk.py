# setup_nltk.py
import ssl
import nltk
import sys


def download_nltk_resources():
    """Descarga los recursos de NLTK necesarios desactivando la verificación SSL."""
    print("Iniciando descarga de recursos NLTK...")

    # Desactivar verificación SSL para solucionar problemas de certificados
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        print("Warning: No se pudo desactivar la verificación SSL")
        pass
    else:
        print("Info: Verificación SSL desactivada para la descarga")
        ssl._create_default_https_context = _create_unverified_https_context

    # Lista de recursos a descargar
    resources = [
        ('punkt', 'Tokenizador de oraciones'),
        ('stopwords', 'Palabras vacías'),
        ('punkt_tab', 'Tokenizador de oraciones para español')
    ]

    # Descargar cada recurso
    success = True
    for resource, description in resources:
        print(f"\nDescargando {description} ({resource})...")
        try:
            nltk.download(resource, quiet=False)
            print(f"✓ Recurso '{resource}' descargado exitosamente")
        except Exception as e:
            print(f"✗ Error al descargar '{resource}': {str(e)}")
            success = False

    # Verificar que los recursos se descargaron correctamente
    if success:
        print("\n✓ Todos los recursos de NLTK han sido descargados correctamente.")
        return 0
    else:
        print("\n✗ Algunos recursos no pudieron ser descargados. Revisa los errores anteriores.")
        return 1


if __name__ == "__main__":
    sys.exit(download_nltk_resources())