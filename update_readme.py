"""Actualiza la primer línea del README poniendo automáticamente
el título y la versión. Lo saca de setup.cfg o de pyproject.toml
"""

import configparser

def update_readme():
    # Leer la configuración del proyecto
    config = configparser.ConfigParser()
    config.read('setup.cfg')

    project_name = config['metadata']['name']
    project_version = config['metadata']['version']

    # Leer el contenido actual de README.md y actualizar
    with open('README.md', 'r') as file:
        readme_contents = file.readlines()

    # Asumiendo que el título y versión están en las primeras líneas del README
    readme_contents[0] = f"# {project_name} v{project_version}\n"
    print(readme_contents[1])
    print(readme_contents[2])
    print(readme_contents[3])
    # Escribir el contenido actualizado de nuevo a README.md
    with open('README.md', 'w') as file:
        file.writelines(readme_contents)

if __name__ == "__main__":
    update_readme()