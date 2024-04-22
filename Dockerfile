# Usar una imagen base oficial de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar solo el archivo de dependencias necesario
COPY requirements.txt .

# Instalar dependencias utilizando pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código fuente del proyecto al directorio de trabajo
COPY . /app

# Comando para ejecutar cuando se inicie el contenedor
CMD ["python", "main.py"]
