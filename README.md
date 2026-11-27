# CocaTrace Edge AI — Sistema de Trazabilidad y Monitoreo Inteligente

Sistema inteligente de monitoreo y trazabilidad que integra inteligencia artificial embebida (TinyML), computación en la nube con AWS y agentes conversacionales mediante MCP (Model Context Protocol).

Utiliza dos nodos físicos independientes basados en microcontroladores ESP32 que ejecutan tres modelos neuronales locales sin depender de conexión constante a internet: un nodo analiza condiciones climáticas (DHT11) y movimiento logístico (MPU6050) mediante modelos compilados con Edge Impulse EON, mientras que una ESP32-CAM realiza detección visual de plagas y estado de hojas en tiempo real con el algoritmo FOMO.

Los datos de inferencia viajan de forma segura por MQTT hacia AWS IoT Core, donde reglas SQL filtran anomalías críticas (estrés térmico, baja temperatura o presencia de hongos) para disparar notificaciones instantáneas por correo electrónico mediante Amazon SNS. Paralelamente, la telemetría histórica se almacena en DynamoDB y se visualiza en un panel web interactivo alojado en Amazon S3 con actualización cada 1.5 segundos.

Ambas placas alojan servidores HTTP JSON-RPC 2.0 directamente en su memoria RAM, permitiendo que el usuario consulte el estado actual del cultivo, historiales y alertas mediante lenguaje natural desde VS Code (GitHub Copilot Agent) sin servidores intermedios.

## Integrantes
* Valentina Arciniegas
* Davinson Tulande
* Carlos Daza
* Andres Coral

# Arquitectura 

![arquitectura](docs/arquitectura.png)

1. **Capa Edge (Inferencia Local en Microcontroladores):**

* **Nodo Principal (`ESP32`):** Ejecuta dos modelos neuronales de Edge Impulse simultáneamente:
  
    * *Modelo Climático :* Clasifica temperatura (`optimo`, `baja_temp`, `estres_termico`) a 1 Hz con sensor DHT11.
  
    * *Modelo de Movimiento:* Clasifica estados logísticos (`reposo`, `transporte`, `volteo`) a 50 Hz con acelerómetro/giroscopio MPU6050.
  
    * *Nodo de Visión (`ESP32-CAM AI-Thinker`):* Ejecuta un modelo de detección de objetos **FOMO** para identificar el estado
 de las hojas (`hoja_sana`, `seca`, `hongo`) junto con sus coordenadas espaciales `(x, y)`.

2. **Capa Cloud & Alertas (AWS Serverless):**

* **AWS IoT Core:** Recepción segura mediante MQTT sobre TLS 1.2 (`esp32/values`, `esp32/movimiento`, `esp32/imagen`).
 
* **Reglas IoT + Amazon SNS:** Filtrado SQL en tiempo real para el disparo automático de alertas personalizadas por correo electrónico ante anomalías térmicas, manipulación brusca o detección de plagas.
  
* **Amazon DynamoDB + AWS Lambda + API Gateway**: Persistencia histórica de telemetría y exposición de endpoints REST para consulta unificada.
* **Amazon S3 (Dashboard Web):** Interfaz interactiva con actualización asíncrona cada 1.5 segundos, gráficas dinámicas (Chart.js) y exportación a CSV. 

3. **Ambas placas ESP32 alojan un servidor HTTP JSON-RPC 2.0 (`/mcp`) con búferes circulares en RAM (últimos 10 registros por modelo), permitiendo consultas en lenguaje natural directamente desde **VS Code (GitHub Copilot Agent)** sin intermediarios.**

## 1. Modelos de clasificación 

### 1.1 Humedad- Temperatura

Este modelo neuronal se entrena y compila con **Edge Impulse (EON Compiler)** para operar directamente sobre el microcontrolador **ESP32** (`tflite_learn_1171835_5`), permitiendo detectar anomalías térmicas en tiempo real sin depender de procesamiento en la nube. (sensor DHT11).Las metricas obtenidas son:

![mctemp](docs/mctemp.PNG)

* **Precisión y Pérdida:** El modelo alcanzó un **100.0 % de exactitud (*Accuracy*)** y una **pérdida (*Loss*) de 0.00** sobre el conjunto de validación, demostrando una convergencia óptima sin errores de clasificación.
  
* **Matriz de Confusión:** Las tres clases (`baja_temp`, `estres_termico` y `optimo`) obtuvieron un **F1-Score de 1.00** (100 % de aciertos en la diagonal principal y 0 % de falsos positivos/negativos entre categorías).
  
* **Separabilidad de Características (*Data Explorer*):** En la proyección del espacio de características, los clústeres de cada estado térmico aparecen claramente diferenciados y sin solapamiento, lo que explica el **AUC-ROC de 1.00** y garantiza inferencias determinísticas y estables en el ESP32.

El dataset de entrenamiento y validación se encuentra en `dataset/dataset temp.rar`

### 1.2 Acelerómetro

Este modelo neuronal se entrena y compila con **Edge Impulse** (`Acelerometro_esp32_inferencing`) sobre el nodo principal **ESP32**, permitiendo supervisar la manipulación física y el estado logístico de las bandejas o contenedores del cultivo en tiempo real. (sensor MPU6050), las metricas obtenidas son:

![mcAc](docs/mcAc.PNG)

* **Precisión y Pérdida:** El modelo alcanzó un **100.0 % de exactitud (*Accuracy*)** y una **pérdida (*Loss*) de 0.00** sobre el conjunto de validación, confirmando una convergencia óptima en la clasificación de patrones inerciales.
  
* **Matriz de Confusión:** Las tres clases (`quieto`, `transporte` y `volteo_hojas`) obtuvieron un **F1-Score de 1.00** (100 % de aciertos en la diagonal principal y 0 % de falsos positivos o confusión entre eventos).
  
* **Separabilidad de Características (*Data Explorer*):** En la visualización espectral de muestras clasificadas, los tres clústeres (`quieto`, `transporte` y `volteo_hojas`) se agrupan de forma compacta y completamente aislada entre sí, respaldando el **AUC-ROC de 1.00** y asegurando detecciones confiables en tiempo real sobre el MPU6050.

El dataset de entrenamiento y validación se encuentra en `dataset/dataset acelerometro.rar`

### 1.3 Cámara

Este modelo de detección de objetos fue entrenado en **Edge Impulse** utilizando la arquitectura **FOMO (*Faster Objects, More Objects*)** (`Train-Esp32cam_inferencing`) y desplegado sobre un nodo independiente **ESP32-CAM**, permitiendo inspeccionar visualmente el estado fitosanitario de las hojas del cultivo en tiempo real y localizar espacialmente posibles anomalías.

Para construir el dataset, se desplegó un servidor HTTP local en la ESP32-CAM AI-Thinker que transmitía video en vivo por red local hacia el navegador (http://<IP_ESP32_CAM>), permitiendo ajustar la exposición/resolución del sensor OV2640 y guardar las capturas (Get Still) directamente desde el lente real. la interfaz es:

![imgserver](docs/imgserver.png)

Las metricas que se obtuvieron son:

![mccam](docs/mccam.PNG)

* **Precisión y Métricas Globales :** El modelo cuantizado alcanzó un **F1-Score global de 85.1 %** (`0.85` en clases *non-background*), con una **precisión (Precision) de 0.95** y una **sensibilidad (Recall) de 0.77** sobre el conjunto de validación, minimizando falsas alarmas durante la inferencia en tiempo real en la ESP32-CAM.

*  **Matriz de Confusión:** Las alertas críticas **`HONGO`** (`100 %` de aciertos) y **`SECA`** (`84.6 %` de aciertos) alcanzaron un **F1-Score de 0.92**, presenta errores al momento de distinguir la hoja `FRESCA` con el background, esto  no genera falsas alarmas ni disparos erróneos en AWS SNS, porque nunca se confunde con HONGO ni con SECA — ambas columnas tienen 0 % de error cruzado.
  
* **Desempeño por Clase en Borde:** La alta precisión en las anomalías fitosanitarias (`HONGO` y `SECA`) asegura detecciones confiables en tiempo real sobre el sensor OV2640 sin disparar falsos positivos cuando no hay follaje en el encuadre.

El dataset de entrenamiento y validación se encuentra en `dataset/dataset fotos.rar`

## 2. Configuración ESP32 – ESP32CAM

### 2.1 ESP32

Este nodo ejecuta de manera concurrente dos modelos de aprendizaje automático compilados desde **Edge Impulse** y gestiona tanto la telemetría segura hacia la nube como el servidor local MCP.

* **Sensores y Conexiones:**

* **DHT11 (Clima):** Conectado al pin digital `GPIO 19` para muestreo de temperatura (°C) y humedad relativa (%) a **1 Hz**.

* **MPU6050 (IMU de 6 ejes):** Conectado por bus I2C (`SDA: GPIO 21`, `SCL: GPIO 22`), configurado con rango de aceleración y giroscopio  a `21 Hz` con muestreo a **50 Hz**.

Al arrancar el microcontrolador, se ejecutan cuatro rutinas críticas en secuencia: 

1. **Sensores:** Inicia el sensor digital **DHT11** en `GPIO 19` y el bus I2C (`SDA: GPIO 21`, `SCL: GPIO 22`) para comunicarse con el **MPU6050**, configurando su rango dinámico para aceleración y giroscopio, activando el filtro digital pasa-bajos (DLPF) a `21 Hz` para mitigar vibraciones mecánicas de alta frecuencia. 

2. **Asignación de Tensores EON (`iniciarModeloClima`):** Inicializa el grafo neuronal compilado (`tflite_learn_1171835_5_init`) reservando memoria dinámica en heap (`ei_calloc`) y enlazando los punteros directos a los tensores de entrada (`input_clima`) y salida (`output_clima`).

3. **Red Wi-Fi y Reloj NTP (`conectarWiFi`):** Establece conexión en modo estación (`WIFI_STA`) con potencia de transmisión ajustada a `11 dBm` para evitar picos de corriente que afecten al DHT11, y sincroniza el reloj interno mediante servidores NTP (`pool.ntp.org`, `time.nist.gov`) en zona horaria **UTC-5** para generar marcas de tiempo ISO-8601 (`YYYY-MM-DDTHH:MM:SS-05:00`).

4. **Servidores de Comunicación (`iniciarServidorMCP` y `conectarAWS`):** Levanta el servidor HTTP en el puerto `80` (endpoint `/mcp`) y realiza el *handshake* TLS 1.2 contra el endpoint de **AWS IoT Core** (puerto `8883`) utilizando certificados X.509 (`AWS_CERT_CA`, `AWS_CERT_CRT`, `AWS_CERT_PRIVATE`).

#### Bucle Principal Asíncrono (`loop`) 

Para evitar que operaciones de red o inferencia bloqueen el muestreo de alta frecuencia, el `loop()` evita el uso de `delay()` y se divide en dos lazos temporizados mediante `millis()`:

* A. Pipeline de Movimiento Inercial (50 Hz — `INTERVALO_MOV_MS = 20 ms`)
  
    * **Adquisición:** Cada 20 milisegundos lee los 6 ejes del MPU6050 (a_x, a_y, a_z, g_x, g_y, g_z) y los apila secuencialmente en `buffer_mov`.

    * ***Inferencia:** Al completar una ventana deslizante de **2 segundos** (`EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE` = 600 floats), invoca `run_classifier()` a través del callback `get_data_mov`.
  
    * **Publicación:** Extrae la clase ganadora (`quieto`, `transporte` o `volteo_hojas`), redondea la probabilidad a 3 decimales, actualiza la memoria local y serializa un payload JSON de 512 bytes hacia el tópico `esp32/movimiento`.
 
El Json que se publica en AWS es:

```bash
json
 { "timestamp": 1793859932,
"id_sensor": "ESP32", "temperatura": 23.8, "humedad": 50.8,
"lote": "LOTE-2026-01", "proceso": "SECADO" }
"model_mov": "quieto",
"confianza": 0.962,
"ax": 0.12, 
"ay": -0.05, 
"az": 9.78,
"gx": 0.01,
"gy": -0.02, 
"gz": 0.00}
```
  
* B. Pipeline Climático (1 Hz — `INTERVALO_CLIMA_MS = 1000 ms`)

    * **Adquisición y Preprocesamiento:** Cada segundo lee temperatura ($^\circ\text{C}$) y humedad ($\%$) del DHT11. Si la lectura es válida (`!isnan(t)`), multiplica la temperatura por el factor de normalización `ESCALA_TEMP = 0.025f` y almacena 3 muestras consecutivas en `buffer_clima`.
      
    * **Inferencia EON Agnostic-Type:** Al completar 3 segundos, copia los valores al tensor de entrada soportando automáticamente tanto modelos en punto flotante (`kTfLiteFloat32`) como modelos cuantizados de 8 bits (`kTfLiteInt8`, aplicando escala y *zero-point*), ejecuta `tflite_learn_1171835_5_invoke()` y obtiene la predicción (`baja_temp`, `estres_termico` u `optimo`) para publicarla en `esp32/values`.

El Json que se publica en AWS es:

```bash
json
 {
"timestamp": 1793859932,
"id_sensor": "ESP32", "temperatura": 23.8, "humedad": 50.8,
"lote": "LOTE-2026-01", "proceso": "SECADO" }
"model_temp": "optimo",
"confianza": 0.985}
```

## Dashboard

El Monitor Edge IA - Cocatrace es una interfaz web de monitoreo en tiempo real diseñada para visualizar y analizar las condiciones de un cultivo durante su proceso de germinación y desarrollo. El sistema integra información proveniente de diferentes dispositivos IoT y modelos de inteligencia artificial (IA), permitiendo supervisar tres aspectos fundamentales:

* Visión artificial (FOMO - ESP32-CAM): permite visualizar las clasificaciones sobre el estado de las hojas.
* Monitoreo de microclima (DHT11): permite consultar las condiciones ambientales del cultivo, principalmente temperatura y humedad relativa.
* Monitoreo de movimiento (MPU6050): permite analizar la actividad, aceleración y giros registrados por el sensor inercial.

La información es consultada mediante una API desplegada en AWS API Gateway, permitiendo actualizar automáticamente los indicadores, gráficas e historiales de registros.

Esta herramienta visual facilita:

* Consultar el estado actual de las hojas mediante visión artificial.
* Identificar condiciones ambientales que puedan representar riesgos para el cultivo.
* Supervisar movimientos, vibraciones y actividad física del sistema.
* Visualizar el comportamiento histórico de las variables mediante gráficas.
* Consultar los registros históricos de cada dispositivo.
* Exportar la información recopilada en archivos CSV.
* Generar alertas visuales cuando se identifican condiciones de riesgo relacionadas con hongos.

El panel principal está dividido en tres tarjetas que presentan la información brindada por los modelos de IA.

### Visión Artificial (FOMO - Cámara)

Muestra los resultados obtenidos mediante el modelo FOMO (Faster Objects, More Objects) de visión artificial, donde se muestra: clasificación del estado de la hoja, confianza de la predicción del modelo, coordenadas X e Y asociadas a la detección y los indicadores visuales mediante iconos y colores según el estado identificado.

### Clima (Red EON - DHT11)

Presenta las condiciones ambientales registradas por el sensor DHT11, donde se muestra: temperatura en grados Celsius (°C), humedad relativa (% HR), clasificación del estado ambiental realizada por el modelo de IA, nivel de confianza de la predicción, proceso actual del cultivo.

### Manejo / Actividad (IMU - MPU6050)

Presenta los datos relacionados con el movimiento del sistema, donde se muestra: clasificación de la actividad mediante IA, confianza del modelo, valores de aceleración en los ejes X, Y y Z, valores de velocidad angular en los ejes X, Y y Z, magnitud resultante de aceleración y giro.

Además el monitor incorpora dos gráficas dinámicas que son:

### Gráfica de temperatura y humedad

Permite observar el comportamiento histórico de Temperatura (°C) y Humedad relativa (%).

### Gráfica de acelerómetro y giroscopio

Permite visualizar la evolución de la Aceleración en el eje X (m/s²) y la Velocidad angular en el eje X (°/s).

Las gráficas se actualizan automáticamente con los nuevos datos recibidos desde la API, sin necesidad de recargar la página.

### Historial de registros

La interfaz cuenta con una tabla dinámica que permite consultar los registros históricos de cada subsistema mediante tres pestañas:

* Cámara (FOMO): Fecha, cámara, lote, tipo de hoja, confianza y coordenadas.
* Clima (DHT11):	Fecha, lote, proceso, estado climático, temperatura, humedad y confianza.
* Movimiento (MPU6050):	Fecha, lote, proceso, actividad detectada, confianza, aceleración y giro.

Cada pestaña actualiza dinámicamente los encabezados y el contenido de la tabla según el tipo de información seleccionado. Los estados se representan mediante etiquetas de colores para facilitar su interpretación.

### Exportación de información CSV

El botón Descargar CSV permite exportar los registros históricos del subsistema seleccionado, teniendo presente las siguientes características:

* Consulta hasta 100 registros desde la API.
* Genera un archivo CSV con los encabezados correspondientes.
* Convierte las fechas a un formato legible.
* Genera automáticamente el nombre del archivo según el subsistema y la fecha de descarga, como por ejemplo "registros_camara_2026-11-26.csv"

Por último, el Monitor Edge IA - Cocatrace proporciona una visualización centralizada de la información generada por los diferentes dispositivos IoT y modelos de IA. Su implementación permite integrar visión artificial, monitoreo ambiental y análisis de movimiento en una única interfaz web, facilitando la supervisión del cultivo, la identificación de posibles condiciones de riesgo y la consulta de información histórica para el análisis del comportamiento del sistema.

![monitor](docs/monitor.png)
