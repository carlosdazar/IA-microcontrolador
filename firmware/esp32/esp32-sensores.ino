#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <WebServer.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <time.h>

// Librería Movimiento
#include <Acelerometro_esp32_inferencing.h>

// Libreria Tempertaura
#include "tflite_learn_1171835_5_compiled.h"
#include "credentials.h"

const char* ETIQUETAS_CLIMA[3] = {"baja_temp", "estres_termico", "optimo"};
const float ESCALA_TEMP = 0.025f;


#define DHTPIN 19
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);
Adafruit_MPU6050 mpu;

const char* TOPIC_CLIMA = "esp32/values";
const char* TOPIC_MOV   = "esp32/movimiento";

WiFiClientSecure net;
PubSubClient client(net);

// Servidor Web HTTP en puerto 80 para el protocolo MCP
WebServer mcpServer(80);

// =========================================================================
// ESTRUCTURAS EN MEMORIA (ESTADO ACTUAL + HISTORIAL DE 10 REGISTROS)
// =========================================================================
#define MAX_HISTORIAL 10

struct RegistroClima {
  float temperatura = 0.0;
  float humedad = 0.0;
  String prediccion = "inicializando";
  float confianza = 0.0;
  String timestamp = "";
};

struct RegistroMov {
  String prediccion = "inicializando";
  float confianza = 0.0;
  float ax = 0, ay = 0, az = 0;
  float gx = 0, gy = 0, gz = 0;
  String timestamp = "";
};

RegistroClima estadoClima;
RegistroMov estadoMov;

RegistroClima histClima[MAX_HISTORIAL];
int totalHistClima = 0;

RegistroMov histMov[MAX_HISTORIAL];
int totalHistMov = 0;

void guardarHistorialClima(const RegistroClima& reg) {
  for (int i = MAX_HISTORIAL - 1; i > 0; i--) {
    histClima[i] = histClima[i - 1];
  }
  histClima[0] = reg;
  if (totalHistClima < MAX_HISTORIAL) totalHistClima++;
}

void guardarHistorialMov(const RegistroMov& reg) {
  for (int i = MAX_HISTORIAL - 1; i > 0; i--) {
    histMov[i] = histMov[i - 1];
  }
  histMov[0] = reg;
  if (totalHistMov < MAX_HISTORIAL) totalHistMov++;
}

// --- Buffer Modelo 1: Movimiento (50 Hz -> 2 seg) ---
float buffer_mov[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
size_t idx_mov = 0;

int get_data_mov(size_t offset, size_t length, float *out_ptr) {
  memcpy(out_ptr, buffer_mov + offset, length * sizeof(float));
  return 0;
}

// --- Tensores Modelo 2: Clima EON Compiled (1 Hz -> 3 seg = 3 floats) ---
TfLiteTensor input_clima;
TfLiteTensor output_clima;
bool clima_listo = false;
float buffer_clima[3];
size_t idx_clima = 0;

unsigned long ultimoMuestreoMov = 0;
unsigned long ultimoMuestreoClima = 0;
const unsigned long INTERVALO_MOV_MS = 20;
const unsigned long INTERVALO_CLIMA_MS = 1000;

void iniciarModeloClima() {
  TfLiteStatus status = tflite_learn_1171835_5_init(ei_calloc);
  if (status != kTfLiteOk) {
    Serial.println("Error iniciando modelo EON de clima");
    return;
  }
  tflite_learn_1171835_5_input(0, &input_clima);
  tflite_learn_1171835_5_output(0, &output_clima);
  clima_listo = true;
  Serial.println("Modelo neuronal de Temperatura (EON) listo!");
}

String obtenerTimestampISO() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) return "1970-01-01T00:00:00-05:00";
  char buf[30];
  strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S-05:00", &timeinfo);
  return String(buf);
}

// =========================================================================
// SERVIDOR MCP EMBEBIDO (JSON-RPC 2.0 sobre HTTP POST /mcp)
// =========================================================================
void manejarMCP() {
  mcpServer.sendHeader("Access-Control-Allow-Origin", "*");
  mcpServer.sendHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
  mcpServer.sendHeader("Access-Control-Allow-Headers", "Content-Type, Accept");

  if (mcpServer.method() == HTTP_OPTIONS) {
    mcpServer.send(204);
    return;
  }

  if (mcpServer.method() != HTTP_POST) {
    mcpServer.send(200, "application/json", "{\"status\":\"MCP Server ESP32 activo en /mcp\"}");
    return;
  }

  String body = mcpServer.arg("plain");
  StaticJsonDocument<1024> req;
  DeserializationError err = deserializeJson(req, body);

  if (err) {
    mcpServer.send(400, "application/json", "{\"jsonrpc\":\"2.0\",\"error\":{\"code\":-32700,\"message\":\"Parse error\"}}");
    return;
  }

  const char* method = req["method"] | "";
  JsonVariant id = req["id"];

  if (id.isNull()) {
    mcpServer.send(202, "application/json", "");
    return;
  }

  DynamicJsonDocument res(6144);
  res["jsonrpc"] = "2.0";
  res["id"] = id;

  //  Handshake MCP: initialize
  if (strcmp(method, "initialize") == 0) {
    JsonObject result = res.createNestedObject("result");
    result["protocolVersion"] = "2024-11-05";
    JsonObject caps = result.createNestedObject("capabilities");
    caps.createNestedObject("tools");
    JsonObject serverInfo = result.createNestedObject("serverInfo");
    serverInfo["name"] = "esp32-invernadero-mcp";
    serverInfo["version"] = "1.1.0";
  }
  // Listado de herramientas: tools/list
  else if (strcmp(method, "tools/list") == 0) {
    JsonObject result = res.createNestedObject("result");
    JsonArray tools = result.createNestedArray("tools");

    // Tool 1: Estado actual completo (sensores + predicciones IA + 6 ejes IMU)
    JsonObject t1 = tools.createNestedObject();
    t1["name"] = "obtener_estado_actual";
    t1["description"] = "Obtiene temperatura, humedad, ejes del acelerometro (ax,ay,az) y giroscopio (gx,gy,gz), y las predicciones actuales de ambos modelos Edge AI.";
    JsonObject s1 = t1.createNestedObject("inputSchema");
    s1["type"] = "object";
    s1.createNestedObject("properties");

    // Últimos 10 registros
    JsonObject t2 = tools.createNestedObject();
    t2["name"] = "obtener_ultimos_registros";
    t2["description"] = "Devuelve los ultimos 10 registros almacenados en el ESP32 tanto de clima/temperatura como de acelerometro/movimiento con sus predicciones.";
    JsonObject s2 = t2.createNestedObject("inputSchema");
    s2["type"] = "object";
    s2.createNestedObject("properties");

    // Verificar alertas
    JsonObject t3 = tools.createNestedObject();
    t3["name"] = "verificar_alertas_cultivo";
    t3["description"] = "Evalua si existen alertas criticas de estres termico, baja temperatura o transporte/movimiento brusco en el lote.";
    JsonObject s3 = t3.createNestedObject("inputSchema");
    s3["type"] = "object";
    s3.createNestedObject("properties");
  }
  // 3. Ejecución de herramienta: tools/call
  else if (strcmp(method, "tools/call") == 0) {
    const char* toolName = req["params"]["name"] | "";
    JsonObject result = res.createNestedObject("result");
    JsonArray content = result.createNestedArray("content");
    JsonObject item = content.createNestedObject();
    item["type"] = "text";

    if (strcmp(toolName, "obtener_estado_actual") == 0) {
      StaticJsonDocument<768> info;
      info["dispositivo"] = THINGSNAME;
      info["lote"] = LOTE;
      info["proceso"] = PROCESO;

      JsonObject clima = info.createNestedObject("clima");
      clima["temperatura_c"] = estadoClima.temperatura;
      clima["humedad_pct"] = estadoClima.humedad;
      clima["prediccion_ia"] = estadoClima.prediccion;
      clima["confianza"] = estadoClima.confianza;
      clima["timestamp"] = estadoClima.timestamp;

      JsonObject mov = info.createNestedObject("movimiento_imu");
      mov["prediccion_ia"] = estadoMov.prediccion;
      mov["confianza"] = estadoMov.confianza;
      mov["ax"] = estadoMov.ax;
      mov["ay"] = estadoMov.ay;
      mov["az"] = estadoMov.az;
      mov["gx"] = estadoMov.gx;
      mov["gy"] = estadoMov.gy;
      mov["gz"] = estadoMov.gz;
      mov["timestamp"] = estadoMov.timestamp;

      String textoSalida;
      serializeJsonPretty(info, textoSalida);
      item["text"] = textoSalida;
    }
    else if (strcmp(toolName, "obtener_ultimos_registros") == 0) {
      DynamicJsonDocument histDoc(4096);
      JsonArray arrClima = histDoc.createNestedArray("ultimos_clima");
      for (int i = 0; i < totalHistClima; i++) {
        JsonObject c = arrClima.createNestedObject();
        c["timestamp"] = histClima[i].timestamp;
        c["temp_c"] = histClima[i].temperatura;
        c["hum_pct"] = histClima[i].humedad;
        c["prediccion"] = histClima[i].prediccion;
        c["confianza"] = histClima[i].confianza;
      }

      JsonArray arrMov = histDoc.createNestedArray("ultimos_movimiento");
      for (int i = 0; i < totalHistMov; i++) {
        JsonObject m = arrMov.createNestedObject();
        m["timestamp"] = histMov[i].timestamp;
        m["prediccion"] = histMov[i].prediccion;
        m["confianza"] = histMov[i].confianza;
        m["ax"] = histMov[i].ax;
        m["ay"] = histMov[i].ay;
        m["az"] = histMov[i].az;
        m["gx"] = histMov[i].gx;
        m["gy"] = histMov[i].gy;
        m["gz"] = histMov[i].gz;
      }

      String textoSalida;
      serializeJson(histDoc, textoSalida);
      item["text"] = textoSalida;
    }
    else if (strcmp(toolName, "verificar_alertas_cultivo") == 0) {
      String reporte = "";
      if (estadoClima.prediccion == "estres_termico") {
        reporte += "ALERTA CRITICA: Estres termico detectado (" + String(estadoClima.temperatura, 1) + " C). ";
      } else if (estadoClima.prediccion == "baja_temp") {
        reporte += "ALERTA FRIO: Baja temperatura detectada (" + String(estadoClima.temperatura, 1) + " C). ";
      }
      if (estadoMov.prediccion == "transporte" || estadoMov.prediccion == "volteo") {
        reporte += "ALERTA MOVIMIENTO: Se detecto actividad de tipo '" + estadoMov.prediccion + "'. ";
      }
      if (reporte == "") {
        reporte = "ESTADO NORMAL: Sin alertas criticas. Clima en '" + estadoClima.prediccion + "' (" + String(estadoClima.temperatura, 1) + " C) y contenedor en '" + estadoMov.prediccion + "'.";
      }
      item["text"] = reporte;
    }
    else {
      result["isError"] = true;
      item["text"] = "Herramienta no encontrada en el ESP32.";
    }
  }
  else {
    JsonObject error = res.createNestedObject("error");
    error["code"] = -32601;
    error["message"] = "Metodo no soportado";
  }

  String output;
  serializeJson(res, output);
  mcpServer.send(200, "application/json", output);
}

void iniciarServidorMCP() {
  mcpServer.on("/mcp", manejarMCP);
  mcpServer.begin();
  Serial.print("Servidor MCP listo en: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/mcp");
}

void conectarWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setTxPower(WIFI_POWER_11dBm);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Conectando WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println(" OK");
  configTime(-5 * 3600, 0, "pool.ntp.org", "time.nist.gov");
}

void conectarAWS() {
  net.setCACert(AWS_CERT_CA);
  net.setCertificate(AWS_CERT_CRT);
  net.setPrivateKey(AWS_CERT_PRIVATE);
  client.setServer(AWS_Iot_Endpoint, 8883);
  client.setBufferSize(512);

  while (!client.connected()) {
    Serial.print("Conectando AWS IoT...");
    if (client.connect(THINGSNAME)) {
      Serial.println(" Conectado!");
    } else {
      Serial.print(" Error rc=");
      Serial.print(client.state());
      delay(2000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  dht.begin();
  Wire.begin(21, 22);

  if (!mpu.begin()) {
    Serial.println("Error: No se detecta MPU6050");
    while (1) delay(10);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  iniciarModeloClima();
  conectarWiFi();
  iniciarServidorMCP();
  conectarAWS();
}

void loop() {
  if (!client.connected()) conectarAWS();
  client.loop();
  mcpServer.handleClient();

  unsigned long ahora = millis();

  // ==========================================================
  // ETAPA 1: MODELO DE MOVIMIENTO (50 Hz -> cada 2 seg)
  // ==========================================================
  if (ahora - ultimoMuestreoMov >= INTERVALO_MOV_MS) {
    ultimoMuestreoMov = ahora;
    sensors_event_t a, g, temp_mpu;
    mpu.getEvent(&a, &g, &temp_mpu);

    if (idx_mov + 6 <= EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
      buffer_mov[idx_mov++] = a.acceleration.x;
      buffer_mov[idx_mov++] = a.acceleration.y;
      buffer_mov[idx_mov++] = a.acceleration.z;
      buffer_mov[idx_mov++] = g.gyro.x;
      buffer_mov[idx_mov++] = g.gyro.y;
      buffer_mov[idx_mov++] = g.gyro.z;
    }

    if (idx_mov >= EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
      idx_mov = 0;
      signal_t signal;
      signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
      signal.get_data = &get_data_mov;

      ei_impulse_result_t result = {0};
      if (run_classifier(&signal, &result, false) == EI_IMPULSE_OK) {
        float max_val = -1.0f;
        size_t max_idx = 0;
        for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
          if (result.classification[i].value > max_val) {
            max_val = result.classification[i].value;
            max_idx = i;
          }
        }

        estadoMov.prediccion = ei_classifier_inferencing_categories[max_idx];
        estadoMov.confianza = round(max_val * 1000) / 1000.0;
        estadoMov.ax = a.acceleration.x;
        estadoMov.ay = a.acceleration.y;
        estadoMov.az = a.acceleration.z;
        estadoMov.gx = g.gyro.x;
        estadoMov.gy = g.gyro.y;
        estadoMov.gz = g.gyro.z;
        estadoMov.timestamp = obtenerTimestampISO();
        guardarHistorialMov(estadoMov);

        StaticJsonDocument<512> doc;
        doc["device_id"] = THINGSNAME;
        doc["lote"] = LOTE;
        doc["proceso"] = PROCESO;
        doc["timestamp"] = estadoMov.timestamp;
        doc["model_mov"] = estadoMov.prediccion;
        doc["confianza"] = estadoMov.confianza;
        doc["ax"] = estadoMov.ax;
        doc["ay"] = estadoMov.ay;
        doc["az"] = estadoMov.az;
        doc["gx"] = estadoMov.gx;
        doc["gy"] = estadoMov.gy;
        doc["gz"] = estadoMov.gz;

        char payload[512];
        serializeJson(doc, payload);
        bool ok = client.publish(TOPIC_MOV, payload);
        Serial.printf("[MODEL_MOV] %s (%.2f) -> AWS: %s\n",
                      estadoMov.prediccion.c_str(), max_val, ok ? "OK" : "FALLO");
      }
    }
  }

  // ==========================================================
  // ETAPA 2: MODELO DE TEMPERATURA (1 Hz -> cada 3 seg)
  // ==========================================================
  if (ahora - ultimoMuestreoClima >= INTERVALO_CLIMA_MS) {
    ultimoMuestreoClima = ahora;
    float t = dht.readTemperature();
    float h = dht.readHumidity();

    if (!isnan(t)) {
      if (idx_clima < 3) {
        buffer_clima[idx_clima++] = t * ESCALA_TEMP;
      }

      if (idx_clima >= 3 && clima_listo) {
        idx_clima = 0;

        for (int i = 0; i < 3; i++) {
          if (input_clima.type == kTfLiteFloat32) {
            input_clima.data.f[i] = buffer_clima[i];
          } else if (input_clima.type == kTfLiteInt8) {
            int32_t q = (int32_t)round(buffer_clima[i] / input_clima.params.scale) + input_clima.params.zero_point;
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            input_clima.data.int8[i] = (int8_t)q;
          }
        }

        if (tflite_learn_1171835_5_invoke() == kTfLiteOk) {
          float max_val = -1.0f;
          int max_idx = 0;
          for (int i = 0; i < 3; i++) {
            float prob = (output_clima.type == kTfLiteFloat32)
              ? output_clima.data.f[i]
              : (output_clima.data.int8[i] - output_clima.params.zero_point) * output_clima.params.scale;
            if (prob > max_val) {
              max_val = prob;
              max_idx = i;
            }
          }

          estadoClima.temperatura = t;
          estadoClima.humedad = isnan(h) ? 0.0 : h;
          estadoClima.prediccion = ETIQUETAS_CLIMA[max_idx];
          estadoClima.confianza = round(max_val * 1000) / 1000.0;
          estadoClima.timestamp = obtenerTimestampISO();
          guardarHistorialClima(estadoClima);

          StaticJsonDocument<384> doc;
          doc["device_id"] = THINGSNAME;
          doc["lote"] = LOTE;
          doc["proceso"] = PROCESO;
          doc["timestamp"] = estadoClima.timestamp;
          doc["temperatura"] = estadoClima.temperatura;
          doc["humedad"] = estadoClima.humedad;
          doc["model_temp"] = estadoClima.prediccion;
          doc["confianza"] = estadoClima.confianza;

          char payload[384];
          serializeJson(doc, payload);
          bool ok = client.publish(TOPIC_CLIMA, payload);
          Serial.printf("[MODEL_TEMP] T:%.1f -> %s (%.2f) -> AWS: %s\n",
                        t, estadoClima.prediccion.c_str(), max_val, ok ? "OK" : "FALLO");
        }
      }
    }
  }
}