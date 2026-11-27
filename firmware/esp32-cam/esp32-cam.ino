#include <Train-Esp32cam_inferencing.h>
#include "edge-impulse-sdk/dsp/image/image.hpp"
#include "esp_camera.h"

#include <WiFiClientSecure.h>
#include <WiFi.h>
#include <WebServer.h>
#include "credentials.h"
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include "time.h"

#define CAMERA_MODEL_AI_THINKER // Has PSRAM

#if defined(CAMERA_MODEL_AI_THINKER)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#else
#error "Camera model not selected"
#endif

/* Configuración AWS IoT Core ---------------------------------------------- */
#define imagen_topic "esp32/imagen"

unsigned long lastMsg = 0;
const long interval = 2000;

WiFiClientSecure espClient;
PubSubClient esp32(espClient);

// Servidor Web HTTP en puerto 80 para el protocolo MCP
WebServer mcpServer(80);

// =========================================================================
// MEMORIA PARA EL SERVIDOR MCP (ESTADO ACTUAL + ÚLTIMOS 10 REGISTROS)
// =========================================================================
#define MAX_HISTORIAL_CAM 10

struct RegistroHoja {
  String estado_hoja = "sin_deteccion_aun";
  float confianza = 0.0;
  int x = 0;
  int y = 0;
  String timestamp = "";
};

RegistroHoja estadoActualHoja;
RegistroHoja historialHoja[MAX_HISTORIAL_CAM];
int totalHistorialHoja = 0;

void guardarHistorialHoja(const RegistroHoja& reg) {
  for (int i = MAX_HISTORIAL_CAM - 1; i > 0; i--) {
    historialHoja[i] = historialHoja[i - 1];
  }
  historialHoja[0] = reg;
  if (totalHistorialHoja < MAX_HISTORIAL_CAM) totalHistorialHoja++;
}

String obtenerTimestampISO() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) return "1970-01-01T00:00:00-05:00";
  char buf[30];
  strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S-05:00", &timeinfo);
  return String(buf);
}

// =========================================================================
// SERVIDOR MCP EMBEBIDO EN ESP32-CAM (POST /mcp)
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
    mcpServer.send(200, "application/json", "{\"status\":\"MCP Server ESP32-CAM activo en /mcp\"}");
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

  DynamicJsonDocument res(4096);
  res["jsonrpc"] = "2.0";
  res["id"] = id;

  // 1. Handshake MCP: initialize
  if (strcmp(method, "initialize") == 0) {
    JsonObject result = res.createNestedObject("result");
    result["protocolVersion"] = "2024-11-05";
    JsonObject caps = result.createNestedObject("capabilities");
    caps.createNestedObject("tools");
    JsonObject serverInfo = result.createNestedObject("serverInfo");
    serverInfo["name"] = "esp32-cam-vision-mcp";
    serverInfo["version"] = "1.0.0";
  }
  // 2. Listado de herramientas: tools/list
  else if (strcmp(method, "tools/list") == 0) {
    JsonObject result = res.createNestedObject("result");
    JsonArray tools = result.createNestedArray("tools");

    JsonObject t1 = tools.createNestedObject();
    t1["name"] = "obtener_estado_hoja";
    t1["description"] = "Obtiene la clasificacion visual mas reciente de la hoja (sana, seca, hongo), su confianza y coordenadas (x, y) detectadas por la ESP32-CAM con FOMO.";
    JsonObject s1 = t1.createNestedObject("inputSchema");
    s1["type"] = "object";
    s1.createNestedObject("properties");

    JsonObject t2 = tools.createNestedObject();
    t2["name"] = "obtener_historial_camara";
    t2["description"] = "Devuelve los ultimos 10 registros de detecciones visuales de hojas capturados por la ESP32-CAM.";
    JsonObject s2 = t2.createNestedObject("inputSchema");
    s2["type"] = "object";
    s2.createNestedObject("properties");
  }
  // 3. Ejecución de herramienta: tools/call
  else if (strcmp(method, "tools/call") == 0) {
    const char* toolName = req["params"]["name"] | "";
    JsonObject result = res.createNestedObject("result");
    JsonArray content = result.createNestedArray("content");
    JsonObject item = content.createNestedObject();
    item["type"] = "text";

    if (strcmp(toolName, "obtener_estado_hoja") == 0) {
      StaticJsonDocument<512> info;
      info["camara_id"] = SENSOR_ID;
      info["lote"] = LOTE;
      info["proceso"] = PROCESO;
      info["estado_hoja"] = estadoActualHoja.estado_hoja;
      info["confianza"] = estadoActualHoja.confianza;
      info["coordenada_x"] = estadoActualHoja.x;
      info["coordenada_y"] = estadoActualHoja.y;
      info["ultima_deteccion"] = estadoActualHoja.timestamp;

      String textoSalida;
      serializeJsonPretty(info, textoSalida);
      item["text"] = textoSalida;
    }
    else if (strcmp(toolName, "obtener_historial_camara") == 0) {
      DynamicJsonDocument histDoc(2048);
      JsonArray arr = histDoc.createNestedArray("ultimas_detecciones_hoja");
      for (int i = 0; i < totalHistorialHoja; i++) {
        JsonObject h = arr.createNestedObject();
        h["timestamp"] = historialHoja[i].timestamp;
        h["estado_hoja"] = historialHoja[i].estado_hoja;
        h["confianza"] = historialHoja[i].confianza;
        h["x"] = historialHoja[i].x;
        h["y"] = historialHoja[i].y;
      }

      String textoSalida;
      serializeJson(histDoc, textoSalida);
      item["text"] = textoSalida;
    }
    else {
      result["isError"] = true;
      item["text"] = "Herramienta no encontrada en ESP32-CAM.";
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
  Serial.print("Servidor MCP de ESP32-CAM listo en: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/mcp");
}

/* Constant defines -------------------------------------------------------- */
#define EI_CAMERA_RAW_FRAME_BUFFER_COLS           320
#define EI_CAMERA_RAW_FRAME_BUFFER_ROWS           240
#define EI_CAMERA_FRAME_BYTE_SIZE                 3

static bool debug_nn = false;
static bool is_initialised = false;
uint8_t *snapshot_buf;

static camera_config_t camera_config = {
    .pin_pwdn = PWDN_GPIO_NUM,
    .pin_reset = RESET_GPIO_NUM,
    .pin_xclk = XCLK_GPIO_NUM,
    .pin_sscb_sda = SIOD_GPIO_NUM,
    .pin_sscb_scl = SIOC_GPIO_NUM,
    .pin_d7 = Y9_GPIO_NUM,
    .pin_d6 = Y8_GPIO_NUM,
    .pin_d5 = Y7_GPIO_NUM,
    .pin_d4 = Y6_GPIO_NUM,
    .pin_d3 = Y5_GPIO_NUM,
    .pin_d2 = Y4_GPIO_NUM,
    .pin_d1 = Y3_GPIO_NUM,
    .pin_d0 = Y2_GPIO_NUM,
    .pin_vsync = VSYNC_GPIO_NUM,
    .pin_href = HREF_GPIO_NUM,
    .pin_pclk = PCLK_GPIO_NUM,
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    .pixel_format = PIXFORMAT_JPEG,
    .frame_size = FRAMESIZE_QVGA,
    .jpeg_quality = 12,
    .fb_count = 1,
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};

bool ei_camera_init(void);
void ei_camera_deinit(void);
bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf);
static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr);

void conncetAWS() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.println("Conectando a Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi conectado");

  espClient.setCACert(AWS_CERT_CA);
  espClient.setCertificate(AWS_CERT_CRT);
  espClient.setPrivateKey(AWS_CERT_PRIVATE);

  esp32.setServer(AWS_Iot_Endpoint, 8883);
  esp32.setBufferSize(512);

  Serial.println("Conectando a AWS IoT...");
  int retries = 0;
  while (!esp32.connect(THINGSNAME)) {
    Serial.print(".");
    delay(500);
    retries++;
    if (retries > 20) {
      Serial.println("\n[ERROR] Timeout al conectar con AWS IoT");
      return;
    }
  }
  Serial.println("\nConectado a AWS IoT :D");
}

void publishDetection(const char* etiqueta, float confianza, int x, int y) {
  // Actualizamos la memoria local para el servidor MCP
  estadoActualHoja.estado_hoja = String(etiqueta);
  estadoActualHoja.confianza   = round(confianza * 1000) / 1000.0;
  estadoActualHoja.x           = x;
  estadoActualHoja.y           = y;
  estadoActualHoja.timestamp   = obtenerTimestampISO();
  guardarHistorialHoja(estadoActualHoja);

  StaticJsonDocument<256> doc;
  time_t now = time(nullptr);

  doc["timestamp"] = now;
  doc["id_sensor"] = SENSOR_ID;
  doc["estado_hoja"] = etiqueta;
  doc["confianza"] = estadoActualHoja.confianza;
  doc["x"] = x;
  doc["y"] = y;
  doc["lote"] = LOTE;
  doc["proceso"] = PROCESO;

  char jsonBuffer[512];
  serializeJson(doc, jsonBuffer);

  if (esp32.publish(imagen_topic, jsonBuffer)) {
    Serial.println("Mensaje JSON publicado en MQTT exitosamente!");
    Serial.print("Payload: ");
    Serial.println(jsonBuffer);
  } else {
    Serial.println("[ERROR] Falló la publicación MQTT.");
  }
}

void reconnect() {
  while (!esp32.connected()) {
    Serial.print("Intentando conexión MQTT...");
    if (esp32.connect(THINGSNAME)) {
      Serial.println("¡Conectado!");
    } else {
      Serial.print("Fallo, rc=");
      Serial.print(esp32.state());
      Serial.println(" Reintentando en 5 segundos...");
      delay(5000);
    }
  }
}

void setup() {
    Serial.begin(115200);
    while (!Serial);
    Serial.println("Edge Impulse Inferencing Demo + MCP");

    if (ei_camera_init() == false) {
        ei_printf("Failed to initialize Camera!\r\n");
    } else {
        ei_printf("Camera initialized\r\n");
    }

    conncetAWS();

    Serial.println("--- Estado Wi-Fi ---");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());

    configTime(-5 * 3600, 0, "pool.ntp.org");
    struct tm timeinfo;
    Serial.print("Obteniendo hora NTP");
    while (!getLocalTime(&timeinfo)) {
        Serial.print(".");
        delay(500);
    }
    Serial.println("\nHora sincronizada correctamente!");

    // Iniciamos el servidor MCP de la cámara
    iniciarServidorMCP();

    ei_printf("\nStarting continious inference in 2 seconds...\n");
    ei_sleep(2000);
}

void loop() {
    if (!esp32.connected()) {
        reconnect();
    }
    esp32.loop();
    mcpServer.handleClient(); // Atiende peticiones MCP de VS Code / Claude

    if (ei_sleep(5) != EI_IMPULSE_OK) {
        return;
    }

    snapshot_buf = (uint8_t*)malloc(EI_CAMERA_RAW_FRAME_BUFFER_COLS * EI_CAMERA_RAW_FRAME_BUFFER_ROWS * EI_CAMERA_FRAME_BYTE_SIZE);

    if (snapshot_buf == nullptr) {
        ei_printf("ERR: Failed to allocate snapshot buffer!\n");
        return;
    }

    ei::signal_t signal;
    signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
    signal.get_data = &ei_camera_get_data;

    if (ei_camera_capture((size_t)EI_CLASSIFIER_INPUT_WIDTH, (size_t)EI_CLASSIFIER_INPUT_HEIGHT, snapshot_buf) == false) {
        ei_printf("Failed to capture image\r\n");
        free(snapshot_buf);
        return;
    }

    ei_impulse_result_t result = { 0 };

    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, debug_nn);
    if (err != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", err);
        free(snapshot_buf);
        return;
    }

#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    float max_val = 0.00;
    const char* best_label = "";
    int best_x = 0, best_y = 0;

    for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
        if (bb.value == 0) continue;

        if (bb.value > max_val) {
            max_val = bb.value;
            best_label = bb.label;
            best_x = bb.x;
            best_y = bb.y;
        }
    }

    unsigned long now = millis();
    if (max_val >= 0.65 && (now - lastMsg > interval)) {
        lastMsg = now;
        publishDetection(best_label, max_val, best_x, best_y);
    }
#endif

    free(snapshot_buf);
}

bool ei_camera_init(void) {
    if (is_initialised) return true;

    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
      Serial.printf("Camera init failed with error 0x%x\n", err);
      return false;
    }

    sensor_t * s = esp_camera_sensor_get();
    if (s->id.PID == OV3660_PID) {
      s->set_vflip(s, 1);
      s->set_brightness(s, 1);
      s->set_saturation(s, 0);
    }

    is_initialised = true;
    return true;
}

void ei_camera_deinit(void) {
    esp_err_t err = esp_camera_deinit();
    if (err != ESP_OK) {
        ei_printf("Camera deinit failed\n");
        return;
    }
    is_initialised = false;
}

bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf) {
    bool do_resize = false;

    if (!is_initialised) {
        ei_printf("ERR: Camera is not initialized\r\n");
        return false;
    }

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ei_printf("Camera capture failed\n");
        return false;
    }

    bool converted = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, snapshot_buf);
    esp_camera_fb_return(fb);

    if (!converted) {
        ei_printf("Conversion failed\n");
        return false;
    }

    if ((img_width != EI_CAMERA_RAW_FRAME_BUFFER_COLS) || (img_height != EI_CAMERA_RAW_FRAME_BUFFER_ROWS)) {
        do_resize = true;
    }

    if (do_resize) {
        ei::image::processing::crop_and_interpolate_rgb888(
        out_buf,
        EI_CAMERA_RAW_FRAME_BUFFER_COLS,
        EI_CAMERA_RAW_FRAME_BUFFER_ROWS,
        out_buf,
        img_width,
        img_height);
    }

    return true;
}

static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr) {
    size_t pixel_ix = offset * 3;
    size_t pixels_left = length;
    size_t out_ptr_ix = 0;

    while (pixels_left != 0) {
        out_ptr[out_ptr_ix] = (snapshot_buf[pixel_ix + 2] << 16) + (snapshot_buf[pixel_ix + 1] << 8) + snapshot_buf[pixel_ix];
        out_ptr_ix++;
        pixel_ix += 3;
        pixels_left--;
    }
    return 0;
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_CAMERA
#error "Invalid model for current sensor"
#endif