import json
import boto3
from boto3.dynamodb.conditions import Key
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta

dynamodb = boto3.resource('dynamodb')

# 1. Nombres exactos de tus 3 tablas
TABLA_CAMARA     = 'valores-pcamara-esp32cam'
TABLA_CLIMA      = 'valores-sensor-temp'      
TABLA_MOVIMIENTO = 'valores-sensor-AcGi'

# 2. Partition Keys (device_id)
DEVICE_ID_CAM      = 'ESP32_CAM'
DEVICE_ID_SENSORES = 'esp32-cocatrace'      

# Zona horaria Colombia (UTC-5)
ZONA_COL = timezone(timedelta(hours=-5))

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        return super(DecimalEncoder, self).default(obj)

def formatear_fecha(ts):
    """Unifica timestamps Unix (ESP32-CAM) e ISO-8601 (ESP32) a YYYY-MM-DD HH:MM:SS."""
    if ts is None:
        return ""
    if isinstance(ts, (int, float, Decimal)):
        dt = datetime.fromtimestamp(float(ts), tz=ZONA_COL)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
   
    ts_str = str(ts).strip()
    if ts_str.isdigit():
        dt = datetime.fromtimestamp(float(ts_str), tz=ZONA_COL)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
   
    return ts_str.replace('T', ' ').replace('-05:00', '')

def desempaquetar_item(item):
    """Extrae atributos encapsulados (STRING/payload) y normaliza la fecha."""
    resultado = dict(item)
    for campo_empaquetado in ('STRING', 'payload', 'data'):
        if campo_empaquetado in resultado:
            contenido = resultado.pop(campo_empaquetado)
            if isinstance(contenido, str):
                try:
                    contenido = json.loads(contenido)
                except Exception:
                    pass
            if isinstance(contenido, dict):
                resultado.update(contenido)

    if 'timestamp' in resultado:
        resultado['timestamp'] = formatear_fecha(resultado['timestamp'])

    return resultado

def consultar_tabla(nombre_tabla, device_id, lote=None, limite=20):
    try:
        tabla = dynamodb.Table(nombre_tabla)
        filtrar_lote = bool(lote and lote != 'TODOS')

        query_kwargs = {
            'KeyConditionExpression': Key('device_id').eq(device_id),
            'ScanIndexForward': False
        }

        # Evita escanear toda la tabla cuando no hay filtro por lote
        if not filtrar_lote:
            query_kwargs['Limit'] = limite

        items_limpios = []

        while True:
            respuesta = tabla.query(**query_kwargs)
            for row in respuesta.get('Items', []):
                item = desempaquetar_item(row)
                if not filtrar_lote or str(item.get('lote', '')) == lote:
                    items_limpios.append(item)
                    if len(items_limpios) >= limite:
                        return items_limpios

            if 'LastEvaluatedKey' in respuesta and len(items_limpios) < limite:
                query_kwargs['ExclusiveStartKey'] = respuesta['LastEvaluatedKey']
            else:
                break

        return items_limpios

    except Exception as e:
        print(f"Error consultando {nombre_tabla}: {str(e)}")
        return []

def lambda_handler(event, context):
    params = event.get('queryStringParameters') or {}
    lote = params.get('lote')
    tabla_unica = params.get('tabla')  # 'camara', 'clima', 'movimiento' o None

    try:
        limite = min(int(params.get('limite', 20)), 100)  # Máximo 100 registros
    except ValueError:
        limite = 20

    resultado = {'camara': [], 'clima': [], 'movimiento': []}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futuros = {}
        if not tabla_unica or tabla_unica == 'camara':
            futuros['camara'] = executor.submit(consultar_tabla, TABLA_CAMARA, DEVICE_ID_CAM, lote, limite)
        if not tabla_unica or tabla_unica == 'clima':
            futuros['clima'] = executor.submit(consultar_tabla, TABLA_CLIMA, DEVICE_ID_SENSORES, lote, limite)
        if not tabla_unica or tabla_unica == 'movimiento':
            futuros['movimiento'] = executor.submit(consultar_tabla, TABLA_MOVIMIENTO, DEVICE_ID_SENSORES, lote, limite)

        for clave, fut in futuros.items():
            resultado[clave] = fut.result()

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET,OPTIONS'
        },
        'body': json.dumps(resultado, cls=DecimalEncoder)
    }