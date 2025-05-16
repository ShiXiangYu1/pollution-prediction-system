# 电厂污染物排放预测系统API文档

## 1. 概述

本文档详细介绍电厂污染物排放预测系统提供的所有API接口，包括请求方式、参数说明、返回值格式和示例。系统API采用RESTful风格设计，基于HTTP协议，返回JSON格式数据。

## 2. 基础信息

- **基础URL**: `http://localhost:8000` （本地部署）或系统管理员提供的URL
- **认证方式**: 当前版本不需要认证
- **通用响应格式**:
  成功响应:
  ```json
  {
    "status": "success",
    "data": {...}  // 具体数据
  }
  ```
  错误响应:
  ```json
  {
    "status": "error",
    "message": "错误信息",
    "error_code": 错误代码
  }
  ```

## 3. API端点

### 3.1 健康检查

#### 请求
- **URL**: `/health`
- **方法**: GET
- **参数**: 无

#### 响应
```json
{
  "status": "healthy",
  "timestamp": "2023-05-11T08:30:45.123Z"
}
```

### 3.2 污染物排放预测

#### 请求
- **URL**: `/predict`
- **方法**: POST
- **Content-Type**: application/json
- **请求体**:
  ```json
  {
    "features": [
      [负荷, 温度, 湿度, 风速, ...],  // 时间点1的特征
      [负荷, 温度, 湿度, 风速, ...],  // 时间点2的特征
      ...
    ],
    "model_type": "lstm"  // 可选值: "lstm", "gru", "transformer"
  }
  ```

#### 响应
```json
{
  "predictions": [
    [SO2预测值, NOx预测值, 烟尘预测值],  // 时间点1的预测结果
    [SO2预测值, NOx预测值, 烟尘预测值],  // 时间点2的预测结果
    ...
  ],
  "model_type": "lstm",
  "timestamp": "2023-05-11T08:32:10.456Z"
}
```

### 3.3 自然语言查询

#### 请求
- **URL**: `/api/nlp_query`
- **方法**: POST
- **Content-Type**: application/json
- **请求体**:
  ```json
  {
    "query": "昨天的SO2平均排放量是多少?",
    "mode": "text"  // 可选值: "text", "speech"
  }
  ```

#### 响应
```json
{
  "answer": "昨天的SO2平均排放浓度为18.5 mg/m³，低于标准限值(35 mg/m³)。",
  "chart_data": {
    "type": "line",
    "title": "SO2排放趋势",
    "xAxis": {
      "type": "time",
      "data": ["2023-05-10T00:00:00", ... , "2023-05-10T23:00:00"]
    },
    "yAxis": {
      "name": "浓度(mg/m³)"
    },
    "series": [
      {
        "name": "SO2浓度",
        "data": [19.2, 18.7, ... , 17.9]
      },
      {
        "name": "标准限值",
        "data": [35, 35, ... , 35],
        "lineStyle": {
          "type": "dashed"
        }
      }
    ]
  },
  "sql": "SELECT AVG(concentration) FROM emissions WHERE pollutant_type='SO2' AND time >= '2023-05-10 00:00:00' AND time < '2023-05-11 00:00:00';"
}
```

### 3.4 文本转SQL

#### 请求
- **URL**: `/nl2sql`
- **方法**: POST
- **Content-Type**: application/json
- **请求体**:
  ```json
  {
    "text": "查询2号机组昨天的NOx排放超标次数"
  }
  ```

#### 响应
```json
{
  "sql": "SELECT COUNT(*) FROM emissions WHERE unit_id = 2 AND pollutant_type = 'NOx' AND concentration > 50 AND time >= '2023-05-10 00:00:00' AND time < '2023-05-11 00:00:00';",
  "explanation": "该SQL查询计算2号机组在昨天（2023-05-10）NOx浓度超过50 mg/m³的记录数量，其中50 mg/m³是NOx的排放标准限值。",
  "timestamp": "2023-05-11T08:35:20.789Z"
}
```

### 3.5 语音转文本

#### 请求
- **URL**: `/speech-to-text`
- **方法**: POST
- **Content-Type**: multipart/form-data
- **参数**:
  - `file`: 音频文件（支持格式：WAV, MP3, 16kHz采样率）

#### 响应
```json
{
  "text": "查询上周所有机组的NOx排放情况",
  "confidence": 0.95,
  "timestamp": "2023-05-11T08:38:15.123Z"
}
```

### 3.6 获取排放数据

#### 请求
- **URL**: `/api/emissions`
- **方法**: GET
- **参数**:
  - `page`: 页码，从1开始（默认：1）
  - `page_size`: 每页数据条数，1-100（默认：10）
  - `type`: 污染物类型，如SO2、NOx、dust（可选）
  - `start_date`: 开始日期，格式YYYY-MM-DD（可选）
  - `end_date`: 结束日期，格式YYYY-MM-DD（可选）
  - `unit_id`: 机组ID（可选）

#### 响应
```json
{
  "data": [
    {
      "id": 12345,
      "timestamp": "2023-05-10T08:00:00Z",
      "unit_id": 1,
      "unit_name": "1号机组",
      "pollutant_type": "SO2",
      "concentration": 18.5,
      "standard_limit": 35,
      "load": 320.5
    },
    // ... 更多数据
  ],
  "pagination": {
    "page": 1,
    "page_size": 10,
    "total_items": 240,
    "total_pages": 24
  }
}
```

### 3.7 获取预测模型信息

#### 请求
- **URL**: `/api/models`
- **方法**: GET
- **参数**:
  - `type`: 模型类型，如lstm、gru、transformer（可选）

#### 响应
```json
{
  "models": [
    {
      "id": "lstm_v1.2",
      "type": "lstm",
      "description": "基于LSTM的污染物排放预测模型",
      "features": [
        {"name": "load", "description": "机组负荷", "unit": "MW"},
        {"name": "temperature", "description": "环境温度", "unit": "°C"},
        // ... 更多特征
      ],
      "metrics": {
        "rmse": 1.25,
        "mae": 0.85,
        "r2": 0.92
      },
      "last_updated": "2023-05-01T10:30:00Z"
    },
    // ... 更多模型
  ]
}
```

### 3.8 上传数据

#### 请求
- **URL**: `/api/upload/data`
- **方法**: POST
- **Content-Type**: multipart/form-data
- **参数**:
  - `file`: CSV或Excel文件
  - `data_type`: 数据类型（emissions、weather、operation等）

#### 响应
```json
{
  "status": "success",
  "message": "数据上传成功",
  "details": {
    "records_processed": 1250,
    "records_inserted": 1230,
    "records_updated": 15,
    "records_failed": 5,
    "failure_reasons": [
      {"line": 23, "reason": "时间格式错误"},
      // ... 更多错误原因
    ]
  }
}
```

### 3.9 系统状态

#### 请求
- **URL**: `/api/system/status`
- **方法**: GET
- **参数**: 无

#### 响应
```json
{
  "system": {
    "version": "1.2.0",
    "uptime": "5天12小时30分钟",
    "cpu_usage": 23.5,
    "memory_usage": 45.2,
    "disk_usage": 38.7
  },
  "services": [
    {
      "name": "数据采集服务",
      "status": "running",
      "last_activity": "2023-05-11T08:40:10Z"
    },
    {
      "name": "预测服务",
      "status": "running",
      "last_activity": "2023-05-11T08:39:55Z"
    },
    {
      "name": "NLP服务",
      "status": "running",
      "last_activity": "2023-05-11T08:38:30Z"
    }
  ],
  "data_stats": {
    "emissions_records": 1250000,
    "latest_data_time": "2023-05-11T08:40:00Z",
    "database_size": "2.3 GB"
  }
}
```

## 4. 错误代码

| 错误代码 | 描述 |
|---------|------|
| 400 | 请求参数错误 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |
| 1001 | 数据验证失败 |
| 1002 | 预测模型错误 |
| 1003 | 自然语言处理错误 |
| 1004 | 语音识别错误 |
| 1005 | 数据库查询错误 |

## 5. 示例代码

### 5.1 Python示例

```python
import requests
import json

# 基础URL
base_url = "http://localhost:8000"

# 预测污染物排放
def predict_emissions(features, model_type="lstm"):
    url = f"{base_url}/predict"
    payload = {
        "features": features,
        "model_type": model_type
    }
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"错误: {response.status_code}")
        print(response.text)
        return None

# 自然语言查询
def nlp_query(query_text):
    url = f"{base_url}/api/nlp_query"
    payload = {
        "query": query_text,
        "mode": "text"
    }
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"错误: {response.status_code}")
        print(response.text)
        return None

# 使用示例
if __name__ == "__main__":
    # 预测示例
    features = [
        [320.5, 25.2, 65.3, 3.2, 2.1, 0.8, 1.0, 0.5, 0.2, 0.1],
        [325.0, 25.5, 64.8, 3.5, 2.2, 0.7, 1.1, 0.6, 0.3, 0.1]
    ]
    prediction_result = predict_emissions(features, "lstm")
    print("预测结果:", prediction_result)
    
    # 查询示例
    query_result = nlp_query("昨天的SO2平均排放量是多少?")
    print("查询结果:", query_result)
```

### 5.2 JavaScript示例

```javascript
// 基础URL
const baseUrl = "http://localhost:8000";

// 预测污染物排放
async function predictEmissions(features, modelType = "lstm") {
  try {
    const response = await fetch(`${baseUrl}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        features: features,
        model_type: modelType
      })
    });
    
    if (response.ok) {
      return await response.json();
    } else {
      console.error(`错误: ${response.status}`);
      console.error(await response.text());
      return null;
    }
  } catch (error) {
    console.error("请求错误:", error);
    return null;
  }
}

// 自然语言查询
async function nlpQuery(queryText) {
  try {
    const response = await fetch(`${baseUrl}/api/nlp_query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        query: queryText,
        mode: "text"
      })
    });
    
    if (response.ok) {
      return await response.json();
    } else {
      console.error(`错误: ${response.status}`);
      console.error(await response.text());
      return null;
    }
  } catch (error) {
    console.error("请求错误:", error);
    return null;
  }
}

// 使用示例
async function runExamples() {
  // 预测示例
  const features = [
    [320.5, 25.2, 65.3, 3.2, 2.1, 0.8, 1.0, 0.5, 0.2, 0.1],
    [325.0, 25.5, 64.8, 3.5, 2.2, 0.7, 1.1, 0.6, 0.3, 0.1]
  ];
  const predictionResult = await predictEmissions(features, "lstm");
  console.log("预测结果:", predictionResult);
  
  // 查询示例
  const queryResult = await nlpQuery("昨天的SO2平均排放量是多少?");
  console.log("查询结果:", queryResult);
}

runExamples();
```

## 6. 注意事项

1. **请求频率限制**: 每个IP每分钟最多可发送100个请求，超过限制可能会被暂时限制访问。
2. **数据大小限制**: 上传文件大小不超过50MB，自然语言查询长度不超过500个字符，语音文件长度不超过1分钟。
3. **时区说明**: 所有时间戳均采用UTC时间，客户端可根据需要转换为本地时间。
4. **API变更**: API可能会有更新，请定期查看文档了解最新变更。
5. **错误处理**: 客户端应实现合理的错误处理机制，对API返回的错误进行适当处理。 