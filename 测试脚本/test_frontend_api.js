/**
 * 前端API交互测试脚本
 * 
 * 使用方法：
 * 1. 在浏览器中打开系统页面
 * 2. 打开浏览器控制台 (F12或右键检查)
 * 3. 将此脚本复制到控制台中运行
 */

// 设置测试配置
const config = {
    logPrefix: '🔍 API测试 | ',
    testDelay: 1000, // 测试间隔时间(毫秒)
};

// 测试结果统计
const testResults = {
    total: 0,
    passed: 0,
    failed: 0,
};

// 日志函数
function log(message, type = 'info') {
    const styles = {
        info: 'color: #2c6ab3; font-weight: bold;',
        success: 'color: #4CAF50; font-weight: bold;',
        error: 'color: #F44336; font-weight: bold;',
        warning: 'color: #FF9800; font-weight: bold;',
    };
    
    console.log(`%c${config.logPrefix}${message}`, styles[type] || styles.info);
}

// 测试函数
async function runTest(name, testFn) {
    log(`开始测试: ${name}...`);
    testResults.total++;
    
    try {
        const result = await testFn();
        if (result) {
            testResults.passed++;
            log(`测试通过: ${name} ✅`, 'success');
        } else {
            testResults.failed++;
            log(`测试失败: ${name} ❌`, 'error');
        }
        return result;
    } catch (error) {
        testResults.failed++;
        log(`测试异常: ${name} ❌ - ${error.message}`, 'error');
        console.error(error);
        return false;
    }
}

// 等待函数
function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// 测试API连接 - 健康检查
async function testHealthEndpoint() {
    try {
        const response = await axios.get('/health');
        return response.status === 200 && response.data.status === 'healthy';
    } catch (error) {
        console.error('健康检查API错误:', error);
        return false;
    }
}

// 测试排放数据API
async function testEmissionsAPI() {
    try {
        const response = await axios.get('/api/emissions');
        return response.status === 200 && 
               response.data.data && 
               Array.isArray(response.data.data) && 
               response.data.data.length > 0;
    } catch (error) {
        console.error('排放数据API错误:', error);
        return false;
    }
}

// 测试预测API
async function testPredictionAPI() {
    try {
        // 准备测试数据
        const features = [
            75.0, 25.0, 60.0, 3.5, 120.0, 
            new Date().getHours(), 
            new Date().getDate(), 
            new Date().getMonth() + 1, 
            new Date().getFullYear(), 
            new Date().getDay(),
            20.0, 40.0, 8.0
        ];
        
        const requestData = {
            features: [features],
            model_type: 'lstm'
        };
        
        const response = await axios.post('/predict', requestData);
        return response.status === 200 && 
               response.data.predictions && 
               Array.isArray(response.data.predictions);
    } catch (error) {
        console.error('预测API错误:', error);
        return false;
    }
}

// 测试自然语言查询API
async function testNLQueryAPI() {
    try {
        const requestData = {
            query: '最近24小时的SO2平均排放浓度是多少?',
            mode: 'text'
        };
        
        const response = await axios.post('/api/nlp_query', requestData);
        return response.status === 200 && response.data.answer;
    } catch (error) {
        console.error('自然语言查询API错误:', error);
        return false;
    }
}

// 测试页面DOM元素是否正确加载
function testPageElements() {
    // 获取当前页面路径
    const path = window.location.pathname;
    
    // 根据不同页面测试不同元素
    if (path === '/' || path === '/index.html') {
        // 首页
        return document.getElementById('so2-avg') !== null &&
               document.getElementById('nox-avg') !== null &&
               document.getElementById('dust-avg') !== null &&
               document.getElementById('emission-status-tbody') !== null;
    } else if (path === '/dashboard') {
        // 数据看板
        return document.getElementById('avg-so2') !== null &&
               document.getElementById('avg-nox') !== null &&
               document.getElementById('avg-dust') !== null &&
               document.getElementById('emissions-chart') !== null;
    } else if (path === '/prediction') {
        // 排放预测
        return document.getElementById('prediction-form') !== null &&
               document.getElementById('feature-load') !== null &&
               document.getElementById('predict-btn') !== null;
    } else if (path === '/nlp') {
        // 自然语言查询
        return document.getElementById('query-form') !== null ||
               document.querySelector('.query-container') !== null;
    } else if (path === '/speech') {
        // 语音识别
        return document.getElementById('record-btn') !== null ||
               document.querySelector('.speech-container') !== null;
    }
    
    return false;
}

// 主测试函数
async function runAllTests() {
    log('开始前端API连接测试...', 'info');
    
    // 测试健康检查API
    await runTest('健康检查API', testHealthEndpoint);
    await wait(config.testDelay);
    
    // 测试排放数据API
    await runTest('排放数据API', testEmissionsAPI);
    await wait(config.testDelay);
    
    // 测试预测API
    await runTest('排放预测API', testPredictionAPI);
    await wait(config.testDelay);
    
    // 测试自然语言查询API
    await runTest('自然语言查询API', testNLQueryAPI);
    await wait(config.testDelay);
    
    // 测试页面元素
    await runTest('页面DOM元素', testPageElements);
    
    // 输出测试结果汇总
    log(`测试完成: 总共 ${testResults.total} 项测试，通过 ${testResults.passed} 项，失败 ${testResults.failed} 项`, 
        testResults.failed === 0 ? 'success' : 'warning');
}

// 检查是否有Axios
if (typeof axios === 'undefined') {
    log('警告: 未检测到Axios库，API测试可能无法进行', 'warning');
    log('请确保页面已加载Axios或手动添加: <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>', 'warning');
} else {
    // 开始运行测试
    runAllTests().catch(error => {
        log(`测试过程中发生错误: ${error.message}`, 'error');
        console.error(error);
    });
} 