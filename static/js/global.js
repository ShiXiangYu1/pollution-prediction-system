/**
 * 全局工具函数库
 */

// 格式化日期时间
function formatDateTime(isoString) {
    if (!isoString) return '-';
    
    try {
        const date = new Date(isoString);
        return date.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            hour12: false
        });
    } catch (e) {
        console.error('日期格式化错误:', e);
        return isoString;
    }
}

// 仅格式化日期部分
function formatDate(isoString) {
    if (!isoString) return '-';
    
    try {
        const date = new Date(isoString);
        return date.toLocaleDateString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit'
        });
    } catch (e) {
        console.error('日期格式化错误:', e);
        return isoString;
    }
}

// 仅格式化时间部分
function formatTime(isoString) {
    if (!isoString) return '-';
    
    try {
        const date = new Date(isoString);
        return date.toLocaleTimeString('zh-CN', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        });
    } catch (e) {
        console.error('时间格式化错误:', e);
        return isoString;
    }
}

// 显示提示信息
function showAlert(message, type = 'info', duration = 3000) {
    // 清除之前的提示
    const oldAlerts = document.querySelectorAll('.alert-floating');
    oldAlerts.forEach(alert => {
        if (alert.dataset.removing !== 'true') {
            removeAlert(alert);
        }
    });
    
    // 创建新提示
    const alert = document.createElement('div');
    alert.className = `alert-floating alert-${type}`;
    alert.textContent = message;
    
    // 添加关闭按钮
    const closeBtn = document.createElement('button');
    closeBtn.className = 'close-btn';
    closeBtn.innerHTML = '&times;';
    closeBtn.addEventListener('click', () => removeAlert(alert));
    alert.appendChild(closeBtn);
    
    // 添加到页面
    document.body.appendChild(alert);
    
    // 延迟显示，用于动画效果
    setTimeout(() => {
        alert.classList.add('show');
    }, 10);
    
    // 自动消失
    if (duration > 0) {
        setTimeout(() => {
            removeAlert(alert);
        }, duration);
    }
    
    const removeAlert = (alertEl) => {
        if (!alertEl) return;
        
        alertEl.dataset.removing = 'true';
        alertEl.classList.remove('show');
        
        // 删除元素
        setTimeout(() => {
            if (alertEl.parentNode) {
                alertEl.parentNode.removeChild(alertEl);
            }
        }, 300); // 与CSS过渡时间匹配
    };
    
    return alert;
}

// 处理API错误
function handleApiError(error) {
    console.error('API错误:', error);
    
    let errorMessage = '操作失败，请稍后重试';
    
    // 处理axios错误
    if (error.response) {
        // 服务器返回错误状态码
        console.error('响应状态码:', error.response.status);
        console.error('响应数据:', error.response.data);
        
        if (error.response.data && error.response.data.detail) {
            errorMessage = `错误: ${error.response.data.detail}`;
        } else if (error.response.status === 404) {
            errorMessage = '请求的资源不存在';
        } else if (error.response.status === 401) {
            errorMessage = '未授权，请重新登录';
            // 可以在这里添加重定向到登录页面的逻辑
        } else if (error.response.status === 403) {
            errorMessage = '您没有权限执行此操作';
        } else if (error.response.status >= 500) {
            errorMessage = '服务器错误，请联系管理员';
        }
    } else if (error.request) {
        // 请求已发送但未收到响应
        console.error('未收到响应:', error.request);
        errorMessage = '服务器未响应，请检查网络连接';
    } else {
        // 请求配置出错
        console.error('请求错误:', error.message);
        errorMessage = `请求错误: ${error.message}`;
    }
    
    // 显示错误提示
    showAlert(errorMessage, 'danger', 5000);
    
    return errorMessage;
}

// 复制到剪贴板
function copyToClipboard(text) {
    // 检查Clipboard API是否可用
    if (navigator.clipboard && window.isSecureContext) {
        return navigator.clipboard.writeText(text)
            .then(() => {
                showAlert('复制成功', 'success');
                return true;
            })
            .catch(err => {
                console.error('复制失败:', err);
                showAlert('复制失败', 'danger');
                return false;
            });
    } else {
        // 回退到旧方法
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        let success = false;
        try {
            success = document.execCommand('copy');
            showAlert(success ? '复制成功' : '复制失败', success ? 'success' : 'danger');
        } catch (err) {
            console.error('复制失败:', err);
            showAlert('复制失败', 'danger');
        }
        
        document.body.removeChild(textArea);
        return success;
    }
}

// 节流函数 - 限制函数调用频率
function throttle(func, delay) {
    let lastCall = 0;
    return function(...args) {
        const now = new Date().getTime();
        if (now - lastCall < delay) {
            return;
        }
        lastCall = now;
        return func(...args);
    };
}

// 防抖函数 - 延迟函数调用，重新计时
function debounce(func, delay) {
    let timer = null;
    return function(...args) {
        clearTimeout(timer);
        timer = setTimeout(() => {
            func(...args);
        }, delay);
    };
}

// 格式化数字，添加千位分隔符和固定小数位
function formatNumber(num, decimals = 2) {
    if (isNaN(num) || num === null) return '-';
    
    return Number(num).toLocaleString('zh-CN', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

// 生成唯一ID
function generateId(prefix = '') {
    return `${prefix}${Date.now()}-${Math.floor(Math.random() * 10000)}`;
}

// 更新URL参数
function updateUrlParam(key, value) {
    const url = new URL(window.location.href);
    url.searchParams.set(key, value);
    window.history.replaceState({}, '', url.toString());
} 