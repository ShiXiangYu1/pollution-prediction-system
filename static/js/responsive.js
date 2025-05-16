/**
 * 响应式设计JavaScript文件
 * 负责处理响应式导航和主题切换等功能
 */

document.addEventListener('DOMContentLoaded', function() {
    // 初始化变量
    const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
    const mobileMenuClose = document.querySelector('.mobile-menu-close');
    const mobileSidebar = document.querySelector('.mobile-sidebar');
    const themeToggle = document.getElementById('theme-toggle');
    const userDropdownToggle = document.querySelector('.user-dropdown-toggle');
    const userDropdownMenu = document.querySelector('.user-dropdown .dropdown-menu');
    const bodyElement = document.body;
    
    // 检查元素是否存在再添加事件监听器
    if (mobileMenuToggle && mobileSidebar) {
        // 移动端菜单切换
        mobileMenuToggle.addEventListener('click', function() {
            mobileSidebar.classList.add('show');
            bodyElement.classList.add('mobile-menu-open');
            // 防止背景滚动
            bodyElement.style.overflow = 'hidden';
        });
    }
    
    if (mobileMenuClose && mobileSidebar) {
        // 关闭移动端菜单
        mobileMenuClose.addEventListener('click', function() {
            mobileSidebar.classList.remove('show');
            bodyElement.classList.remove('mobile-menu-open');
            // 恢复背景滚动
            bodyElement.style.overflow = '';
        });
        
        // 点击菜单外部关闭菜单
        document.addEventListener('click', function(event) {
            if (mobileSidebar.classList.contains('show') && 
                !mobileSidebar.contains(event.target) && 
                !mobileMenuToggle.contains(event.target)) {
                mobileSidebar.classList.remove('show');
                bodyElement.classList.remove('mobile-menu-open');
                bodyElement.style.overflow = '';
            }
        });
    }
    
    // 主题切换功能
    if (themeToggle) {
        // 检查本地存储的主题偏好
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            bodyElement.classList.add('dark-theme');
            updateThemeIcon(true);
        }
        
        themeToggle.addEventListener('click', function() {
            const isDarkTheme = bodyElement.classList.toggle('dark-theme');
            updateThemeIcon(isDarkTheme);
            
            // 保存主题偏好到本地存储
            localStorage.setItem('theme', isDarkTheme ? 'dark' : 'light');
        });
    }
    
    // 用户下拉菜单
    if (userDropdownToggle && userDropdownMenu) {
        userDropdownToggle.addEventListener('click', function(event) {
            event.stopPropagation();
            userDropdownMenu.classList.toggle('show');
        });
        
        // 点击下拉菜单外部关闭菜单
        document.addEventListener('click', function(event) {
            if (userDropdownMenu.classList.contains('show') && 
                !userDropdownMenu.contains(event.target) && 
                !userDropdownToggle.contains(event.target)) {
                userDropdownMenu.classList.remove('show');
            }
        });
    }
    
    // 处理移动端的子菜单展开
    const mobileNavItemsWithSubmenus = document.querySelectorAll('.mobile-nav-item.has-submenu');
    if (mobileNavItemsWithSubmenus) {
        mobileNavItemsWithSubmenus.forEach(function(item) {
            const toggleButton = item.querySelector('.submenu-toggle');
            if (toggleButton) {
                toggleButton.addEventListener('click', function(event) {
                    event.preventDefault();
                    event.stopPropagation();
                    
                    // 切换子菜单显示状态
                    const submenu = item.querySelector('.submenu');
                    if (submenu) {
                        const isExpanded = submenu.classList.toggle('show');
                        // 更新图标
                        toggleButton.querySelector('i').classList.toggle('bi-chevron-down', !isExpanded);
                        toggleButton.querySelector('i').classList.toggle('bi-chevron-up', isExpanded);
                    }
                });
            }
        });
    }
    
    // 响应屏幕尺寸变化
    const handleResize = function() {
        // 如果屏幕变为桌面宽度，确保移动菜单隐藏
        if (window.innerWidth >= 992 && mobileSidebar && mobileSidebar.classList.contains('show')) {
            mobileSidebar.classList.remove('show');
            bodyElement.classList.remove('mobile-menu-open');
            bodyElement.style.overflow = '';
        }
        
        // 调整图表大小（如果有的话）
        resizeCharts();
    };
    
    // 注册窗口大小改变事件
    window.addEventListener('resize', handleResize);
    
    // 初始化执行一次大小处理
    handleResize();
    
    // 自动调整图表大小（如果页面中有图表）
    function resizeCharts() {
        const charts = document.querySelectorAll('.chart-container canvas');
        if (charts.length > 0 && typeof Chart !== 'undefined') {
            charts.forEach(function(canvas) {
                if (canvas.chart) {
                    canvas.chart.resize();
                }
            });
        }
    }
    
    // 更新主题图标
    function updateThemeIcon(isDarkTheme) {
        if (themeToggle) {
            const icon = themeToggle.querySelector('i');
            if (icon) {
                if (isDarkTheme) {
                    icon.classList.remove('bi-moon');
                    icon.classList.add('bi-sun');
                } else {
                    icon.classList.remove('bi-sun');
                    icon.classList.add('bi-moon');
                }
            }
        }
    }
    
    // 表格响应式处理
    const tables = document.querySelectorAll('table:not(.table-responsive)');
    if (tables.length > 0) {
        tables.forEach(function(table) {
            // 如果表格不在.table-responsive容器中，添加一个容器
            if (!table.parentElement.classList.contains('table-responsive')) {
                const wrapper = document.createElement('div');
                wrapper.classList.add('table-responsive');
                table.parentNode.insertBefore(wrapper, table);
                wrapper.appendChild(table);
            }
        });
    }
    
    // 表单控件响应式增强
    const mobileFormControls = document.querySelectorAll('input[type="text"], input[type="email"], input[type="password"], input[type="number"], input[type="tel"], textarea, select');
    if (mobileFormControls.length > 0) {
        mobileFormControls.forEach(function(control) {
            // 在移动设备上放大字体以防止iOS缩放
            if (window.innerWidth < 768) {
                control.style.fontSize = '16px';
            }
            
            // 添加触摸优化类
            control.classList.add('touch-input');
        });
    }
    
    // 添加触摸按钮优化
    const buttonElements = document.querySelectorAll('button:not(.btn-icon), .btn');
    if (buttonElements.length > 0) {
        buttonElements.forEach(function(button) {
            button.classList.add('touch-button');
        });
    }
    
    // 滚动效果和性能优化
    const scrollableElements = document.querySelectorAll('.scrollable');
    if (scrollableElements.length > 0) {
        scrollableElements.forEach(function(element) {
            // 添加平滑滚动
            element.style.scrollBehavior = 'smooth';
            // 在移动设备上添加惯性滚动
            element.style.webkitOverflowScrolling = 'touch';
        });
    }
    
    // 懒加载图片
    if ('loading' in HTMLImageElement.prototype) {
        // 浏览器支持原生懒加载
        const lazyImages = document.querySelectorAll('img[loading="lazy"]');
        lazyImages.forEach(function(img) {
            img.src = img.dataset.src;
        });
    } else {
        // 回退方案: 使用IntersectionObserver进行懒加载
        const lazyImages = document.querySelectorAll('img.lazy');
        if (lazyImages.length > 0 && 'IntersectionObserver' in window) {
            const imageObserver = new IntersectionObserver(function(entries, observer) {
                entries.forEach(function(entry) {
                    if (entry.isIntersecting) {
                        const image = entry.target;
                        image.src = image.dataset.src;
                        image.classList.remove('lazy');
                        imageObserver.unobserve(image);
                    }
                });
            });
            
            lazyImages.forEach(function(image) {
                imageObserver.observe(image);
            });
        }
    }
    
    // 检测屏幕方向变化
    window.addEventListener('orientationchange', function() {
        // 重新调整布局元素
        setTimeout(handleResize, 200); // 给一点时间让浏览器完成方向变化
    });
    
    // 检测系统暗色模式偏好变化
    if (window.matchMedia) {
        const darkModeMediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
        
        // 添加变化监听
        try {
            // Chrome & Firefox
            darkModeMediaQuery.addEventListener('change', function(e) {
                if (!localStorage.getItem('theme')) {
                    // 仅当用户未明确设置主题时才响应系统变化
                    bodyElement.classList.toggle('dark-theme', e.matches);
                    updateThemeIcon(e.matches);
                }
            });
        } catch (e1) {
            try {
                // Safari
                darkModeMediaQuery.addListener(function(e) {
                    if (!localStorage.getItem('theme')) {
                        bodyElement.classList.toggle('dark-theme', e.matches);
                        updateThemeIcon(e.matches);
                    }
                });
            } catch (e2) {
                console.error('浏览器不支持媒体查询监听器', e2);
            }
        }
        
        // 初始化检查系统暗色模式
        if (!localStorage.getItem('theme') && darkModeMediaQuery.matches) {
            bodyElement.classList.add('dark-theme');
            updateThemeIcon(true);
        }
    }
    
    // 初始化工具提示（如果有Bootstrap）
    if (typeof $ !== 'undefined' && typeof $.fn.tooltip !== 'undefined') {
        $('[data-toggle="tooltip"]').tooltip();
    }
    
    // 初始化下拉菜单（如果没有使用Bootstrap的JS）
    const dropdownToggles = document.querySelectorAll('.dropdown-toggle');
    if (dropdownToggles.length > 0) {
        dropdownToggles.forEach(function(toggle) {
            toggle.addEventListener('click', function(event) {
                event.preventDefault();
                event.stopPropagation();
                
                const dropdownMenu = this.nextElementSibling;
                if (dropdownMenu && dropdownMenu.classList.contains('dropdown-menu')) {
                    dropdownMenu.classList.toggle('show');
                }
            });
        });
        
        // 点击外部关闭所有下拉菜单
        document.addEventListener('click', function() {
            const openDropdowns = document.querySelectorAll('.dropdown-menu.show');
            openDropdowns.forEach(function(dropdown) {
                dropdown.classList.remove('show');
            });
        });
    }
    
    // 添加toast消息关闭功能
    const toasts = document.querySelectorAll('.toast');
    if (toasts.length > 0) {
        toasts.forEach(function(toast) {
            const closeBtn = toast.querySelector('.close');
            if (closeBtn) {
                closeBtn.addEventListener('click', function() {
                    toast.classList.remove('show');
                    // 300ms后移除，与CSS动画同步
                    setTimeout(function() {
                        toast.remove();
                    }, 300);
                });
            }
            
            // 如果有自动关闭属性
            if (toast.dataset.autohide !== 'false') {
                const delay = parseInt(toast.dataset.delay || 5000);
                setTimeout(function() {
                    toast.classList.remove('show');
                    setTimeout(function() {
                        toast.remove();
                    }, 300);
                }, delay);
            }
        });
    }
}); 