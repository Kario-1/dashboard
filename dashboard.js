// dashboard.js

document.addEventListener('DOMContentLoaded', () => {
    const statsGrid = document.getElementById('statsGrid');
    const dashboardApiUrl = 'https://enabled-early-vulture.ngrok-free.app/admin/dashboard/stats';

    // عناصر الـ Modal
    const dataModal = document.getElementById('dataModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalControls = document.getElementById('modalControls');
    const modalBody = document.getElementById('modalBody');
    const closeButton = document.querySelector('.close-button');

    // URLs للـ APIs التفصيلية
    const revenueApiUrl = 'https://enabled-early-vulture.ngrok-free.app/admin/revenue';
    const usersApiBaseUrl = 'https://enabled-early-vulture.ngrok-free.app/admin/users';
    const subscriptionsApiBaseUrl = 'https://enabled-early-vulture.ngrok-free.app/admin/subscriptions';
    const subscriptionByIdApiBaseUrl = 'https://enabled-early-vulture.ngrok-free.app/admin/'; // هنضيف user_id/subscription ليها

    // دالة لجلب البيانات وعرضها (زي ما هي)
    async function fetchAndDisplayDashboardStats() {
        const accessToken = localStorage.getItem('accessToken');
        const tokenType = localStorage.getItem('tokenType');

        if (!accessToken || !tokenType) {
            console.error('Access Token or Token Type not found in localStorage. Please log in.');
            displayErrorMessage('عفواً، لا يمكن تحميل البيانات. برجاء تسجيل الدخول أولاً.');
            removeSkeletonCards();
            return;
        }

        try {
            const response = await fetch(dashboardApiUrl, {
                method: 'GET',
                headers: {
                    'accept': 'application/json',
                    'Authorization': `${tokenType} ${accessToken}`,
                    'ngrok-skip-browser-warning': 'true'
                }
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Failed to fetch dashboard stats:', response.status, errorText);
                displayErrorMessage(`حدث خطأ أثناء تحميل الإحصائيات: ${response.status} - ${errorText.substring(0, 100)}...`);
                removeSkeletonCards();
                return;
            }

            const stats = await response.json();
            displayStats(stats);
        } catch (error) {
            console.error('Error fetching dashboard stats:', error);
            displayErrorMessage(`حدث خطأ غير متوقع: ${error.message}.`);
            removeSkeletonCards();
        }
    }

    // دالة لعرض البيانات في الكروت (هنضيف عليها event listeners)
    function displayStats(stats) {
        removeSkeletonCards();
        statsGrid.innerHTML = '';

        const statItems = [
            { label: 'إجمالي الحسابات', key: 'total_accounts', icon: 'fas fa-users', action: 'showUsers' }, // إضافة action
            { label: 'إجمالي المستخدمين', key: 'total_users', icon: 'fas fa-user' }, 
            { label: 'إجمالي المحامين', key: 'total_lawyers', icon: 'fas fa-gavel' },
            { label: 'إجمالي الشركات', key: 'total_companies', icon: 'fas fa-building' },
            { label: 'الاشتراكات النشطة', key: 'total_active_subscriptions', icon: 'fas fa-credit-card', action: 'showSubscriptions' }, // إضافة action
            { label: 'إجمالي الإيرادات', key: 'total_revenue', icon: 'fas fa-dollar-sign', isCurrency: true, action: 'showRevenue' }, // إضافة action
            { label: 'الإيرادات الشهرية', key: 'monthly_revenue', icon: 'fas fa-money-bill-wave', isCurrency: true },
            { label: 'المحادثات النشطة', key: 'active_chats', icon: 'fas fa-comments' },
            { label: 'إجمالي الرسائل', key: 'total_messages', icon: 'fas fa-envelope' },
        ];

        statItems.forEach(item => {
            const value = stats[item.key] !== undefined ? stats[item.key] : 'N/A';
            const formattedValue = item.isCurrency ? `${value.toLocaleString('ar-EG', { style: 'currency', currency: 'EGP' })}` : value.toLocaleString('ar-EG');

            const card = document.createElement('div');
            card.classList.add('stat-card');
            card.innerHTML = `
                <i class="${item.icon} stat-icon"></i>
                <div class="stat-label">${item.label}</div>
                <div class="stat-value">${formattedValue}</div>
            `;
            // إضافة Event Listener إذا كان الكارت له action
            if (item.action) {
                card.classList.add('clickable-card'); // عشان نضيف عليها تنسيق خاص
                card.addEventListener('click', () => handleCardClick(item.action));
            }
            statsGrid.appendChild(card);
        });
    }

    // دالة للتعامل مع ضغطات الكروت
    async function handleCardClick(action) {
        modalControls.innerHTML = ''; // تفريغ عناصر التحكم
        modalBody.innerHTML = '<div class="loading-spinner"></div>'; // عرض Spinner للتحميل
        dataModal.style.display = 'block'; // فتح الـ Modal

        if (action === 'showRevenue') {
            modalTitle.textContent = 'تفاصيل الإيرادات';
            await fetchAndDisplayRevenue();
        } else if (action === 'showUsers') {
            modalTitle.textContent = 'تفاصيل المستخدمين';
            await promptForLimitAndFetchUsers();
        } else if (action === 'showSubscriptions') {
            modalTitle.textContent = 'تفاصيل الاشتراكات';
            await promptForUserIdOrAllSubscriptions();
        }
    }

    // --- دوال خاصة بجلب وعرض البيانات التفصيلية ---

    async function fetchData(url) {
        const accessToken = localStorage.getItem('accessToken');
        const tokenType = localStorage.getItem('tokenType');
        try {
            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'accept': 'application/json',
                    'Authorization': `${tokenType} ${accessToken}`,
                    'ngrok-skip-browser-warning': 'true'
                }
            });
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Error ${response.status}: ${errorText.substring(0, 100)}...`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching data:', error);
            modalBody.innerHTML = `<div class="error-message">حدث خطأ أثناء جلب البيانات: ${error.message}</div>`;
            return null;
        }
    }

    // عرض الإيرادات
    async function fetchAndDisplayRevenue() {
        modalBody.innerHTML = '<div class="loading-spinner"></div>';
        const revenueData = await fetchData(revenueApiUrl);
        if (revenueData) {
            displayListInModal(revenueData, 'الإيرادات', (item) => `
                <div class="list-item-card">
                    <p><strong>التاريخ:</strong> ${new Date(item.date).toLocaleString('ar-EG')}</p>
                    <p><strong>المبلغ:</strong> ${item.amount.toLocaleString('ar-EG', { style: 'currency', currency: 'EGP' })}</p>
                    <p><strong>نوع الاشتراك:</strong> ${item.subscription_type}</p>
                    <p><strong>معرف المستخدم:</strong> ${item.user_id}</p>
                </div>
            `);
        }
    }

    // عرض المستخدمين
    async function promptForLimitAndFetchUsers() {
        modalBody.innerHTML = `
            <p>هل تريد تحديد عدد المستخدمين المعروضين؟</p>
            <input type="number" id="limitInput" placeholder="أدخل العدد (اختياري)">
            <button id="fetchUsersBtn">عرض</button>
        `;
        const fetchUsersBtn = document.getElementById('fetchUsersBtn');
        fetchUsersBtn.onclick = async () => {
            const limit = document.getElementById('limitInput').value;
            const url = limit ? `${usersApiBaseUrl}?skip=0&limit=${limit}` : usersApiBaseUrl;
            modalBody.innerHTML = '<div class="loading-spinner"></div>';
            const usersData = await fetchData(url);
            if (usersData) {
                displayListInModal(usersData, 'المستخدمون', (item) => `
                    <div class="list-item-card">
                        <p><strong>معرف المستخدم:</strong> ${item.user_id}</p>
                        <p><strong>البريد الإلكتروني:</strong> ${item.email}</p>
                        <p><strong>الدور:</strong> ${item.role}</p>
                        <p><strong>حالة الاشتراك:</strong> ${item.subscription_status || 'لا يوجد'}</p>
                        <p><strong>تاريخ انتهاء الاشتراك:</strong> ${item.subscription_end_date ? new Date(item.subscription_end_date).toLocaleDateString('ar-EG') : 'لا يوجد'}</p>
                        <p><strong>إجمالي الرسائل:</strong> ${item.total_messages}</p>
                        <p><strong>الرسائل المتبقية:</strong> ${item.remaining_messages || 'لا يوجد'}</p>
                        <p><strong>تاريخ الإنشاء:</strong> ${new Date(item.created_at).toLocaleDateString('ar-EG')}</p>
                    </div>
                `);
            }
        };
    }

    // عرض الاشتراكات (سواء لـ ID معين أو كلها)
    async function promptForUserIdOrAllSubscriptions() {
        modalBody.innerHTML = `
            <p>أدخل معرف المستخدم لعرض اشتراكاته أو اترك الحقل فارغاً لعرض كل الاشتراكات:</p>
            <input type="number" id="userIdInput" placeholder="معرف المستخدم (اختياري)">
            <button id="fetchSubscriptionsBtn">عرض</button>
        `;
        const fetchSubscriptionsBtn = document.getElementById('fetchSubscriptionsBtn');
        fetchSubscriptionsBtn.onclick = async () => {
            const userId = document.getElementById('userIdInput').value;
            let url;
            if (userId) {
                url = `${subscriptionByIdApiBaseUrl}${userId}/subscription`; // API لعرض اشتراك مستخدم واحد
            } else {
                url = `${subscriptionsApiBaseUrl}?skip=0&limit=100`; // API لعرض كل الاشتراكات (ممكن تزود الـ limit)
            }
            modalBody.innerHTML = '<div class="loading-spinner"></div>';
            const subscriptionsData = await fetchData(url);

            if (subscriptionsData) {
                // لو بيرجع أوبجيكت واحد (اشتراك مستخدم واحد) حوله لـ array عشان displayListInModal تشتغل صح
                const dataToShow = Array.isArray(subscriptionsData) ? subscriptionsData : [subscriptionsData];
                
                displayListInModal(dataToShow, 'الاشتراكات', (item) => `
                    <div class="list-item-card">
                        <p><strong>معرف الاشتراك:</strong> ${item.sub_id}</p>
                        <p><strong>معرف المستخدم:</strong> ${item.user_id}</p>
                        <p><strong>اسم الخطة:</strong> ${item.plan_name}</p>
                        <p><strong>السعر:</strong> ${item.price.toLocaleString('ar-EG', { style: 'currency', currency: 'EGP' })}</p>
                        <p><strong>تاريخ البدء:</strong> ${new Date(item.start_date).toLocaleDateString('ar-EG')}</p>
                        <p><strong>تاريخ الانتهاء:</strong> ${new Date(item.end_date).toLocaleDateString('ar-EG')}</p>
                        <p><strong>الحالة:</strong> ${item.status}</p>
                    </div>
                `);
            }
        };
    }


    // دالة مساعدة لعرض البيانات في شكل قائمة من الكروت داخل الـ Modal
    function displayListInModal(data, title, itemFormatter) {
        modalTitle.textContent = title;
        modalBody.innerHTML = ''; // تفريغ المحتوى
        if (data && data.length > 0) {
            data.forEach(item => {
                const itemHtml = itemFormatter(item);
                modalBody.insertAdjacentHTML('beforeend', itemHtml);
            });
        } else {
            modalBody.innerHTML = `<div class="info-message">لا توجد بيانات لعرضها.</div>`;
        }
    }

    // دالة لعرض رسالة خطأ (زي ما هي)
    function displayErrorMessage(message) {
        statsGrid.innerHTML = `<div class="error-message">${message}</div>`;
    }

    // دالة لإزالة الكروت الهيكلية (زي ما هي)
    function removeSkeletonCards() {
        const skeletons = document.querySelectorAll('.skeleton-card');
        skeletons.forEach(skeleton => skeleton.remove());
    }

    // --- ربط أحداث الـ Modal ---
    closeButton.addEventListener('click', () => {
        dataModal.style.display = 'none';
    });

    window.addEventListener('click', (event) => {
        if (event.target === dataModal) {
            dataModal.style.display = 'none';
        }
    });

    // استدعاء الدالة عند تحميل الصفحة
    fetchAndDisplayDashboardStats();
});