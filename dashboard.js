// dashboard.js

document.addEventListener('DOMContentLoaded', () => {
    const statsGrid = document.getElementById('statsGrid');
    const dashboardApiUrl = 'https://enabled-early-vulture.ngrok-free.app/admin/dashboard/stats';

    // Modal elements
    const dataModal = document.getElementById('dataModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalControls = document.getElementById('modalControls');
    const modalBody = document.getElementById('modalBody');
    const closeButton = document.querySelector('.close-button');

    // Detailed APIs URLs
    const revenueApiUrl = 'https://enabled-early-vulture.ngrok-free.app/admin/revenue';
    const usersApiBaseUrl = 'https://enabled-early-vulture.ngrok-free.app/admin/users'; // هذا الـ API سيعود بكل أنواع الحسابات، وسنقوم بالفلترة في الـ frontend
    const subscriptionsApiBaseUrl = 'https://enabled-early-vulture.ngrok-free.app/admin/subscriptions';
    const subscriptionByIdApiBaseUrl = 'https://enabled-early-vulture.ngrok-free.app/admin/'; // We'll append user_id/subscription to it

    // Function to fetch and display dashboard stats
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

    // Function to display data in cards
    function displayStats(stats) {
        removeSkeletonCards();
        statsGrid.innerHTML = '';

        const statItems = [
            { label: 'إجمالي الحسابات', key: 'total_accounts', icon: 'fas fa-users', action: 'showAllAccounts' },
            { label: 'إجمالي المستخدمين', key: 'total_users', icon: 'fas fa-user', action: 'showOnlyUsers' }, // إضافة action
            { label: 'إجمالي المحامين', key: 'total_lawyers', icon: 'fas fa-gavel', action: 'showOnlyLawyers' }, // إضافة action
            { label: 'إجمالي الشركات', key: 'total_companies', icon: 'fas fa-building', action: 'showOnlyCompanies' }, // إضافة action
            { label: 'المشتركون النشطون', key: 'total_active_subscriptions', icon: 'fas fa-user-check', action: 'showActiveSubscriptions' },
            { label: 'إجمالي الاشتراكات', key: 'total_subscriptions', icon: 'fas fa-credit-card', action: 'showAllSubscriptions' },
            { label: 'إجمالي الإيرادات', key: 'total_revenue', icon: 'fas fa-dollar-sign', isCurrency: true, action: 'showRevenue' },
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
            if (item.action) {
                card.classList.add('clickable-card');
                card.addEventListener('click', () => handleCardClick(item.action));
            }
            statsGrid.appendChild(card);
        });
    }

    // Function to handle card clicks
    async function handleCardClick(action) {
        modalControls.innerHTML = '';
        modalBody.innerHTML = '<div class="loading-spinner"></div>';
        dataModal.style.display = 'block';

        if (action === 'showRevenue') {
            modalTitle.textContent = 'تفاصيل الإيرادات';
            await fetchAndDisplayRevenue();
        } else if (action === 'showAllAccounts') {
            modalTitle.textContent = 'تفاصيل جميع الحسابات';
            await promptForSpecificUsersOrAll('all'); // 'all' indicates no specific role filter
        } else if (action === 'showOnlyUsers') {
            modalTitle.textContent = 'تفاصيل المستخدمين';
            await promptForSpecificUsersOrAll('user'); // 'user' to filter for users
        } else if (action === 'showOnlyLawyers') {
            modalTitle.textContent = 'تفاصيل المحامين';
            await promptForSpecificUsersOrAll('lawyer'); // 'lawyer' to filter for lawyers
        } else if (action === 'showOnlyCompanies') {
            modalTitle.textContent = 'تفاصيل الشركات';
            await promptForSpecificUsersOrAll('company'); // 'company' to filter for companies
        } else if (action === 'showAllSubscriptions') {
            modalTitle.textContent = 'تفاصيل إجمالي الاشتراكات';
            await promptForUserIdOrAllSubscriptions(false);
        } else if (action === 'showActiveSubscriptions') {
            modalTitle.textContent = 'تفاصيل المشتركين النشطين';
            await promptForUserIdOrAllSubscriptions(true);
        }
    }

    // --- Detailed data fetching and display functions ---

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

    // Display Revenue (as is)
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

    // New unified function to prompt for specific user type ID or display all of that type
    async function promptForSpecificUsersOrAll(userType = 'all') { // userType can be 'all', 'user', 'lawyer', 'company'
        let promptText = '';
        let modalTitleText = '';
        let idKey = ''; // Key for the ID field (e.g., user_id, lawyer_id, company_id)

        switch (userType) {
            case 'user':
                promptText = 'أدخل معرف المستخدم للبحث أو اترك الحقل فارغاً لعرض كل المستخدمين:';
                modalTitleText = 'تفاصيل المستخدمين';
                idKey = 'user_id';
                break;
            case 'lawyer':
                promptText = 'أدخل معرف المحامي للبحث أو اترك الحقل فارغاً لعرض كل المحامين:';
                modalTitleText = 'تفاصيل المحامين';
                idKey = 'lawyer_id';
                break;
            case 'company':
                promptText = 'أدخل معرف الشركة للبحث أو اترك الحقل فارغاً لعرض كل الشركات:';
                modalTitleText = 'تفاصيل الشركات';
                idKey = 'company_id';
                break;
            case 'all':
            default:
                promptText = 'أدخل معرف الحساب (ID) للبحث أو اترك الحقل فارغاً لعرض كل الحسابات:';
                modalTitleText = 'تفاصيل جميع الحسابات';
                idKey = 'user_id'; // Default to user_id for generic accounts
                break;
        }

        modalTitle.textContent = modalTitleText;
        modalBody.innerHTML = `
            <p>${promptText}</p>
            <input type="text" id="limitInput" placeholder="أدخل المعرف (اختياري)">
            <button id="fetchUsersBtn">عرض</button>
        `;

        const fetchBtn = document.getElementById('fetchUsersBtn');
        fetchBtn.onclick = async () => {
            const searchId = document.getElementById('limitInput').value.trim();
            modalBody.innerHTML = '<div class="loading-spinner"></div>';

            const allUsersData = await fetchData(usersApiBaseUrl); // Fetch all data from the common users API

            if (allUsersData) {
                let dataToShow = allUsersData;

                // Filter by role if a specific userType is requested (not 'all')
                if (userType !== 'all') {
                    dataToShow = dataToShow.filter(item => item.role && item.role.toLowerCase() === userType);
                }

                // Filter by ID if searchId is provided
                if (searchId) {
                    dataToShow = dataToShow.filter(item => item[idKey] && item[idKey].toString() === searchId);
                    if (dataToShow.length === 0) {
                        modalBody.innerHTML = `<div class="info-message">لا توجد ${modalTitleText.replace('تفاصيل', '').trim()} مطابقة لـ ID: ${searchId}.</div>`;
                        return;
                    }
                }

                displayListInModal(dataToShow, modalTitleText, (item) => `
                    <div class="list-item-card">
                        <p><strong>معرف الحساب:</strong> ${item.user_id || item.lawyer_id || item.company_id || 'N/A'}</p>
                        <p><strong>البريد الإلكتروني:</strong> ${item.email || 'N/A'}</p>
                        <p><strong>الدور:</strong> ${item.role || 'N/A'}</p>
                        <p><strong>حالة الاشتراك:</strong> ${item.subscription_status || 'لا يوجد'}</p>
                        <p><strong>تاريخ انتهاء الاشتراك:</strong> ${item.subscription_end_date ? new Date(item.subscription_end_date).toLocaleDateString('ar-EG') : 'لا يوجد'}</p>
                        <p><strong>إجمالي الرسائل:</strong> ${item.total_messages || 'N/A'}</p>
                        <p><strong>الرسائل المتبقية:</strong> ${item.remaining_messages || 'لا يوجد'}</p>
                        <p><strong>تاريخ الإنشاء:</strong> ${new Date(item.created_at).toLocaleDateString('ar-EG')}</p>
                    </div>
                `);
            }
        };
    }

    // Display Subscriptions (either by user ID or all, with active filter)
    async function promptForUserIdOrAllSubscriptions(activeOnly = false) {
        let titlePrompt = activeOnly ? 'أدخل معرف المستخدم لعرض اشتراكاته النشطة أو اترك الحقل فارغاً لعرض كل الاشتراكات النشطة:' : 'أدخل معرف المستخدم لعرض اشتراكاته أو اترك الحقل فارغاً لعرض كل الاشتراكات:';
        modalBody.innerHTML = `
            <p>${titlePrompt}</p>
            <input type="number" id="userIdInput" placeholder="معرف المستخدم (اختياري)">
            <button id="fetchSubscriptionsBtn">عرض</button>
        `;
        const fetchSubscriptionsBtn = document.getElementById('fetchSubscriptionsBtn');
        fetchSubscriptionsBtn.onclick = async () => {
            const userId = document.getElementById('userIdInput').value;
            let subscriptionsData;

            modalBody.innerHTML = '<div class="loading-spinner"></div>';

            if (userId) {
                // If a user ID is provided, fetch specific user subscription(s)
                const url = `${subscriptionByIdApiBaseUrl}${userId}/subscription`;
                subscriptionsData = await fetchData(url);
                // Ensure subscriptionsData is an array for displayListInModal
                subscriptionsData = Array.isArray(subscriptionsData) ? subscriptionsData : (subscriptionsData ? [subscriptionsData] : []);
            } else {
                // If no user ID, fetch all subscriptions
                const url = `${subscriptionsApiBaseUrl}?skip=0&limit=200`; // Increased limit to fetch more subscriptions
                subscriptionsData = await fetchData(url);
                subscriptionsData = Array.isArray(subscriptionsData) ? subscriptionsData : []; // Ensure it's an array
            }
            
            if (subscriptionsData) {
                let dataToShow = subscriptionsData;

                if (activeOnly) {
                    // Filter for active subscriptions if activeOnly is true
                    dataToShow = dataToShow.filter(sub => sub.status && sub.status.toLowerCase() === 'active');
                }

                const title = activeOnly ? 'الاشتراكات النشطة' : 'إجمالي الاشتراكات';
                displayListInModal(dataToShow, title, (item) => `
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

    // Helper function to display data as a list of cards inside the modal
    function displayListInModal(data, title, itemFormatter) {
        modalTitle.textContent = title;
        modalBody.innerHTML = ''; // Clear content
        if (data && data.length > 0) {
            data.forEach(item => {
                const itemHtml = itemFormatter(item);
                modalBody.insertAdjacentHTML('beforeend', itemHtml);
            });
        } else {
            modalBody.innerHTML = `<div class="info-message">لا توجد بيانات لعرضها.</div>`;
        }
    }

    // Function to display an error message
    function displayErrorMessage(message) {
        statsGrid.innerHTML = `<div class="error-message">${message}</div>`;
    }

    // Function to remove skeleton cards
    function removeSkeletonCards() {
        const skeletons = document.querySelectorAll('.skeleton-card');
        skeletons.forEach(skeleton => skeleton.remove());
    }

    // --- Modal event binding ---
    closeButton.addEventListener('click', () => {
        dataModal.style.display = 'none';
    });

    window.addEventListener('click', (event) => {
        if (event.target === dataModal) {
            dataModal.style.display = 'none';
        }
    });

    // Call the function on page load
    fetchAndDisplayDashboardStats();
});
