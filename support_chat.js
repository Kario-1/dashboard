document.addEventListener('DOMContentLoaded', () => {
    // تحديد العناصر الأساسية
    const chatHeaderName = document.getElementById('chatHeaderName');
    const chatHeaderStatus = document.getElementById('chatHeaderStatus');
    const endChatButton = document.getElementById('endChatButton');
    const chatMessagesContainer = document.getElementById('chatMessages');
    const messageInput = document.getElementById('messageInput');
    const sendMessageButton = document.getElementById('sendMessageButton');
    const conversationSearchInput = document.getElementById('conversationSearchInput');
    const searchChatsButton = document.getElementById('searchChatsButton');
    const toggleEndedChatsButton = document.getElementById('toggleEndedChatsButton');

    const clientsList = document.getElementById('clientsList');
    const lawyersList = document.getElementById('lawyersList');
    const companiesList = document.getElementById('companiesList');
    const categoryTitles = document.querySelectorAll('.category-title');

    const unreadClientsBadge = document.getElementById('unreadClients');
    const unreadLawyersBadge = document.getElementById('unreadLawyers');
    const unreadCompaniesBadge = document.getElementById('unreadCompanies');

    const conversationBar = document.getElementById('conversation-bar');
    const openConversationSidebar = document.getElementById('openConversationSidebar'); // أيقونة فتح الشريط من السايد بار الأيسر (للدسكتوب)
    const closeConversationBar = document.getElementById('closeConversationBar');
    const resizeHandle = document.querySelector('#conversation-bar .resize-handle');

    const infoIcon = document.getElementById('infoIcon');
    const infoModal = document.getElementById('infoModal');
    const closeInfoModal = document.getElementById('closeInfoModal');
    const modalInfoContent = document.getElementById('modalInfoContent');
    let infoData = null; // لتخزين بيانات المحادثة للمودال

    let currentChatId = null; // ID للمحادثة الحالية اللي مفتوحة
    let allChatsData = []; // لتخزين جميع المحادثات اللي جاية من الـ API
    let filteredChatsData = []; // للمحادثات بعد تطبيق البحث
    let showEndedChats = false; // الحالة الأولية: إخفاء المحادثات المنتهية (لأن الزر يقول "إظهار المنتهية")

    // URLs للـ APIs
    const supportChatsApiUrl = 'https://enabled-early-vulture.ngrok-free.app/admin/support/chats';
    const sendMessageApiUrl = 'https://enabled-early-vulture.ngrok-free.app/support/chat';
    const fullChatHistoryApiUrl = 'https://enabled-early-vulture.ngrok-free.app/chats';
    const endChatApiUrl = 'https://enabled-early-vulture.ngrok-free.app/chats';

    // Helper Functions
    // --------------------------------------------------------------------------------------------------------------------
    function getAuthHeaders() {
        const accessToken = localStorage.getItem('accessToken');
        const tokenType = localStorage.getItem('tokenType');
        if (!accessToken || !tokenType) {
            console.error('No authentication token found. Please log in.');
            displayInitialMessage('عفواً، لا يمكن تحميل البيانات. برجاء تسجيل الدخول أولاً.', true);
            return null;
        }
        return {
            'accept': 'application/json',
            'Authorization': `${tokenType} ${accessToken}`,
            'ngrok-skip-browser-warning': 'true',
            'Content-Type': 'application/json'
        };
    }

    function displayInitialMessage(message, isError = false) {
        chatMessagesContainer.innerHTML = `<div class="initial-message" style="color: ${isError ? 'red' : '#777'};">${message}</div>`;
        chatHeaderName.textContent = 'اختر محادثة';
        chatHeaderStatus.textContent = '';
        endChatButton.style.display = 'none';
        messageInput.disabled = true;
        sendMessageButton.disabled = true;
    }

    function formatDateTime(isoString) {
        if (!isoString) return '';
        const date = new Date(isoString);
        if (isNaN(date.getTime())) return '';

        const now = new Date();
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        const messageDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());

        if (messageDate.getTime() === today.getTime()) {
            return date.toLocaleTimeString('ar-EG', { hour: '2-digit', minute: '2-digit' });
        } else {
            return date.toLocaleDateString('ar-EG', { day: '2-digit', month: '2-digit', year: 'numeric' });
        }
    }

    function formatMessageContent(content) {
        if (!content) return '';
        let formattedText = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formattedText = formattedText.replace(/_([^_]+)_/g, '<em>$1</em>');
        formattedText = formattedText.replace(/~([^~]+)~/g, '<s>$1</s>');
        formattedText = formattedText.replace(/\\n/g, '<br>');
        formattedText = formattedText.replace(/(https?:\/\/[^\s]+)/g, (url) => `<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`);
        return formattedText;
    }

    // New helper function to truncate string
    function truncateString(str, num) {
        if (!str) return '';
        if (str.length <= num) {
            return str;
        }
        return str.slice(0, num) + '...';
    }

    function appendMessage(msg) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message');

        const isMyMessage = msg.role === 'admin';
        messageElement.classList.add(isMyMessage ? 'user-message' : 'received-message');

        messageElement.innerHTML = `
            <div class="message-content">${formatMessageContent(msg.content)}</div>
            <div class="message-timestamp">
                <span>${formatDateTime(msg.timestamp)}</span>
                ${isMyMessage ? `<span class="read-status ${msg.is_read ? 'read' : 'unread'}">
                    <i class="fas fa-check-double"></i>
                </span>` : ''}
            </div>
        `;
        chatMessagesContainer.appendChild(messageElement);
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
    }

    // --------------------------------------------------------------------------------------------------------------------
    // Fetch and render chat list
    async function fetchAndRenderChatList() {
        const headers = getAuthHeaders();
        if (!headers) {
            displayInitialMessage('فشل تحميل المحادثات: خطأ في المصادقة.', true);
            return;
        }

        try {
            const response = await fetch(supportChatsApiUrl, { headers });
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to fetch chats: ${response.status} - ${errorText.substring(0, 100)}...`);
            }
            allChatsData = await response.json();
            if (!Array.isArray(allChatsData)) {
                console.warn("API response is not an array, initializing as empty array.");
                allChatsData = [];
            }
            applyFiltersAndRender();
        } catch (error) {
            console.error('Error fetching chat list:', error);
            displayInitialMessage(`خطأ في تحميل قائمة المحادثات: ${error.message}`, true);
        }
    }

    function applyFiltersAndRender() {
        const searchTerm = conversationSearchInput.value.toLowerCase().trim();
        filteredChatsData = allChatsData.filter(chat => {
            // الوصول الصحيح لاسم المستخدم والبريد الإلكتروني من الـ API `admin/support/chats`
            const userName = chat.user_name ? chat.user_name.toLowerCase() : '';
            const userEmail = chat.user_email ? chat.user_email.toLowerCase() : '';
            const chatIdStr = chat.chat_id ? chat.chat_id.toString() : '';
            const lastMessage = chat.last_message ? chat.last_message.toLowerCase() : '';

            const matchesSearch = userName.includes(searchTerm) ||
                                  userEmail.includes(searchTerm) ||
                                  chatIdStr.includes(searchTerm) ||
                                  lastMessage.includes(searchTerm);

            const matchesEndedFilter = showEndedChats || (chat.status === 'active');
            return matchesSearch && matchesEndedFilter;
        });
        renderChatList(filteredChatsData);
    }

    function renderChatList(chats) {
        clientsList.innerHTML = '';
        lawyersList.innerHTML = '';
        companiesList.innerHTML = '';

        let unreadClients = 0;
        let unreadLawyers = 0;
        let unreadCompanies = 0;

        // فرز المحادثات وتصنيفها
        chats.forEach(chat => {
            const chatItem = document.createElement('div');
            chatItem.classList.add('conversation-item');
            if (chat.chat_id === currentChatId) {
                chatItem.classList.add('active');
            }
            if (chat.status === 'ended') {
                chatItem.classList.add('ended-chat');
            }
            chatItem.dataset.chatId = chat.chat_id;

            let unreadCountHtml = '';
            if (chat.unread_count > 0) {
                unreadCountHtml = `<span class="unread-count">${chat.unread_count}</span>`;
                if (chat.user_type === 'client') unreadClients += 1; // تصحيح جمع العداد
                else if (chat.user_type === 'lawyer') unreadLawyers += +1; // تصحيح جمع العداد
                else if (chat.user_type === 'company') unreadCompanies += +1; // تصحيح جمع العداد
            }

            // عرض أول 20 حرف فقط من آخر رسالة
            const displayedLastMessage = truncateString(chat.last_message, 22);

            chatItem.innerHTML = `
                <div class="user-info">
                    <h5>${chat.user_name || 'مستخدم غير معروف'}</h5>
                    <p>${displayedLastMessage || 'لا توجد رسائل بعد'}</p>
                </div>
                <div class="time-unread">
                    <div class="time">${formatDateTime(chat.last_message_time)}</div>
                    ${unreadCountHtml}
                </div>
                <div class="status-indicator ${chat.status === 'active' ? 'active' : 'inactive'}" title="محادثة ${chat.status === 'active' ? 'نشطة' : 'منتهية'}"></div>
            `;

            chatItem.addEventListener('click', () => openChat(chat.chat_id));

            if (chat.user_type === 'client') {
                clientsList.appendChild(chatItem);
            } else if (chat.user_type === 'lawyer') {
                lawyersList.appendChild(chatItem);
            } else if (chat.user_type === 'company') {
                companiesList.appendChild(chatItem);
            }
        });

        // تحديث شارات الرسائل غير المقروءة وإخفائها إذا كانت صفر
        unreadClientsBadge.textContent = unreadClients > 0 ? unreadClients : '';
        unreadClientsBadge.style.display = unreadClients > 0 ? 'inline-block' : 'none';

        unreadLawyersBadge.textContent = unreadLawyers > 0 ? unreadLawyers : '';
        unreadLawyersBadge.style.display = unreadLawyers > 0 ? 'inline-block' : 'none';

        unreadCompaniesBadge.textContent = unreadCompanies > 0 ? unreadCompanies : '';
        unreadCompaniesBadge.style.display = unreadCompanies > 0 ? 'inline-block' : 'none';

        // رسالة "لا توجد محادثات" إذا كانت القائمة فارغة
        if (clientsList.children.length === 0) clientsList.innerHTML = '<div class="empty-list">لا توجد محادثات.</div>';
        if (lawyersList.children.length === 0) lawyersList.innerHTML = '<div class="empty-list">لا توجد محادثات.</div>';
        if (companiesList.children.length === 0) companiesList.innerHTML = '<div class="empty-list">لا توجد محادثات.</div>';
    }


    async function openChat(chatId) {


        currentChatId = chatId;
        displayInitialMessage('جارٍ تحميل الرسائل...');

        document.querySelectorAll('.conversation-item').forEach(item => item.classList.remove('active'));
        const selectedItem = document.querySelector(`.conversation-item[data-chat-id="${chatId}"]`);
        if (selectedItem) {
            selectedItem.classList.add('active');
            const unreadCountSpan = selectedItem.querySelector('.unread-count');
            if (unreadCountSpan) {
                const chatInAllData = allChatsData.find(c => c.chat_id === chatId);
                if (chatInAllData) {
                    // تصحيح: استخدام chatInAllData لتحديث العدادات
                    if (chatInAllData.user_type === 'client') unreadClientsBadge.textContent = Math.max(0, parseInt(unreadClientsBadge.textContent || 0) - chatInAllData.unread_count);
                    if (chatInAllData.user_type === 'lawyer') unreadLawyersBadge.textContent = Math.max(0, parseInt(unreadLawyersBadge.textContent || 0) - chatInAllData.unread_count);
                    if (chatInAllData.user_type === 'company') unreadCompaniesBadge.textContent = Math.max(0, parseInt(unreadCompaniesBadge.textContent || 0) - chatInAllData.unread_count);
                    chatInAllData.unread_count = 0; // إعادة تعيين عدد الرسائل غير المقروءة في البيانات
                    unreadCountSpan.remove(); // إزالة الـ badge من الـ DOM بعد فتح المحادثة
                }
            }
        }

        const headers = getAuthHeaders();
        if (!headers) return;

        // جلب تفاصيل المحادثة من allChatsData لملء الهيدر والمودال
        const chatDetailsFromList = allChatsData.find(chat => chat.chat_id === chatId);

        // إذا لم يتم العثور على بيانات المحادثة في القائمة، اعرض خطأ
        if (!chatDetailsFromList) {
            console.error(`Chat with ID ${chatId} not found in allChatsData.`);
            displayInitialMessage('تعذر العثور على معلومات المحادثة.', true);
            return;
        }

        try {
            const response = await fetch(`${fullChatHistoryApiUrl}/${chatId}/full`, { headers });
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to fetch messages: ${response.status} - ${errorText.substring(0, 100)}...`);
            }
            const chatDataFullHistory = await response.json();

            // تحديث رأس المحادثة باستخدام البيانات من القائمة (chatDetailsFromList)
            const userNameForHeader = chatDetailsFromList.user_name || `محادثة رقم ${chatDetailsFromList.chat_id}`;
            chatHeaderName.textContent = userNameForHeader;
            chatHeaderStatus.textContent = chatDetailsFromList.status === 'active' ? 'نشطة' : 'منتهية';
            chatHeaderStatus.className = `status ${chatDetailsFromList.status}`;

            // ملء بيانات الـ modal باستخدام البيانات من القائمة (chatDetailsFromList)
            infoData = {
                chat_id: chatDetailsFromList.chat_id,
                user_id: chatDetailsFromList.user_id || 'غير متاح',
                user_name: chatDetailsFromList.user_name || 'غير معروف',
                user_email: chatDetailsFromList.user_email || 'غير متاح',
                user_type: chatDetailsFromList.user_type || 'غير متاح',
                status: chatDetailsFromList.status,
                started_at: chatDetailsFromList.created_at, // استخدام created_at من list API
                ended_at: chatDetailsFromList.status === 'ended' ? chatDetailsFromList.last_message_time : 'غير متاح' // استخدام last_message_time إذا كانت المحادثة منتهية
            };

            // عرض الرسائل
            chatMessagesContainer.innerHTML = '';
            if (chatDataFullHistory.messages && chatDataFullHistory.messages.length > 0) {
                chatDataFullHistory.messages.forEach(msg => appendMessage(msg));
            } else {
                displayInitialMessage('لا توجد رسائل في هذه المحادثة بعد.');
            }

            // إظهار/إخفاء زر إنهاء المحادثة
            endChatButton.style.display = chatDetailsFromList.status === 'active' ? 'block' : 'none';
            messageInput.disabled = chatDetailsFromList.status !== 'active';
            sendMessageButton.disabled = chatDetailsFromList.status !== 'active';

        } catch (error) {
            console.error('Error opening chat:', error);
            displayInitialMessage(`خطأ في فتح المحادثة: ${error.message}`, true);
        }
    }

    async function sendMessage() {
        const content = messageInput.value.trim();
        if (!content || !currentChatId) {
            console.warn('من فضلك اختر محادثة واكتب رسالتك.');
            return;
        }

        const headers = getAuthHeaders();
        if (!headers) return;

        try {
            const response = await fetch(`${sendMessageApiUrl}/${currentChatId}/message?content=${encodeURIComponent(content)}`, {
                method: 'POST',
                headers: headers
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to send message: ${response.status} - ${errorText.substring(0, 100)}...`);
            }

            // --- بداية منطق العرض الفوري للرسالة ---
            // بناء كائن الرسالة للعرض الفوري
            const newMessage = {
                role: 'admin', // الدور هو 'admin' لأننا نرسل من لوحة الأدمن
                content: content,
                timestamp: new Date().toISOString(), // الوقت الحالي للرسالة
                is_read: true // الرسالة المرسلة افتراضياً مقروءة
            };
            appendMessage(newMessage); // إضافة الرسالة مباشرة إلى واجهة الشات

            messageInput.value = ''; // تفريغ حقل الإدخال

            // تحديث آخر رسالة في بيانات allChatsData (للقائمة الجانبية)
            const chatToUpdate = allChatsData.find(chat => chat.chat_id === currentChatId);
            if (chatToUpdate) {
                chatToUpdate.last_message = content;
                chatToUpdate.last_message_time = newMessage.timestamp;
                // لا نحتاج لتغيير unread_count لرسالة المرسل نفسه
            }
            fetchAndRenderChatList(); // إعادة عرض قائمة المحادثات لتحديث Sidebar
            // --- نهاية منطق العرض الفوري للرسالة ---

            // لا حاجة لـ openChat(currentChatId) هنا بعد الآن لأننا أضفنا الرسالة يدوياً

        } catch (error) {
            console.error('Error sending message:', error);
            console.error(`فشل إرسال الرسالة: ${error.message}`);
        }
    }

    async function endChat() {
        // سيتم استبدال هذا بـ custom modal في بيئة الإنتاج
        if (!currentChatId || !confirm('هل أنت متأكد من إنهاء هذه المحادثة؟')) {
            return;
        }

        const headers = getAuthHeaders();
        if (!headers) return;

        try {
            const response = await fetch(`${endChatApiUrl}/${currentChatId}/end`, {
                method: 'PUT',
                headers: headers
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to end chat: ${response.status} - ${errorText.substring(0, 100)}...`);
            }

            console.log('تم إنهاء المحادثة بنجاح.');
            endChatButton.style.display = 'none';
            messageInput.disabled = true;
            sendMessageButton.disabled = true;
            chatHeaderStatus.textContent = 'منتهية';
            chatHeaderStatus.className = 'status ended';
            fetchAndRenderChatList();
            currentChatId = null;
            displayInitialMessage('اختر محادثة من القائمة للبدء.');
        } catch (error) {
            console.error('Error ending chat:', error);
            console.error(`فشل إنهاء المحادثة: ${error.message}`);
        }
    }

    // --------------------------------------------------------------------------------------------------------------------
    // Event Listeners
    // --------------------------------------------------------------------------------------------------------------------

    sendMessageButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    endChatButton.addEventListener('click', endChat);

    searchChatsButton.addEventListener('click', applyFiltersAndRender);
    conversationSearchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            applyFiltersAndRender();
        }
    });

    // زر إظهار/إخفاء المحادثات المنتهية
    toggleEndedChatsButton.addEventListener('click', () => {
        showEndedChats = !showEndedChats;
        toggleEndedChatsButton.innerHTML = showEndedChats ?
            '<i class="fas fa-eye-slash"></i> إخفاء المحادثات المنتهية' :
            '<i class="fas fa-eye"></i> إظهار المحادثات المنتهية';
        
        toggleEndedChatsButton.classList.toggle('active-filter', showEndedChats);

        applyFiltersAndRender();
    });

    // Toggle categories (رؤوس الأقسام)
    categoryTitles.forEach(title => {
        title.addEventListener('click', function() {
            const list = title.nextElementSibling;
            const icon = title.querySelector('.toggle-icon');

            list.classList.toggle('collapsed');
            // تبديل الأيقونات بشكل صحيح بناءً على ما إذا كانت القائمة مطوية أم لا
            icon.classList.toggle('fa-chevron-down', list.classList.contains('collapsed'));
            icon.classList.toggle('fa-chevron-up', !list.classList.contains('collapsed'));
        });
    });

    // Event listener لفتح شريط المحادثات من السايد بار الأيسر (للدسكتوب)
    if (openConversationSidebar) {
        openConversationSidebar.addEventListener('click', (e) => {
            e.preventDefault();
            conversationBar.classList.remove('collapsed'); // فتح شريط المحادثات
            conversationBar.classList.remove('open'); // التأكد من إزالة كلاس الموبايل 'open' إذا كان موجوداً
            // إخفاء أيقونة الفتح من السايد بار الأيسر عندما يكون الشريط مفتوحاً على الديسكتوب
            openConversationSidebar.style.display = 'none';

            // التأكد من إخفاء الزر العائم للموبايل إذا كان مرئياً (للدسكتوب)
            const openConvBtnMobile = document.getElementById('openConvBtn');
            if (openConvBtnMobile) {
                openConvBtnMobile.style.display = 'none';
            }
        });
    }

    // زر الإغلاق (X) داخل شريط المحادثات الأيمن
    if (closeConversationBar) {
        closeConversationBar.addEventListener('click', function() {
            conversationBar.classList.add('collapsed'); // إخفاء شريط المحادثات
            conversationBar.classList.remove('open'); // إزالة كلاس 'open' للموبايل
            
            // إظهار أيقونة الفتح الصحيحة بناءً على حجم الشاشة
            if (window.innerWidth < 992) {
                const openConvBtnMobile = document.getElementById('openConvBtn');
                if (openConvBtnMobile) {
                    openConvBtnMobile.style.display = 'flex'; // أظهر الزر العائم للموبايل
                }
                if (openConversationSidebar) { // أخفي أيقونة السايدبار الأيسر على الموبايل
                    openConversationSidebar.style.display = 'none';
                }
            } else {
                if (openConversationSidebar) {
                    openConversationSidebar.style.display = 'block'; // أظهر الأيقونة في السايدبار الأيسر على الديسكتوب
                }
                const openConvBtnMobile = document.getElementById('openConvBtn'); // أخفي الزر العائم للموبايل على الديسكتوب
                if (openConvBtnMobile) {
                    openConvBtnMobile.style.display = 'none';
                }
            }
        });
    }

    // Resizing functionality for conversation bar (only on desktop)
    let isResizing = false;
    let startX, startWidth;

    if (resizeHandle) {
        resizeHandle.addEventListener('mousedown', function(e) {
            if (window.innerWidth >= 992) {
                isResizing = true;
                startX = e.clientX;
                startWidth = parseInt(getComputedStyle(conversationBar).width, 10);
                document.body.style.userSelect = 'none';
                document.body.style.cursor = 'col-resize';
            }
        });
    }

    document.addEventListener('mousemove', function(e) {
        if (!isResizing) return;
        let newWidth = startWidth + (startX - e.clientX); // لـ RTL، الحركة عكسية
        if (newWidth < 300) newWidth = 300;
        if (newWidth > 500) newWidth = 500;
        conversationBar.style.width = newWidth + 'px';
    });

    document.addEventListener('mouseup', function() {
        if (isResizing) {
            isResizing = false;
            document.body.style.userSelect = '';
            document.body.style.cursor = '';
        }
    });

    // Event listener for info icon to show modal
    if (infoIcon) {
        infoIcon.addEventListener('click', function() {
            if (!currentChatId || !infoData) {
                console.warn('من فضلك اختر محادثة لعرض معلوماتها.');
                return;
            }

            modalInfoContent.innerHTML = `
                <p><strong>اسم المستخدم:</strong> ${infoData.user_name || 'غير معروف'}</p>
                <p><strong>البريد الإلكتروني:</strong> ${infoData.user_email || 'غير متاح'}</p>
                <p><strong>نوع المستخدم:</strong> ${infoData.user_type || 'غير متاح'}</p>
                <p><strong>معرف المستخدم:</strong> ${infoData.user_id || 'غير متاح'}</p>
                <p><strong>معرف المحادثة:</strong> ${infoData.chat_id || 'غير متاح'}</p>
                <p><strong>حالة المحادثة:</strong> ${infoData.status === 'active' ? 'نشطة' : 'منتهية'}</p>
                <p><strong>تاريخ البدء:</strong> ${infoData.started_at ? formatDateTime(infoData.started_at) : 'غير متاح'}</p>
                <p><strong>تاريخ الانتهاء:</strong> ${infoData.ended_at ? formatDateTime(infoData.ended_at) : 'غير متاح'}</p>
            `;
            infoModal.classList.add('active');
        });
    }

    if (closeInfoModal) {
        closeInfoModal.addEventListener('click', function() {
            infoModal.classList.remove('active');
        });
    }
    if (infoModal) {
        infoModal.addEventListener('click', function(e) {
            if (e.target === infoModal) infoModal.classList.remove('active');
        });
    }


    // --------------------------------------------------------------------------------------------------------------------
    // Initial Setup on Page Load and Window Resize
    // --------------------------------------------------------------------------------------------------------------------

    function setInitialConversationBarState() {
        const openConvBtnMobile = document.getElementById('openConvBtn');
        if (window.innerWidth < 992) {
            conversationBar.classList.add('collapsed'); // مخفي على الموبايل
            conversationBar.classList.remove('open'); // التأكد من إزالة كلاس open
            if (openConvBtnMobile) {
                 openConvBtnMobile.style.display = 'flex'; // يظهر الزر العائم
            }
            if (openConversationSidebar) {
                openConversationSidebar.style.display = 'none'; // أخفي أيقونة السايدبار الأيسر على الموبايل
            }
        } else {
            conversationBar.classList.remove('collapsed'); // ظاهر على الديسكتوب
            conversationBar.classList.remove('open'); // التأكد من إزالة كلاس open
            if (openConvBtnMobile) {
                openConvBtnMobile.style.display = 'none'; // يخفى الزر العائم على الديسكتوب
            }
            if (openConversationSidebar) {
                openConversationSidebar.style.display = 'block'; // أظهر أيقونة السايدبار الأيسر على الديسكتوب
            }
        }
    }

    setInitialConversationBarState();
    window.addEventListener('resize', setInitialConversationBarState);

    fetchAndRenderChatList();

    // ضبط نص الزر الأولي "إظهار المحادثات المنتهية" (لأن showEndedChats = false افتراضيًا)
    toggleEndedChatsButton.innerHTML = showEndedChats ?
        '<i class="fas fa-eye-slash"></i> إخفاء المحادثات المنتهية' :
        '<i class="fas fa-eye"></i> إظهار المحادثات المنتهية';
    // ضبط حالة الكلاس الأولية للزر
    toggleEndedChatsButton.classList.toggle('active-filter', showEndedChats);


    displayInitialMessage('اختر محادثة من القائمة للبدء.');

    // جعل قوائم المحادثات مفتوحة افتراضيًا وتصحيح الأيقونات
    categoryTitles.forEach(title => {
        const list = title.nextElementSibling;
        const icon = title.querySelector('.toggle-icon');
        list.classList.remove('collapsed'); // إزالة كلاس collapsed لضمان أنها مفتوحة
        icon.classList.remove('fa-chevron-down'); // التأكد من إزالة أيقونة السهم لأسفل
        icon.classList.add('fa-chevron-up');    // التأكد من وجود أيقونة السهم لأعلى
    });
});
