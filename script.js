document.addEventListener('DOMContentLoaded', function () {
  const toggleBtn = document.getElementById('toggleBtn');
  const sidebar = document.getElementById('sidebar');
  const icon = toggleBtn.querySelector('i');

  // حالة البدء (تصغير على الأجهزة المحمولة)
  const startCollapsed = window.innerWidth < 992;
  if (startCollapsed) {
    sidebar.classList.add('collapsed');
    icon.classList.remove('fa-times');
    icon.classList.add('fa-bars');
  }

  // تبديل الشريط الجانبي
  toggleBtn.addEventListener('click', function () {
    sidebar.classList.toggle('collapsed');

    // تغيير الأيقونة
    if (sidebar.classList.contains('collapsed')) {
      icon.classList.remove('fa-times');
      icon.classList.add('fa-bars');
    } else {
      icon.classList.remove('fa-bars');
      icon.classList.add('fa-times');
    }
  });

  // إغلاق الشريط عند النقر على عنصر (للموبايل)
  const navLinks = document.querySelectorAll('.sidebar-menu ul li a');
  navLinks.forEach((link) => {
    link.addEventListener('click', function () {
      if (window.innerWidth < 992) {
        sidebar.classList.add('collapsed');
        icon.classList.remove('fa-times');
        icon.classList.add('fa-bars');
      }

      // إزالة النشط من جميع العناصر
      navLinks.forEach((l) => {
        l.parentElement.classList.remove('active');
      });

      // إضافة النشط للعنصر الحالي
      this.parentElement.classList.add('active');

      // حفظ الصفحة النشطة في localStorage
      const page = this.getAttribute('href');
      if (page && page !== '#') {
        localStorage.setItem('activePage', page);
      }
    });
  });

  // التكيف مع تغيير حجم الشاشة
  window.addEventListener('resize', function () {
    if (window.innerWidth < 992) {
      sidebar.classList.add('collapsed');
      icon.classList.remove('fa-times');
      icon.classList.add('fa-bars');
    } else {
      sidebar.classList.remove('collapsed');
      icon.classList.remove('fa-bars');
      icon.classList.add('fa-times');
    }
  });

  // تحديد العنصر النشط بناءً على الصفحة الحالية
  function setActiveMenuItem() {
    const currentPage =
      window.location.pathname.split('/').pop() || 'index.html';

    navLinks.forEach((link) => {
      const linkPage = link.getAttribute('href');
      if (
        linkPage === currentPage ||
        (linkPage === 'index.html' && currentPage === '')
      ) {
        link.parentElement.classList.add('active');
      } else {
        link.parentElement.classList.remove('active');
      }
    });
  }

  // استدعاء الدالة عند تحميل الصفحة
  setActiveMenuItem();
});

// كود الشات بوت (يجب وضعه في صفحة index.html فقط)
if (
  window.location.pathname.includes('index.html') ||
  window.location.pathname === '/'
) {
  document.addEventListener('DOMContentLoaded', function () {
    // عناصر واجهة المستخدم
    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const apiKeyInput = document.getElementById('api-key');
    const apiUrlInput = document.getElementById('api-url');
    const apiModelSelect = document.getElementById('api-model');
    const saveApiBtn = document.getElementById('save-api');

    // إعدادات API الافتراضية
    let apiSettings = {
      apiKey: localStorage.getItem('chatbot_api_key') || '',
      apiUrl:
        localStorage.getItem('chatbot_api_url') ||
        'https://api.openai.com/v1/chat/completions',
      apiModel: localStorage.getItem('chatbot_api_model') || 'gpt-3.5',
    };

    // تهيئة واجهة المستخدم مع الإعدادات المحفوظة
    if (apiKeyInput) apiKeyInput.value = apiSettings.apiKey;
    if (apiUrlInput) apiUrlInput.value = apiSettings.apiUrl;
    if (apiModelSelect) apiModelSelect.value = apiSettings.apiModel;

    // حفظ إعدادات API
    if (saveApiBtn) {
      saveApiBtn.addEventListener('click', function () {
        apiSettings = {
          apiKey: apiKeyInput.value,
          apiUrl: apiUrlInput.value,
          apiModel: apiModelSelect.value,
        };

        localStorage.setItem('chatbot_api_key', apiSettings.apiKey);
        localStorage.setItem('chatbot_api_url', apiSettings.apiUrl);
        localStorage.setItem('chatbot_api_model', apiSettings.apiModel);

        alert('تم حفظ الإعدادات بنجاح!');
      });
    }

    // إرسال رسالة عند النقر على زر الإرسال أو الضغط على Enter
    if (sendButton) sendButton.addEventListener('click', sendMessage);
    if (userInput)
      userInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
          sendMessage();
        }
      });

    // دالة إرسال الرسالة
    async function sendMessage() {
      const message = userInput.value.trim();
      if (!message) return;

      // إضافة رسالة المستخدم إلى الشات
      addMessageToChat('user', message);
      userInput.value = '';

      // إظهار رسالة "يكتب..." أثناء انتظار الرد
      const typingIndicator = addTypingIndicator();

      try {
        // إرسال الرسالة إلى API
        const botResponse = await sendToChatAPI(message);

        // إزالة مؤشر الكتابة وإضافة رد البوت
        removeTypingIndicator(typingIndicator);
        addMessageToChat('bot', botResponse);
      } catch (error) {
        removeTypingIndicator(typingIndicator);
        addMessageToChat(
          'bot',
          'عذرًا، حدث خطأ في الاتصال بالخادم. يرجى التحقق من إعدادات API.'
        );
        console.error('API Error:', error);
      }
    }

    // دالة لإضافة رسالة إلى الشات
    function addMessageToChat(sender, message) {
      const messageDiv = document.createElement('div');
      messageDiv.classList.add('message', sender + '-message');
      messageDiv.textContent = message;
      chatMessages.appendChild(messageDiv);
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // دالة لإضافة مؤشر "يكتب..."
    function addTypingIndicator() {
      const typingDiv = document.createElement('div');
      typingDiv.classList.add('message', 'bot-message', 'typing');
      typingDiv.id = 'typing-indicator';
      typingDiv.textContent = '...يكتب';
      chatMessages.appendChild(typingDiv);
      chatMessages.scrollTop = chatMessages.scrollHeight;
      return typingDiv;
    }

    // دالة لإزالة مؤشر "يكتب..."
    function removeTypingIndicator(indicator) {
      if (indicator && indicator.parentNode) {
        indicator.parentNode.removeChild(indicator);
      }
    }

    // دالة للاتصال بـ API
    async function sendToChatAPI(message) {
      console.log('api', apiSettings);
      if (!apiSettings.apiKey || !apiSettings.apiUrl) {
        throw new Error('إعدادات API غير مكتملة');
      }

      const response = await fetch(apiSettings.apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiSettings.apiKey}`,
        },
        body: JSON.stringify({
          model: apiSettings.apiModel,
          messages: [{ role: 'user', content: message }],
          temperature: 0.7,
        }),
      });

      if (!response.ok) {
        throw new Error(`خطأ في API: ${response.status}`);
      }

      const data = await response.json();
      return data.choices[0].message.content;
    }

    // رسالة ترحيبية عند تحميل الصفحة
    if (chatMessages && chatMessages.children.length === 0) {
      addMessageToChat('bot', 'مرحبًا! كيف يمكنني مساعدتك اليوم؟');
    }
  });
}
