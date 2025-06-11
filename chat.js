// جلب العناصر الأساسية من الـ HTML
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const dataSourcesCheckboxes = document.querySelectorAll('.checkbox-group input[type="checkbox"]');

// الـ API Endpoints
const chatApiUrl = 'https://enabled-early-vulture.ngrok-free.app/admin/model/chat';
const oldChatApiUrl = 'https://enabled-early-vulture.ngrok-free.app/chats/99/full'; // رابط الشات القديم

// رقم الشات الثابت، ممكن تخليه متغير لو عندك أكتر من شات
const fixedChatId = 99;

// دالة لمعالجة النص اللي بيرجع من الموديل (الـ response)
function formatModelResponse(message) {
    let formattedMessage = message;

    // 1. استبدال النجوم (أربع نجوم) بـ <strong> (Bold)
    // المثال: **كلمة** -> <strong>كلمة</strong>
    formattedMessage = formattedMessage.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // 2. استبدال الـ \n (سطر جديد) بـ <br> (break line في الـ HTML)
    formattedMessage = formattedMessage.replace(/\\n/g, '<br>');
    formattedMessage = formattedMessage.replace(/\\t/g, '&nbsp;&nbsp;&nbsp;&nbsp;');
    
    // 3. استبدال الشرط المائلة المزدوجة \" (double quote) بـ " عادية
    formattedMessage = formattedMessage.replace(/\\"/g, '"');

    // مثال: لو بيستخدم _كلمة_ للـ italic:
    formattedMessage = formattedMessage.replace(/_(.*?)_/g, '<em>$1</em>');
    // formatted: لو بيستخدم - - للـ strikethrough:
    formattedMessage = formattedMessage.replace(/--(.*?)--/g, '<s>$1</s>');

    return formattedMessage;
}
// دالة لإضافة مؤشر "يكتب..." مع تأثيرات حركية
function addTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.classList.add('message', 'bot-message', 'typing');
    typingDiv.id = 'typing-indicator';

    // إضافة كلمة "يكتب"
    const typingText = document.createElement('span');
    typingText.textContent = 'يكتب';
    typingDiv.appendChild(typingText);

    // إضافة النقط كعناصر <span> منفصلة لتطبيق التأثير عليها
    for (let i = 0; i < 3; i++) {
        const dotSpan = document.createElement('span');
        dotSpan.textContent = '.';
        typingDiv.appendChild(dotSpan);
    }

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

// دالة لإضافة رسالة جديدة إلى نافذة الشات
function appendMessage(sender, message) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');
    // إضافة كلاس لتمييز رسالة المستخدم عن رد الموديل
    messageElement.classList.add(sender === 'user' ? 'user-message' : 'bot-message');

    // إذا كانت الرسالة من الروبوت، قم بتنسيقها
    const displayedMessage = sender === 'bot' ? formatModelResponse(message) : message;

    messageElement.innerHTML = `<strong>${sender === 'user' ? 'الادمن' : 'المودل'}:</strong><br> ${displayedMessage}`;
    chatMessages.appendChild(messageElement);

    // ننزل لآخر الشات عشان نشوف آخر رسالة
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// دالة إرسال السؤال للموديل AI
async function sendQuestionToModel() {
    const question = userInput.value.trim(); // جلب السؤال من الـ input وحذف المسافات الزائدة
    if (!question) {
        alert('من فضلك أدخل سؤالك.'); // لو الـ input فاضي
        return;
    }

    // عرض سؤال المستخدم في نافذة الشات
    appendMessage('user', question);
    userInput.value = ''; // تفريغ الـ input بعد الإرسال

    // إظهار رسالة "يكتب..." أثناء انتظار الرد
    const typingIndicator = addTypingIndicator();


    // جلب الـ Access Token و الـ Token Type من الـ localStorage
    const accessToken = localStorage.getItem('accessToken');
    const tokenType = localStorage.getItem('tokenType'); // غالباً هتكون "bearer"

    if (!accessToken || !tokenType) {
        console.error('Access Token or Token Type not found in localStorage. Please log in.');
        appendMessage('bot', 'عفواً، لا يمكن إرسال الرسالة. برجاء تسجيل الدخول أولاً.');
        // ممكن توجه المستخدم لصفحة تسجيل الدخول
        window.location.href = '/login.html';
        return;
    }

    const selectedDataSources = [];
    dataSourcesCheckboxes.forEach(checkbox => {
        if (checkbox.checked) {
            selectedDataSources.push(checkbox.value);
        }
    });

    // لو مفيش أي مصدر بيانات تم اختياره، ممكن تعمل تنبيه للمستخدم
    if (selectedDataSources.length === 0) {
        appendMessage('bot', 'برجاء اختيار مصدر بيانات واحد على الأقل للموديل.');
        return;
    }

    try {
        const response = await fetch(chatApiUrl, {
            method: 'POST',
            headers: {
                'accept': 'application/json',
                'Authorization': `${tokenType} ${accessToken}`, // التوكن يتم إضافته هنا
                'Content-Type': 'application/json', // الـ Content-Type هنا JSON كما في الـ curl
                'ngrok-skip-browser-warning': 'true'
            },
            body: JSON.stringify({
                "question": question, // السؤال متغير
                "chat_id": fixedChatId, // استخدام رقم الشات الثابت هنا
                "temperature": 0.7,    // ثابت
                "max_tokens": 1000,     // ثابت
                "selected_sources": selectedDataSources // ***** إرسال المصادر المختارة للباك إند *****
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(`Failed to get response from AI model: ${response.status} - ${JSON.stringify(errorData)}`);
        }

        const data = await response.json();
        console.log('AI Model response:', data);
        
        // إزالة مؤشر الكتابة وإضافة رد البوت
        removeTypingIndicator(typingIndicator);
        // عرض رد الموديل في نافذة الشات
        if (data.response) {
            appendMessage('bot', data.response);
        } else {
            appendMessage('bot', 'عفواً، لم أتمكن من فهم الرد.');
        }

    } catch (error) {
        removeTypingIndicator(typingIndicator);
        console.error('Error sending message to AI model:', error.message);
        appendMessage('bot', `حدث خطأ: ${error.message}. برجاء المحاولة لاحقاً.`);
    }
}

// دالة لتحميل الشات القديم
async function loadOldChat() {
    console.log("Attempting to load old chat...");
    const accessToken = localStorage.getItem('accessToken');
    const tokenType = localStorage.getItem('tokenType');

    if (!accessToken || !tokenType) {
        console.warn('No access token found. Cannot load old chat. Please log in first.');
        appendMessage('bot', 'عفواً، لا يمكن تحميل المحادثات السابقة. برجاء تسجيل الدخول أولاً.');
        // ممكن توجه المستخدم لصفحة تسجيل الدخول لو التوكن مش موجود
        window.location.href = '/login.html';
        return;
    }

    try {
        const response = await fetch(oldChatApiUrl, {
            method: 'GET',
            headers: {
                'accept': 'application/json',
                'Authorization': `${tokenType} ${accessToken}`,
                'ngrok-skip-browser-warning': 'true'
            }
        });

        if (!response.ok) {
            // **هنا التعديل الرئيسي:** بدل response.json()، هنستخدم response.text()
            const errorResponseText = await response.text(); // اقرا الرد كنص عادي
            console.error('API Error Response Text:', errorResponseText); // اطبع النص ده في الـ console

            // حاول تشوف لو النص ده فيه كلمة "expired" أو "invalid token"
            if (errorResponseText.includes('expired') || errorResponseText.includes('invalid')) {
                throw new Error('Access Token expired or invalid. Please log in again.');
            } else {
                // لو مشكلة تانية، ارمي الـ Error بالنص اللي رجع
                throw new Error(`Failed to load old chat: ${response.status} - ${errorResponseText.substring(0, 200)}...`);
            }
        }

        const chatData = await response.json(); // لو الرد كان ok، يبقى كمل كـ JSON
        console.log('Old chat loaded:', chatData);

        // عرض الرسائل القديمة
        if (chatData.messages && chatData.messages.length > 0) {
            // هنا تأكد إن الـ messages array بتضم الـ sender والـ content
            chatData.messages.forEach(msg => {
                // افتراض أن رسائل الشات القديمة تحتوي على sender و content
                // إذا كان الـ sender مش معروف، ممكن تفترض إنه الموديل
                appendMessage(msg.role , formatModelResponse(msg.content));
            });
        } else {
            appendMessage('bot', 'مرحباً! لا توجد محادثات سابقة حتى الآن. كيف يمكنني مساعدتك؟');
        }

    } catch (error) {
        console.error('Error loading old chat:', error.message);
        appendMessage('bot', `حدث خطأ أثناء تحميل المحادثات السابقة: ${error.message}`);
        // ممكن هنا تحذف التوكن من الـ localStorage وتوجه المستخدم لصفحة تسجيل الدخول لو المشكلة في التوكن
        if (error.message.includes('expired') || error.message.includes('invalid')) {
            localStorage.removeItem('accessToken');
            localStorage.removeItem('tokenType');
            localStorage.removeItem('userId');
            window.location.href = '/index.html';
        }
    }
}


// ربط زر الإرسال بالدالة
sendButton.addEventListener('click', sendQuestionToModel);

// تمكين الإرسال بزر Enter في حقل الإدخال
userInput.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // لمنع إرسال الفورم لو الـ input جوه form
        sendQuestionToModel();
    }
});

// استدعاء دالة تحميل الشات القديم عند تحميل الصفحة بالكامل
document.addEventListener('DOMContentLoaded', loadOldChat);