const btn = document.getElementById("send");
const apiUrl = 'https://enabled-early-vulture.ngrok-free.app/login';

// متغيرات عالمية (global variables) ممكن نحفظ فيها التوكن والـ user ID
let accessToken = null;
let userId = null;

async function login() {

    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    console.log("Username:", username);
    console.log("Password:", password);

    const requestBody = new URLSearchParams();
    requestBody.append('grant_type', 'password');
    requestBody.append('username', username);
    requestBody.append('password', password);
    requestBody.append('scope', '');
    requestBody.append('client_id', 'string');
    requestBody.append('client_secret', 'string');

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'accept': 'application/json',
                'ngrok-skip-browser-warning': 'true'
            },
            body: requestBody
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Login failed: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        console.log('Login successful:', data);

        // --- الجزء الجديد هنا ---
        // 1. الاحتفاظ بالبيانات في متغيرات JavaScript
        accessToken = data.access_token;
        userId = data.user_id;

        console.log('Access Token stored:', accessToken);
        console.log('User ID stored:', userId);

        // 2. الاحتفاظ بالبيانات في localStorage (عشان تفضل موجودة حتى لو قفلت المتصفح)
        localStorage.setItem('accessToken', data.access_token);
        localStorage.setItem('userId', data.user_id);
        localStorage.setItem('tokenType', data.token_type); // ممكن تحفظ الـ token_type كمان

        console.log('Access Token saved to localStorage.');
        // -------------------------

        // ممكن هنا توجه المستخدم لصفحة تانية بعد تسجيل الدخول بنجاح
        window.location.href = '\index.html';

    } catch (error) {
        console.error('Error during login:', error.message);
    }
}

btn.onclick = function(event) {
    event.preventDefault();
    login();
};

// مثال: كيف تستخدم التوكن المحفوظ لعمل طلب مؤكد (authenticated request)
// الدالة دي ممكن تستدعيها بعد ما تكون سجلت دخول ونجحت العملية
function makeAuthenticatedRequest() {
    // بنجيب التوكن من الـ localStorage (أو من المتغير accessToken لو حافظه فيه)
    const storedToken = localStorage.getItem('accessToken');
    const storedTokenType = localStorage.getItem('tokenType'); // هنجيب الـ bearer

    if (storedToken && storedTokenType) {
        console.log('Making authenticated request with token:', storedToken);
        // مثال لطلب API بيحتاج توكن
        fetch('https://some-api-endpoint.com/profile', { // غير ده لـ API endpoint حقيقي بيحتاج توكن
            method: 'GET',
            headers: {
                'Authorization': `${storedTokenType} ${storedToken}`, // مهم جداً: "Bearer " + التوكن
                'accept': 'application/json'
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Auth request failed: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Authenticated request successful:', data);
        })
        .catch(error => {
            console.error('Error in authenticated request:', error.message);
        });
    } else {
        console.log('No access token found. Please log in first.');
    }
}

// مثال على استخدام makeAuthenticatedRequest بعد ما المستخدم يسجل دخول
// ممكن تعمل زرار تاني في الـ HTML يستدعي الدالة دي، أو تستدعيها بعد نجاح الـ login مباشرةً.
// setTimeout(makeAuthenticatedRequest, 5000); // مثال: تستدعيها بعد 5 ثواني من تحميل الصفحة (لا تستخدم في الإنتاج)