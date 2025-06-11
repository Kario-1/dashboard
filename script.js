document.addEventListener('DOMContentLoaded', function () {
    const toggleBtn = document.getElementById('toggleBtn');
    const sidebar = document.getElementById('sidebar');
    const icon = toggleBtn.querySelector('i');

    // **التعديل هنا:** جعل السايد بار مغلقاً افتراضياً على جميع الشاشات.
    sidebar.classList.add('collapsed');
    icon.classList.remove('fa-times');
    icon.classList.add('fa-bars');

    // تبديل الشريط الجانبي (فتح/إغلاق)
    toggleBtn.addEventListener('click', function () {
        sidebar.classList.toggle('collapsed');

        // تغيير الأيقونة بناءً على حالة الشريط
        if (sidebar.classList.contains('collapsed')) {
            icon.classList.remove('fa-times');
            icon.classList.add('fa-bars');
        } else {
            icon.classList.remove('fa-bars');
            icon.classList.add('fa-times');
        }
    });

    // إغلاق الشريط عند النقر على عنصر في القائمة (للموبايل فقط)
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
            // على الشاشات الصغيرة، تأكد أنه مغلق
            sidebar.classList.add('collapsed');
            icon.classList.remove('fa-times');
            icon.classList.add('fa-bars');
        } else {
            // على الشاشات الكبيرة، لا نفتح السايدبار تلقائياً إذا كان مغلقاً
            // يبقى في حالته (مغلق إذا كان مغلق، مفتوح إذا كان مفتوح)
            // إذا كنت تريده أن يفتح دائماً على الديسكتوب، أعد سطر:
            // sidebar.classList.remove('collapsed');
            // icon.classList.remove('fa-bars');
            // icon.classList.add('fa-times');
            // حالياً، تركته كما هو ليحتفظ بالحالة عند التكبير من شاشة صغيرة
            // إذا تم فتحه يدويا على شاشة صغيرة، سيظل مفتوحا عند التكبير إلى شاشة كبيرة.
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
