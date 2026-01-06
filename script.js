// Mobile Menu Toggle
const mobileMenuToggle = document.getElementById('mobileMenuToggle');
const navMenu = document.getElementById('navMenu');

if (mobileMenuToggle) {
    mobileMenuToggle.addEventListener('click', () => {
        navMenu.classList.toggle('active');
        mobileMenuToggle.classList.toggle('active');
    });
}

// Close mobile menu when clicking on a link
const navLinks = document.querySelectorAll('.nav-menu a');
navLinks.forEach(link => {
    link.addEventListener('click', () => {
        navMenu.classList.remove('active');
        mobileMenuToggle.classList.remove('active');
    });
});

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            const offsetTop = target.offsetTop - 80;
            window.scrollTo({
                top: offsetTop,
                behavior: 'smooth'
            });
        }
    });
});

// Navbar scroll effect
let lastScroll = 0;
const navbar = document.querySelector('.navbar');

window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;
    
    if (currentScroll > 100) {
        navbar.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.15)';
    } else {
        navbar.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.1)';
    }
    
    lastScroll = currentScroll;
});

// Intersection Observer for fade-in animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe elements for animation
document.addEventListener('DOMContentLoaded', () => {
    const animateElements = document.querySelectorAll('.feature-card, .unique-item, .step, .pricing-card, .stat-item');
    
    animateElements.forEach((el, index) => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = `opacity 0.6s ease ${index * 0.1}s, transform 0.6s ease ${index * 0.1}s`;
        observer.observe(el);
    });
});

// Form submission
const contactForm = document.querySelector('.contact-form');
if (contactForm) {
    contactForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        // Get form values
        const formData = new FormData(contactForm);
        const name = contactForm.querySelector('input[type="text"]').value;
        const email = contactForm.querySelector('input[type="email"]').value;
        const message = contactForm.querySelector('textarea').value;
        
        // Simple validation
        if (name && email && message) {
            // Show success message
            alert('Mesajınız başarıyla gönderildi! En kısa sürede size dönüş yapacağız.');
            contactForm.reset();
        } else {
            alert('Lütfen tüm alanları doldurun.');
        }
    });
}

// CTA Button clicks
const ctaButtons = document.querySelectorAll('.btn-primary, .btn-secondary');
ctaButtons.forEach(button => {
    if (button.textContent.includes('Sipariş') || button.textContent.includes('Başla')) {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            // Scroll to pricing section
            const pricingSection = document.getElementById('pricing');
            if (pricingSection) {
                const offsetTop = pricingSection.offsetTop - 80;
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    }
});

// Detection box animation
const detectionBoxes = document.querySelectorAll('.detection-box');
detectionBoxes.forEach((box, index) => {
    setTimeout(() => {
        box.style.opacity = '1';
        box.style.transform = 'scale(1)';
    }, index * 500);
});

// Stats counter animation
const statsNumbers = document.querySelectorAll('.stat-number');
const animateCounter = (element) => {
    const target = element.textContent;
    const isNumber = !isNaN(parseFloat(target));
    
    if (isNumber && target.includes('%')) {
        const num = parseFloat(target);
        let current = 0;
        const increment = num / 50;
        const timer = setInterval(() => {
            current += increment;
            if (current >= num) {
                element.textContent = target;
                clearInterval(timer);
            } else {
                element.textContent = Math.floor(current) + '%';
            }
        }, 30);
    }
};

// Observe stats section for counter animation
const statsObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            statsNumbers.forEach(stat => {
                animateCounter(stat);
            });
            statsObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.5 });

const statsSection = document.querySelector('.stats');
if (statsSection) {
    statsObserver.observe(statsSection);
}

// Add active class to current nav item based on scroll position
const sections = document.querySelectorAll('section[id]');
const navItems = document.querySelectorAll('.nav-menu a');

window.addEventListener('scroll', () => {
    let current = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop - 100;
        const sectionHeight = section.clientHeight;
        
        if (window.pageYOffset >= sectionTop && window.pageYOffset < sectionTop + sectionHeight) {
            current = section.getAttribute('id');
        }
    });
    
    navItems.forEach(item => {
        item.classList.remove('active');
        if (item.getAttribute('href') === `#${current}`) {
            item.classList.add('active');
        }
    });
});

// Add parallax effect to hero section
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const heroImage = document.querySelector('.hero-image');
    if (heroImage && scrolled < window.innerHeight) {
        heroImage.style.transform = `translateY(${scrolled * 0.3}px)`;
        heroImage.style.opacity = 1 - (scrolled / window.innerHeight);
    }
});

// Console welcome message
console.log('%c🏊 PoolGuard AI', 'font-size: 20px; font-weight: bold; color: #0066ff;');
console.log('%cHavuz güvenliğiniz için gelişmiş AI teknolojisi', 'font-size: 12px; color: #666;');

