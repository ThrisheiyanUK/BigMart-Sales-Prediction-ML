// BigMart Frontend JavaScript with Anime.js
class BigMartApp {
    constructor() {
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initAnimations();
        this.initCharts();
        this.initTabs();
        this.showLoadingAnimation();
    }

    setupEventListeners() {
        // Smooth scrolling for navigation links
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', this.handleNavClick.bind(this));
        });

        // Window scroll events
        window.addEventListener('scroll', this.handleScroll.bind(this));
        
        // Resize events
        window.addEventListener('resize', this.handleResize.bind(this));
    }

    handleNavClick(e) {
        e.preventDefault();
        const target = e.target.getAttribute('href');
        
        // Update active nav link
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        e.target.classList.add('active');
        
        // Smooth scroll to section
        if (target.startsWith('#')) {
            const section = document.querySelector(target);
            if (section) {
                section.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        }
    }

    handleScroll() {
        const navbar = document.querySelector('.navbar');
        const scrollTop = window.pageYOffset;
        
        // Add shadow to navbar on scroll
        if (scrollTop > 10) {
            navbar.style.boxShadow = '0 2px 20px rgba(0,0,0,0.1)';
        } else {
            navbar.style.boxShadow = 'none';
        }

        // Animate elements on scroll
        this.animateOnScroll();
    }

    handleResize() {
        // Handle responsive behavior
        this.updateChartSizes();
    }

    showLoadingAnimation() {
        const overlay = document.getElementById('loadingOverlay');
        overlay.classList.add('active');
        
        // Animate loading spinner
        anime({
            targets: '.loading-spinner',
            rotate: '360deg',
            duration: 1000,
            easing: 'linear',
            loop: true
        });

        // Hide loading after content is ready
        setTimeout(() => {
            anime({
                targets: overlay,
                opacity: 0,
                duration: 500,
                easing: 'easeOutQuad',
                complete: () => {
                    overlay.classList.remove('active');
                    this.playIntroAnimations();
                }
            });
        }, 2000);
    }

    playIntroAnimations() {
        // Animate hero title
        anime({
            targets: '.hero-title',
            opacity: [0, 1],
            translateY: [50, 0],
            duration: 800,
            easing: 'easeOutQuad',
            delay: 200
        });

        // Animate hero subtitle
        anime({
            targets: '.hero-subtitle',
            opacity: [0, 1],
            translateY: [30, 0],
            duration: 600,
            easing: 'easeOutQuad',
            delay: 400
        });

        // Animate stat cards with stagger
        anime({
            targets: '.stat-card',
            opacity: [0, 1],
            translateY: [40, 0],
            scale: [0.9, 1],
            duration: 600,
            easing: 'easeOutBack',
            delay: anime.stagger(150, {start: 600})
        });

        // Animate counters in stat cards
        this.animateCounters();
    }

    animateCounters() {
        const counters = document.querySelectorAll('.stat-number');
        counters.forEach(counter => {
            const target = parseInt(counter.textContent.replace(/,/g, ''));
            const duration = 2000;
            
            anime({
                targets: counter,
                innerHTML: [0, target],
                duration: duration,
                easing: 'easeOutQuad',
                delay: 800,
                update: function(anim) {
                    counter.innerHTML = Math.round(anim.animatables[0].target.innerHTML).toLocaleString();
                }
            });
        });
    }

    animateOnScroll() {
        const elements = document.querySelectorAll('[data-animate]');
        
        elements.forEach(element => {
            if (this.isElementInViewport(element) && !element.classList.contains('animated')) {
                element.classList.add('animated');
                const animationType = element.getAttribute('data-animate');
                this.playScrollAnimation(element, animationType);
            }
        });
    }

    isElementInViewport(el) {
        const rect = el.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
    }

    playScrollAnimation(element, type) {
        let animationProps = {};
        
        switch (type) {
            case 'fadeInUp':
                animationProps = {
                    opacity: [0, 1],
                    translateY: [40, 0],
                    duration: 600,
                    easing: 'easeOutQuad'
                };
                break;
            case 'slideInLeft':
                animationProps = {
                    opacity: [0, 1],
                    translateX: [-60, 0],
                    duration: 700,
                    easing: 'easeOutQuad'
                };
                break;
            case 'slideInRight':
                animationProps = {
                    opacity: [0, 1],
                    translateX: [60, 0],
                    duration: 700,
                    easing: 'easeOutQuad'
                };
                break;
            case 'zoomIn':
                animationProps = {
                    opacity: [0, 1],
                    scale: [0.8, 1],
                    duration: 600,
                    easing: 'easeOutBack'
                };
                break;
            default:
                animationProps = {
                    opacity: [0, 1],
                    duration: 500,
                    easing: 'easeOutQuad'
                };
        }
        
        // Set initial state
        element.style.opacity = '0';
        
        anime({
            targets: element,
            ...animationProps
        });
    }

    initTabs() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabPanels = document.querySelectorAll('.tab-panel');
        
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.getAttribute('data-tab');
                
                // Remove active class from all buttons and panels
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabPanels.forEach(panel => panel.classList.remove('active'));
                
                // Add active class to clicked button and corresponding panel
                button.classList.add('active');
                document.getElementById(targetTab).classList.add('active');
                
                // Animate tab switch
                anime({
                    targets: `#${targetTab}`,
                    opacity: [0, 1],
                    translateY: [20, 0],
                    duration: 300,
                    easing: 'easeOutQuad'
                });
            });
        });
    }

    initCharts() {
        // Initialize Chart.js with sample data
        const ctx = document.getElementById('salesChart');
        if (ctx) {
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                    datasets: [{
                        label: 'Sales ($)',
                        data: [12000, 19000, 15000, 25000, 22000, 30000],
                        borderColor: '#2563eb',
                        backgroundColor: 'rgba(37, 99, 235, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#2563eb',
                        pointBorderColor: '#fff',
                        pointBorderWidth: 2,
                        pointRadius: 6,
                        pointHoverRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                display: false
                            },
                            ticks: {
                                color: '#64748b'
                            }
                        },
                        y: {
                            grid: {
                                color: '#e2e8f0'
                            },
                            ticks: {
                                color: '#64748b',
                                callback: function(value) {
                                    return '$' + value.toLocaleString();
                                }
                            }
                        }
                    },
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    }
                }
            });
        }
    }

    updateChartSizes() {
        // Handle chart responsiveness
        Chart.helpers.each(Chart.instances, function(instance) {
            instance.resize();
        });
    }

    // Hover animations for interactive elements
    initHoverAnimations() {
        // Stat cards hover effect
        document.querySelectorAll('.stat-card').forEach(card => {
            card.addEventListener('mouseenter', () => {
                anime({
                    targets: card,
                    scale: 1.05,
                    duration: 200,
                    easing: 'easeOutQuad'
                });
            });
            
            card.addEventListener('mouseleave', () => {
                anime({
                    targets: card,
                    scale: 1,
                    duration: 200,
                    easing: 'easeOutQuad'
                });
            });
        });

        // Product items hover effect
        document.querySelectorAll('.product-item').forEach(item => {
            item.addEventListener('mouseenter', () => {
                anime({
                    targets: item.querySelector('.product-sales'),
                    scale: 1.1,
                    color: '#059669',
                    duration: 200,
                    easing: 'easeOutQuad'
                });
            });
            
            item.addEventListener('mouseleave', () => {
                anime({
                    targets: item.querySelector('.product-sales'),
                    scale: 1,
                    color: '#10b981',
                    duration: 200,
                    easing: 'easeOutQuad'
                });
            });
        });

        // Insight cards pulse effect
        document.querySelectorAll('.insight-card').forEach(card => {
            card.addEventListener('mouseenter', () => {
                anime({
                    targets: card.querySelector('.insight-icon'),
                    scale: [1, 1.1, 1],
                    duration: 600,
                    easing: 'easeInOutQuad'
                });
            });
        });
    }

    // Utility method for creating floating particles (optional enhancement)
    createFloatingParticles() {
        const particles = [];
        const particleCount = 20;
        
        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.cssText = `
                position: absolute;
                width: 4px;
                height: 4px;
                background: rgba(37, 99, 235, 0.3);
                border-radius: 50%;
                pointer-events: none;
            `;
            
            document.body.appendChild(particle);
            particles.push(particle);
            
            // Animate particles
            anime({
                targets: particle,
                translateX: anime.random(-100, 100),
                translateY: anime.random(-100, 100),
                scale: anime.random(0.5, 1.5),
                opacity: [0, 1, 0],
                duration: anime.random(3000, 6000),
                easing: 'easeInOutQuad',
                loop: true,
                direction: 'alternate'
            });
        }
    }

    // Method to load and display actual data from CSV files
    async loadDataFromCSV() {
        try {
            // This would be implemented to load your actual CSV data
            // For now, using mock data
            console.log('Loading data from CSV files...');
            // You could use libraries like Papa Parse to read CSV files
        } catch (error) {
            console.error('Error loading CSV data:', error);
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new BigMartApp();
    
    // Initialize hover animations after a short delay
    setTimeout(() => {
        app.initHoverAnimations();
    }, 1000);
});

// Global utility functions
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

function formatNumber(number) {
    return new Intl.NumberFormat('en-US').format(number);
}

// Export for potential module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BigMartApp;
}
