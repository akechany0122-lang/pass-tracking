document.addEventListener("DOMContentLoaded", function () {
    const options = { root: null, rootMargin: "0px", threshold: 0.15 };
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add("is-visible");
                observer.unobserve(entry.target);
            }
        });
    }, options);

    document.querySelectorAll(".scroll-trigger").forEach(t => observer.observe(t));

    const video = document.getElementById('main-video');
    const soundBtn = document.getElementById('sound-btn');
    if (video && soundBtn) {
        video.play().catch(e => console.log(e));
        soundBtn.addEventListener('click', function () {
            if (video.muted) {
                video.muted = false;
                soundBtn.textContent = "SOUND OFF";
                soundBtn.style.background = "#fff";
                soundBtn.style.color = "#000";
            } else {
                video.muted = true;
                soundBtn.textContent = "SOUND ON";
                soundBtn.style.background = "rgba(0, 0, 0, 0.5)";
                soundBtn.style.color = "#fff";
            }
        });
    }

    const uiToggleBtn = document.getElementById('ui-toggle-btn');
    const controlPanel = document.getElementById('control-panel');
    if (uiToggleBtn && controlPanel) {
        uiToggleBtn.addEventListener('click', () => {
            controlPanel.classList.toggle('is-active');
        });
    }

    const scrollContainer = document.getElementById('auto-scroll-container');
    if (scrollContainer) {
        let isHovered = false;
        scrollContainer.addEventListener('mouseenter', () => isHovered = true);
        scrollContainer.addEventListener('mouseleave', () => isHovered = false);
        function autoScroll() {
            if (!isHovered) {
                scrollContainer.scrollLeft += 1;
            }
            if (scrollContainer.scrollLeft >= (scrollContainer.scrollWidth - scrollContainer.clientWidth)) {
                scrollContainer.scrollLeft = 0;
            }
            requestAnimationFrame(autoScroll);
        }
        autoScroll();
    }
});