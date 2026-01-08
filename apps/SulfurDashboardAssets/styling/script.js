// Section glow
document.querySelectorAll('.glowing-section').forEach(section => {
    const ring = section.querySelector('.section-ring');

    function center() {
        const b = section.getBoundingClientRect();
        return { x: b.width / 2, y: b.height / 2, r: Math.min(b.width, b.height) / 2 };
    }

    section.addEventListener('mousemove', e => {
        const rect = section.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const m = center();
        const ang = (Math.atan2(my - m.y, mx - m.x) * 180 / Math.PI + 360) % 360;
        ring.style.setProperty('--angle', ang + 'deg');
        section.style.transform = `translate(${(mx - m.x) / m.r * 8}px, ${(my - m.y) / m.r * 8}px)`;
    });

    section.addEventListener('mouseenter', () => {
        ring.style.boxShadow = '0 0 25px 12px rgba(255,165,0,0.6)';
    });

    section.addEventListener('mouseleave', () => {
        ring.style.boxShadow = 'none';
        section.style.transform = 'translate(0,0)';
    });
});

// Section glow sidebar
document.querySelectorAll('.glowing-section-sidebar').forEach(section => {
    const ring = section.querySelector('.section-ring-sidebar');

    function center() {
        const b = section.getBoundingClientRect();
        return { x: b.width / 2, y: b.height / 2, r: Math.min(b.width, b.height) / 2 };
    }

    section.addEventListener('mousemove', e => {
        const rect = section.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const m = center();
        const ang = (Math.atan2(my - m.y, mx - m.x) * 180 / Math.PI + 360) % 360;
        ring.style.setProperty('--angle', ang + 'deg');
        section.style.transform = `translate(${(mx - m.x) / m.r * 8}px, ${(my - m.y) / m.r * 8}px)`;
    });

    section.addEventListener('mouseenter', () => {
        ring.style.boxShadow = '0 0 25px 12px rgba(255,165,0,0.6)';
    });

    section.addEventListener('mouseleave', () => {
        ring.style.boxShadow = 'none';
        section.style.transform = 'translate(0,0)';
    });
});

// Chart glow
document.querySelectorAll('.glowing-chart').forEach(chart => {
    const ring = chart.querySelector('.chart-ring');
    chart.addEventListener('mousemove', e => {
        const rect = chart.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const cx = rect.width / 2, cy = rect.height / 2;
        const ang = (Math.atan2(my - cy, mx - cx) * 180 / Math.PI + 360) % 360;
        ring.style.setProperty('--angle', ang + 'deg');
        chart.style.transform = `translate(${(mx - cx) / cx * 4}px, ${(my - cy) / cy * 4}px)`;
    });
    chart.addEventListener('mouseenter', () => {
        ring.style.boxShadow = '0 0 20px 8px rgba(255,165,0,0.7)';
    });
    chart.addEventListener('mouseleave', () => {
        ring.style.boxShadow = 'none';
        chart.style.transform = 'translate(0,0)';
    });
});

// Slider glow
document.querySelectorAll('.slider-wrapper').forEach(wrapper => {
    const ring = wrapper.querySelector('.slider-ring');
    wrapper.addEventListener('mousemove', e => {
        const rect = wrapper.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const cx = rect.width / 2, cy = rect.height / 2;
        const ang = (Math.atan2(my - cy, mx - cx) * 180 / Math.PI + 360) % 360;
        ring.style.setProperty('--angle', ang + 'deg');
        wrapper.style.transform = `translate(${(mx - cx) / cx * 4}px, ${(my - cy) / cy * 4}px)`;
    });
    wrapper.addEventListener('mouseenter', () => {
        ring.style.boxShadow = '0 0 16px 6px rgba(255,165,0,0.6)';
    });
    wrapper.addEventListener('mouseleave', () => {
        ring.style.boxShadow = 'none';
        wrapper.style.transform = 'translate(0,0)';
    });
});

// Chart switching logic
{chart_scripts}