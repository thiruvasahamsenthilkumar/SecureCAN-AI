// SecureCAN AI - Frontend Dashboard Script

let messageCount = 0;
let attackCount = 0;
let detectionChart, attackTypeChart;
const alerts = [];

// Initialize Charts
function initCharts() {
    // Detection Timeline Chart
    const detectionCtx = document.getElementById('detectionChart').getContext('2d');
    detectionChart = new Chart(detectionCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Normal Traffic',
                    data: [],
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 3,
                    pointHoverRadius: 6,
                    pointBackgroundColor: '#00d4ff'
                },
                {
                    label: 'Attacks Detected',
                    data: [],
                    borderColor: '#ff006e',
                    backgroundColor: 'rgba(255, 0, 110, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 3,
                    pointHoverRadius: 6,
                    pointBackgroundColor: '#ff006e'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: {
                        color: '#e0e0e0',
                        font: { size: 12 }
                    }
                }
            },
            scales: {
                y: {
                    ticks: { color: '#888' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    beginAtZero: true
                },
                x: {
                    ticks: { color: '#888' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            }
        }
    });

    // Attack Type Distribution Chart
    const attackCtx = document.getElementById('attackTypeChart').getContext('2d');
    attackTypeChart = new Chart(attackCtx, {
        type: 'doughnut',
        data: {
            labels: ['DoS', 'Fuzzy', 'Fake RPM', 'Fake Gear'],
            datasets: [{
                data: [2847, 892, 546, 423],
                backgroundColor: [
                    'rgba(255, 0, 0, 0.6)',
                    'rgba(255, 165, 0, 0.6)',
                    'rgba(255, 0, 110, 0.6)',
                    'rgba(0, 212, 255, 0.6)'
                ],
                borderColor: [
                    '#ff0000',
                    '#ffa500',
                    '#ff006e',
                    '#00d4ff'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: {
                        color: '#e0e0e0',
                        font: { size: 12 },
                        padding: 15
                    }
                }
            }
        }
    });
}

// Simulate Real-Time Data
function simulateRealTimeData() {
    setInterval(() => {
        messageCount += Math.floor(Math.random() * 50) + 20;

        // 3-5% chance of attack
        if (Math.random() < 0.04) {
            attackCount++;
            addAlert();
        }

        // Update stats
        document.getElementById('totalMessages').textContent = messageCount.toLocaleString();
        document.getElementById('attackCount').textContent = attackCount.toLocaleString();

        // Update detection rate
        const detectionRate = messageCount > 0 ? ((messageCount - attackCount) / messageCount * 100).toFixed(1) : 0;
        document.getElementById('detectionRate').textContent = detectionRate + '%';

        // Update chart every 10 iterations
        if (messageCount % 500 === 0) {
            updateCharts();
        }
    }, 1000);
}

// Update Charts
function updateCharts() {
    const timeLabel = new Date().toLocaleTimeString();
    const normalCount = messageCount - attackCount;

    detectionChart.data.labels.push(timeLabel);
    detectionChart.data.datasets[0].data.push(normalCount);
    detectionChart.data.datasets[1].data.push(attackCount);

    // Keep only last 20 data points
    if (detectionChart.data.labels.length > 20) {
        detectionChart.data.labels.shift();
        detectionChart.data.datasets[0].data.shift();
        detectionChart.data.datasets[1].data.shift();
    }

    detectionChart.update();
}

// Attack Types
const attackTypes = [
    {
        type: 'DoS (Denial of Service)',
        description: 'High-frequency message flood detected',
        confidence: 99
    },
    {
        type: 'Fuzzy Injection',
        description: 'Malformed CAN message detected',
        confidence: 96
    },
    {
        type: 'Fake RPM Spoofing',
        description: 'Anomalous engine RPM signal detected',
        confidence: 98
    },
    {
        type: 'Fake Gear Signal',
        description: 'Transmission gear inconsistency detected',
        confidence: 94
    }
];

// Add Alert
function addAlert() {
    const attack = attackTypes[Math.floor(Math.random() * attackTypes.length)];
    const alertTime = new Date().toLocaleTimeString();
    const canID = '0x' + Math.floor(Math.random() * 0x800).toString(16).toUpperCase().padStart(3, '0');
    const latency = (Math.random() * 4 + 1).toFixed(2);

    const alert = {
        time: alertTime,
        type: attack.type,
        description: attack.description,
        confidence: attack.confidence,
        canID: canID,
        latency: latency
    };

    alerts.unshift(alert);

    // Keep only last 50 alerts
    if (alerts.length > 50) {
        alerts.pop();
    }

    renderAlerts();
}

// Render Alerts
function renderAlerts() {
    const container = document.getElementById('alertsContainer');
    container.innerHTML = '';

    if (alerts.length === 0) {
        container.innerHTML = '<div class="alert-placeholder"><p>Waiting for CAN bus data...</p></div>';
        return;
    }

    alerts.forEach(alert => {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert-item ${alert.type.includes('DoS') ? 'danger' : 'warning'}`;
        alertDiv.innerHTML = `
            <div class="alert-time">⏰ ${alert.time}</div>
            <div class="alert-message">🚨 ${alert.type}</div>
            <div class="alert-details">
                <div><strong>CAN ID:</strong> ${alert.canID}</div>
                <div><strong>Confidence:</strong> ${alert.confidence}%</div>
                <div><strong>Details:</strong> ${alert.description}</div>
                <div><strong>Latency:</strong> ${alert.latency}ms</div>
            </div>
        `;
        container.appendChild(alertDiv);
    });
}

// Clear Alerts
function clearAlerts() {
    alerts.length = 0;
    renderAlerts();
}

// Simulate Network Status Changes
function simulateNetworkStatus() {
    setInterval(() => {
        const ecus = ['ecu1', 'ecu2', 'ecu3', 'ecu4'];
        const randomEcu = ecus[Math.floor(Math.random() * ecus.length)];
        const ecuElement = document.getElementById(randomEcu).querySelector('.node-status');

        // 1% chance of alert
        if (Math.random() < 0.01) {
            ecuElement.textContent = 'Alert';
            ecuElement.classList.remove('safe');
            ecuElement.classList.add('alert');

            setTimeout(() => {
                ecuElement.textContent = 'Safe';
                ecuElement.classList.remove('alert');
                ecuElement.classList.add('safe');
            }, 3000);
        }
    }, 5000);
}

// Update Latency Display
function updateLatency() {
    setInterval(() => {
        const latency = (Math.random() * 4 + 1).toFixed(2);
        document.getElementById('avgLatency').textContent = latency + 'ms';
    }, 2000);
}

// Initialize on Page Load
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 SecureCAN AI Dashboard Initialized');

    initCharts();
    simulateRealTimeData();
    simulateNetworkStatus();
    updateLatency();
});

// Keyboard Shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'c' && e.ctrlKey) {
        clearAlerts();
        console.log('Alerts cleared');
    }
});
