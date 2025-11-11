(function (global) {
    if (global.Chart) {
        return;
    }

    function toNumber(value) {
        const n = Number(value);
        return Number.isFinite(n) ? n : null;
    }

    class Chart {
        constructor(ctx, config = {}) {
            this.ctx = ctx;
            this.canvas = ctx.canvas;
            this.type = config.type || 'line';
            this.config = config;
            this.data = config.data || { labels: [], datasets: [] };
            this.options = config.options || {};
            this._background = this.canvas.style.background;
            this.update();
        }

        update() {
            this.clear();
            this.draw();
        }

        clear() {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }

        destroy() {
            this.clear();
            this.canvas.style.background = this._background;
        }

        draw() {
            switch ((this.type || '').toLowerCase()) {
                case 'radar':
                    this.drawRadar();
                    break;
                case 'line':
                default:
                    this.drawLine();
                    break;
            }
        }

        drawLine() {
            const ctx = this.ctx;
            const width = this.canvas.width;
            const height = this.canvas.height;
            const margin = 24;
            const dataset = (this.data.datasets && this.data.datasets[0]) || { data: [] };
            const values = (dataset.data || []).map(toNumber).filter((n) => n !== null);
            ctx.save();
            ctx.clearRect(0, 0, width, height);

            ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
            ctx.lineWidth = 1;
            const steps = 5;
            for (let i = 0; i <= steps; i += 1) {
                const y = margin + ((height - margin * 2) * i) / steps;
                ctx.beginPath();
                ctx.moveTo(margin, y);
                ctx.lineTo(width - margin, y);
                ctx.stroke();
            }

            if (!values.length) {
                ctx.restore();
                return;
            }

            const min = Math.min(...values);
            const max = Math.max(...values);
            const range = max - min || 1;
            const points = (dataset.data || []).map((value, index) => {
                const val = toNumber(value);
                const x = margin + ((width - margin * 2) * (index / Math.max(1, dataset.data.length - 1)));
                const ratio = val === null ? 0.5 : (val - min) / range;
                const y = height - margin - ratio * (height - margin * 2);
                return { x, y, valid: val !== null };
            });

            if (dataset.backgroundColor) {
                ctx.fillStyle = dataset.backgroundColor;
                ctx.beginPath();
                points.forEach((point, index) => {
                    if (index === 0) {
                        ctx.moveTo(point.x, point.y);
                    } else {
                        ctx.lineTo(point.x, point.y);
                    }
                });
                ctx.lineTo(points[points.length - 1].x, height - margin);
                ctx.lineTo(points[0].x, height - margin);
                ctx.closePath();
                ctx.fill();
            }

            ctx.strokeStyle = dataset.borderColor || '#00d4ff';
            ctx.lineWidth = dataset.borderWidth || 2;
            ctx.beginPath();
            points.forEach((point, index) => {
                if (!point.valid) {
                    return;
                }
                if (index === 0) {
                    ctx.moveTo(point.x, point.y);
                } else {
                    ctx.lineTo(point.x, point.y);
                }
            });
            ctx.stroke();
            ctx.restore();
        }

        drawRadar() {
            const ctx = this.ctx;
            const width = this.canvas.width;
            const height = this.canvas.height;
            const radius = Math.min(width, height) / 2 - 24;
            const centerX = width / 2;
            const centerY = height / 2;
            const dataset = (this.data.datasets && this.data.datasets[0]) || { data: [] };
            const values = (dataset.data || []).map((value) => {
                const v = toNumber(value);
                return Number.isFinite(v) ? v : 0;
            });
            const labels = this.data.labels || [];
            const count = Math.max(values.length, labels.length, 3);

            ctx.save();
            ctx.clearRect(0, 0, width, height);
            ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
            ctx.lineWidth = 1;
            const rings = 4;
            for (let i = 1; i <= rings; i += 1) {
                const r = (radius * i) / rings;
                ctx.beginPath();
                ctx.arc(centerX, centerY, r, 0, Math.PI * 2);
                ctx.stroke();
            }

            const angleStep = (Math.PI * 2) / count;
            ctx.beginPath();
            for (let i = 0; i < count; i += 1) {
                const angle = angleStep * i - Math.PI / 2;
                const x = centerX + Math.cos(angle) * radius;
                const y = centerY + Math.sin(angle) * radius;
                ctx.moveTo(centerX, centerY);
                ctx.lineTo(x, y);
            }
            ctx.stroke();

            ctx.beginPath();
            ctx.fillStyle = dataset.backgroundColor || 'rgba(0, 212, 255, 0.1)';
            ctx.strokeStyle = dataset.borderColor || '#00d4ff';
            ctx.lineWidth = dataset.borderWidth || 2;
            const maxValue = Math.max(...values, 1);
            for (let i = 0; i < count; i += 1) {
                const angle = angleStep * i - Math.PI / 2;
                const raw = values[i] ?? 0;
                const ratio = maxValue ? raw / maxValue : 0;
                const r = radius * ratio;
                const x = centerX + Math.cos(angle) * r;
                const y = centerY + Math.sin(angle) * r;
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
            ctx.restore();
        }
    }

    global.Chart = Chart;
})(typeof window !== 'undefined' ? window : globalThis);
