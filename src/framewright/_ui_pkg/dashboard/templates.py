"""HTML templates for the FrameWright dashboard.

This module contains all HTML, CSS, and JavaScript templates as Python strings
for the web dashboard. Uses only Python standard library features.
"""

from typing import Dict, Any, List, Optional

# CSS Styles - Dark theme with responsive design
DASHBOARD_CSS = """
:root {
    --bg-primary: #0f0f1a;
    --bg-secondary: #1a1a2e;
    --bg-card: #16213e;
    --bg-card-hover: #1f2f4e;
    --text-primary: #f5f5f5;
    --text-secondary: #a0a0b0;
    --text-muted: #666680;
    --accent: #e94560;
    --accent-light: #ff6b8a;
    --success: #4ecca3;
    --warning: #ffd93d;
    --error: #ff6b6b;
    --info: #5eb5f7;
    --border-color: rgba(255, 255, 255, 0.08);
    --shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    --transition: all 0.2s ease;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html {
    font-size: 16px;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                 Oxygen, Ubuntu, Cantarell, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    line-height: 1.5;
}

/* Layout */
.app-container {
    display: flex;
    min-height: 100vh;
}

/* Sidebar */
.sidebar {
    width: 260px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    position: fixed;
    height: 100vh;
    z-index: 100;
}

.sidebar-header {
    padding: 1.5rem;
    border-bottom: 1px solid var(--border-color);
}

.logo {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
}

.logo span {
    color: var(--accent);
}

.sidebar-nav {
    flex: 1;
    padding: 1rem 0;
}

.nav-item {
    display: flex;
    align-items: center;
    padding: 0.75rem 1.5rem;
    color: var(--text-secondary);
    text-decoration: none;
    transition: var(--transition);
    cursor: pointer;
}

.nav-item:hover,
.nav-item.active {
    background: rgba(233, 69, 96, 0.1);
    color: var(--accent);
    border-right: 3px solid var(--accent);
}

.nav-item svg {
    width: 20px;
    height: 20px;
    margin-right: 12px;
}

.sidebar-footer {
    padding: 1rem 1.5rem;
    border-top: 1px solid var(--border-color);
    font-size: 0.85rem;
    color: var(--text-muted);
}

/* Main Content */
.main-content {
    flex: 1;
    margin-left: 260px;
    min-height: 100vh;
}

.header {
    background: var(--bg-secondary);
    padding: 1rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border-color);
    position: sticky;
    top: 0;
    z-index: 50;
}

.header-title {
    font-size: 1.25rem;
    font-weight: 600;
}

.header-actions {
    display: flex;
    gap: 1rem;
    align-items: center;
}

.status-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: rgba(78, 204, 163, 0.1);
    border-radius: 20px;
    font-size: 0.875rem;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--success);
    animation: pulse 2s ease-in-out infinite;
}

.status-dot.disconnected {
    background: var(--error);
    animation: none;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.content {
    padding: 2rem;
}

/* Cards */
.card {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
    transition: var(--transition);
}

.card:hover {
    background: var(--bg-card-hover);
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-color);
}

.card-title {
    font-size: 0.875rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
}

.card-body {
    padding: 0.5rem 0;
}

/* Grid */
.grid {
    display: grid;
    gap: 1.5rem;
}

.grid-2 {
    grid-template-columns: repeat(2, 1fr);
}

.grid-3 {
    grid-template-columns: repeat(3, 1fr);
}

.grid-4 {
    grid-template-columns: repeat(4, 1fr);
}

/* Stats */
.stat-card {
    display: flex;
    flex-direction: column;
    padding: 1.25rem;
}

.stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}

.stat-value.accent { color: var(--accent); }
.stat-value.success { color: var(--success); }
.stat-value.warning { color: var(--warning); }
.stat-value.error { color: var(--error); }

.stat-change {
    font-size: 0.75rem;
    margin-top: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.25rem;
}

.stat-change.positive { color: var(--success); }
.stat-change.negative { color: var(--error); }

/* Progress bars */
.progress-container {
    margin: 0.5rem 0;
}

.progress-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
    font-size: 0.875rem;
}

.progress-bar {
    width: 100%;
    height: 8px;
    background: var(--bg-primary);
    border-radius: 4px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
    background: linear-gradient(90deg, var(--accent), var(--accent-light));
}

.progress-fill.success {
    background: linear-gradient(90deg, var(--success), #6fe8c0);
}

.progress-fill.warning {
    background: linear-gradient(90deg, var(--warning), #ffe86b);
}

.progress-fill.error {
    background: linear-gradient(90deg, var(--error), #ff9999);
}

/* Metric rows */
.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 0;
    border-bottom: 1px solid var(--border-color);
}

.metric-row:last-child {
    border-bottom: none;
}

.metric-name {
    color: var(--text-secondary);
    font-size: 0.9rem;
}

.metric-value {
    font-weight: 600;
    font-size: 0.9rem;
}

/* Job list */
.job-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.job-card {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 1.25rem;
    border-left: 4px solid var(--accent);
    transition: var(--transition);
}

.job-card:hover {
    transform: translateX(4px);
}

.job-card.processing { border-left-color: var(--info); }
.job-card.completed { border-left-color: var(--success); }
.job-card.failed { border-left-color: var(--error); }
.job-card.pending { border-left-color: var(--warning); }
.job-card.cancelled { border-left-color: var(--text-muted); }

.job-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.75rem;
}

.job-title {
    font-weight: 600;
    font-size: 0.95rem;
    word-break: break-all;
}

.job-id {
    font-family: monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    background: var(--bg-primary);
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
}

.job-status {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.job-status.processing { background: rgba(94, 181, 247, 0.2); color: var(--info); }
.job-status.completed { background: rgba(78, 204, 163, 0.2); color: var(--success); }
.job-status.failed { background: rgba(255, 107, 107, 0.2); color: var(--error); }
.job-status.pending { background: rgba(255, 217, 61, 0.2); color: var(--warning); }
.job-status.cancelled { background: rgba(102, 102, 128, 0.2); color: var(--text-muted); }

.job-details {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-top: 1rem;
}

.job-detail-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.job-detail-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    color: var(--text-muted);
}

.job-actions {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
}

/* Buttons */
.btn {
    padding: 0.5rem 1rem;
    border-radius: 6px;
    border: none;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: var(--transition);
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

.btn-primary {
    background: var(--accent);
    color: white;
}

.btn-primary:hover {
    background: var(--accent-light);
}

.btn-secondary {
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
}

.btn-secondary:hover {
    background: var(--bg-card-hover);
}

.btn-danger {
    background: rgba(255, 107, 107, 0.1);
    color: var(--error);
    border: 1px solid var(--error);
}

.btn-danger:hover {
    background: rgba(255, 107, 107, 0.2);
}

.btn-sm {
    padding: 0.35rem 0.75rem;
    font-size: 0.75rem;
}

/* Tables */
.table-container {
    overflow-x: auto;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th, td {
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
}

th {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    font-weight: 600;
}

tr:hover {
    background: rgba(255, 255, 255, 0.02);
}

/* Forms */
.form-group {
    margin-bottom: 1.25rem;
}

.form-label {
    display: block;
    margin-bottom: 0.5rem;
    font-size: 0.875rem;
    color: var(--text-secondary);
}

.form-input {
    width: 100%;
    padding: 0.75rem 1rem;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 0.9rem;
    transition: var(--transition);
}

.form-input:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(233, 69, 96, 0.1);
}

.form-select {
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23a0a0b0' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 1rem center;
    padding-right: 2.5rem;
}

/* Log viewer */
.log-viewer {
    background: var(--bg-primary);
    border-radius: 6px;
    padding: 1rem;
    font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
    font-size: 0.8rem;
    max-height: 400px;
    overflow-y: auto;
    line-height: 1.6;
}

.log-entry {
    padding: 0.25rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
}

.log-time {
    color: var(--text-muted);
    margin-right: 0.5rem;
}

.log-level {
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-size: 0.7rem;
    margin-right: 0.5rem;
}

.log-level.info { background: rgba(94, 181, 247, 0.2); color: var(--info); }
.log-level.warn { background: rgba(255, 217, 61, 0.2); color: var(--warning); }
.log-level.error { background: rgba(255, 107, 107, 0.2); color: var(--error); }
.log-level.debug { background: rgba(102, 102, 128, 0.2); color: var(--text-muted); }

/* Model cards */
.model-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem;
}

.model-card {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 1.25rem;
    border: 1px solid var(--border-color);
    transition: var(--transition);
}

.model-card:hover {
    border-color: var(--accent);
}

.model-name {
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.model-type {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
}

.model-info {
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.model-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.75rem;
}

.model-status.loaded {
    color: var(--success);
}

.model-status.available {
    color: var(--text-muted);
}

/* Modal */
.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
    opacity: 0;
    visibility: hidden;
    transition: var(--transition);
}

.modal-overlay.active {
    opacity: 1;
    visibility: visible;
}

.modal {
    background: var(--bg-card);
    border-radius: 12px;
    width: 100%;
    max-width: 600px;
    max-height: 90vh;
    overflow-y: auto;
    transform: scale(0.9);
    transition: var(--transition);
}

.modal-overlay.active .modal {
    transform: scale(1);
}

.modal-header {
    padding: 1.5rem;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.modal-title {
    font-size: 1.1rem;
    font-weight: 600;
}

.modal-close {
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 1.5rem;
    line-height: 1;
}

.modal-close:hover {
    color: var(--text-primary);
}

.modal-body {
    padding: 1.5rem;
}

.modal-footer {
    padding: 1rem 1.5rem;
    border-top: 1px solid var(--border-color);
    display: flex;
    justify-content: flex-end;
    gap: 1rem;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: var(--text-muted);
}

.empty-state svg {
    width: 64px;
    height: 64px;
    margin-bottom: 1rem;
    opacity: 0.5;
}

.empty-state h3 {
    margin-bottom: 0.5rem;
    color: var(--text-secondary);
}

/* Responsive */
@media (max-width: 1200px) {
    .grid-4 { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 992px) {
    .grid-3 { grid-template-columns: repeat(2, 1fr); }
    .sidebar { width: 220px; }
    .main-content { margin-left: 220px; }
}

@media (max-width: 768px) {
    .sidebar {
        transform: translateX(-100%);
        transition: transform 0.3s ease;
    }
    .sidebar.open { transform: translateX(0); }
    .main-content { margin-left: 0; }
    .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; }
    .job-details { grid-template-columns: 1fr; }
    .header { padding: 1rem; }
    .content { padding: 1rem; }
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
    background: var(--bg-card);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--bg-card-hover);
}

/* Utility classes */
.text-muted { color: var(--text-muted); }
.text-success { color: var(--success); }
.text-warning { color: var(--warning); }
.text-error { color: var(--error); }
.text-info { color: var(--info); }
.text-center { text-align: center; }
.mt-1 { margin-top: 0.5rem; }
.mt-2 { margin-top: 1rem; }
.mb-1 { margin-bottom: 0.5rem; }
.mb-2 { margin-bottom: 1rem; }
.hidden { display: none; }
"""

# SVG Icons
ICONS = {
    "dashboard": '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M4 13h6a1 1 0 0 0 1-1V4a1 1 0 0 0-1-1H4a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1zm-1 7a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-4a1 1 0 0 0-1-1H4a1 1 0 0 0-1 1v4zm10 0a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-8a1 1 0 0 0-1-1h-6a1 1 0 0 0-1 1v8zm1-17a1 1 0 0 0-1 1v4a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4a1 1 0 0 0-1-1h-6z"/></svg>',
    "jobs": '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 14l-5-5 1.41-1.41L12 14.17l4.59-4.58L18 11l-6 6z"/></svg>',
    "models": '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>',
    "system": '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M19.43 12.98c.04-.32.07-.64.07-.98s-.03-.66-.07-.98l2.11-1.65c.19-.15.24-.42.12-.64l-2-3.46c-.12-.22-.39-.3-.61-.22l-2.49 1c-.52-.4-1.08-.73-1.69-.98l-.38-2.65C14.46 2.18 14.25 2 14 2h-4c-.25 0-.46.18-.49.42l-.38 2.65c-.61.25-1.17.59-1.69.98l-2.49-1c-.23-.09-.49 0-.61.22l-2 3.46c-.13.22-.07.49.12.64l2.11 1.65c-.04.32-.07.65-.07.98s.03.66.07.98l-2.11 1.65c-.19.15-.24.42-.12.64l2 3.46c.12.22.39.3.61.22l2.49-1c.52.4 1.08.73 1.69.98l.38 2.65c.03.24.24.42.49.42h4c.25 0 .46-.18.49-.42l.38-2.65c.61-.25 1.17-.59 1.69-.98l2.49 1c.23.09.49 0 .61-.22l2-3.46c.12-.22.07-.49-.12-.64l-2.11-1.65zM12 15.5c-1.93 0-3.5-1.57-3.5-3.5s1.57-3.5 3.5-3.5 3.5 1.57 3.5 3.5-1.57 3.5-3.5 3.5z"/></svg>',
    "logs": '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/></svg>',
    "play": '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>',
    "stop": '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 6h12v12H6z"/></svg>',
    "refresh": '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>',
    "plus": '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg>',
    "trash": '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>',
    "check": '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>',
    "close": '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>',
    "menu": '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z"/></svg>',
    "folder": '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>',
    "download": '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>',
    "gpu": '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 5h10v2h3v9h-3v2H7v-2H4V7h3V5zm8 4H9v6h6V9z"/></svg>',
}


def icon(name: str) -> str:
    """Get an SVG icon by name."""
    return ICONS.get(name, "")


# JavaScript code
DASHBOARD_JS = """
// FrameWright Dashboard JavaScript

class Dashboard {
    constructor() {
        this.wsConnection = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;
        this.refreshInterval = null;
        this.currentPage = 'dashboard';

        this.init();
    }

    init() {
        this.setupNavigation();
        this.connectWebSocket();
        this.startPolling();
        this.loadInitialData();
    }

    setupNavigation() {
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const page = item.dataset.page;
                this.navigateTo(page);
            });
        });

        // Mobile menu toggle
        const menuToggle = document.getElementById('menu-toggle');
        if (menuToggle) {
            menuToggle.addEventListener('click', () => {
                document.querySelector('.sidebar').classList.toggle('open');
            });
        }
    }

    navigateTo(page) {
        this.currentPage = page;
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.toggle('active', item.dataset.page === page);
        });
        document.querySelectorAll('.page').forEach(p => {
            p.classList.toggle('hidden', p.id !== `page-${page}`);
        });

        // Load page-specific data
        switch(page) {
            case 'jobs':
                this.loadJobs();
                break;
            case 'models':
                this.loadModels();
                break;
            case 'system':
                this.loadSystemInfo();
                break;
        }
    }

    connectWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`;

        try {
            this.wsConnection = new WebSocket(wsUrl);

            this.wsConnection.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true);
            };

            this.wsConnection.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (e) {
                    console.error('Failed to parse WebSocket message:', e);
                }
            };

            this.wsConnection.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
                this.attemptReconnect();
            };

            this.wsConnection.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        } catch (e) {
            console.error('Failed to create WebSocket:', e);
            this.updateConnectionStatus(false);
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Reconnecting... attempt ${this.reconnectAttempts}`);
            setTimeout(() => this.connectWebSocket(), this.reconnectDelay);
        }
    }

    handleWebSocketMessage(data) {
        if (data.type === 'system_update') {
            this.updateSystemMetrics(data.data);
        } else if (data.type === 'job_update') {
            this.updateJob(data.data);
        } else if (data.type === 'jobs_list') {
            this.renderJobs(data.data);
        } else if (data.type === 'log') {
            this.addLogEntry(data.data);
        }
    }

    updateConnectionStatus(connected) {
        const dot = document.getElementById('connection-dot');
        const text = document.getElementById('connection-text');
        if (dot) {
            dot.classList.toggle('disconnected', !connected);
        }
        if (text) {
            text.textContent = connected ? 'Connected' : 'Disconnected';
        }
    }

    startPolling() {
        // Fallback polling if WebSocket fails
        this.refreshInterval = setInterval(() => {
            if (!this.wsConnection || this.wsConnection.readyState !== WebSocket.OPEN) {
                this.loadSystemInfo();
                if (this.currentPage === 'jobs' || this.currentPage === 'dashboard') {
                    this.loadJobs();
                }
            }
        }, 5000);
    }

    async loadInitialData() {
        await Promise.all([
            this.loadSystemInfo(),
            this.loadJobs(),
            this.loadStats()
        ]);
    }

    async apiCall(endpoint, options = {}) {
        try {
            const response = await fetch(endpoint, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            });
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            return await response.json();
        } catch (e) {
            console.error(`API call failed: ${endpoint}`, e);
            return null;
        }
    }

    async loadSystemInfo() {
        const data = await this.apiCall('/system');
        if (data) {
            this.updateSystemMetrics(data);
        }
    }

    updateSystemMetrics(data) {
        // CPU
        this.updateMetric('cpu-usage', `${data.cpu_percent?.toFixed(1) || 0}%`);
        this.updateProgress('cpu-bar', data.cpu_percent || 0);

        // RAM
        const ramUsed = data.ram_used_gb?.toFixed(1) || 0;
        const ramTotal = data.ram_total_gb?.toFixed(1) || 0;
        const ramPercent = ramTotal > 0 ? (ramUsed / ramTotal * 100) : 0;
        this.updateMetric('ram-usage', `${ramUsed} / ${ramTotal} GB`);
        this.updateProgress('ram-bar', ramPercent);

        // VRAM
        const vramUsed = data.vram_used_gb?.toFixed(1) || 0;
        const vramTotal = data.vram_total_gb?.toFixed(1) || 0;
        const vramPercent = vramTotal > 0 ? (vramUsed / vramTotal * 100) : 0;
        this.updateMetric('vram-usage', `${vramUsed} / ${vramTotal} GB`);
        this.updateProgress('vram-bar', vramPercent);

        // GPU
        this.updateMetric('gpu-name', data.gpu_name || 'N/A');
        this.updateMetric('gpu-temp', `${data.gpu_temp || 0}C`);

        // Update timestamp
        this.updateMetric('last-update', new Date().toLocaleTimeString());
    }

    updateMetric(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    }

    updateProgress(id, percent) {
        const el = document.getElementById(id);
        if (el) el.style.width = `${Math.min(100, percent)}%`;
    }

    async loadJobs() {
        const data = await this.apiCall('/jobs');
        if (data && data.jobs) {
            this.renderJobs(data.jobs);
            this.updateJobStats(data.jobs);
        }
    }

    renderJobs(jobs) {
        const container = document.getElementById('job-list');
        if (!container) return;

        if (!jobs || jobs.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    ${ICON_FOLDER}
                    <h3>No Jobs</h3>
                    <p>Submit a new job to get started</p>
                </div>
            `;
            return;
        }

        container.innerHTML = jobs.map(job => this.renderJobCard(job)).join('');

        // Add event listeners to job action buttons
        container.querySelectorAll('.btn-cancel-job').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const jobId = e.target.closest('.job-card').dataset.jobId;
                this.cancelJob(jobId);
            });
        });

        container.querySelectorAll('.btn-view-job').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const jobId = e.target.closest('.job-card').dataset.jobId;
                this.viewJobDetails(jobId);
            });
        });
    }

    renderJobCard(job) {
        const progress = job.total_frames > 0
            ? (job.frames_processed / job.total_frames * 100).toFixed(1)
            : 0;

        const eta = this.calculateETA(job);
        const inputFile = job.input_path?.split(/[\\\\/]/).pop() || 'Unknown';

        return `
            <div class="job-card ${job.state}" data-job-id="${job.job_id}">
                <div class="job-header">
                    <div>
                        <div class="job-title">${inputFile}</div>
                        <div class="job-id">${job.job_id}</div>
                    </div>
                    <span class="job-status ${job.state}">${job.state}</span>
                </div>
                <div class="progress-container">
                    <div class="progress-label">
                        <span>Progress</span>
                        <span>${progress}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill ${job.state === 'completed' ? 'success' : ''}"
                             style="width: ${progress}%"></div>
                    </div>
                </div>
                <div class="job-details">
                    <div class="job-detail-item">
                        <span class="job-detail-label">Frames</span>
                        <span>${job.frames_processed} / ${job.total_frames}</span>
                    </div>
                    <div class="job-detail-item">
                        <span class="job-detail-label">Avg Time</span>
                        <span>${job.avg_frame_time_ms?.toFixed(0) || '--'} ms</span>
                    </div>
                    <div class="job-detail-item">
                        <span class="job-detail-label">ETA</span>
                        <span>${eta}</span>
                    </div>
                </div>
                ${job.state === 'processing' || job.state === 'pending' ? `
                    <div class="job-actions">
                        <button class="btn btn-secondary btn-sm btn-view-job">View Details</button>
                        <button class="btn btn-danger btn-sm btn-cancel-job">Cancel</button>
                    </div>
                ` : `
                    <div class="job-actions">
                        <button class="btn btn-secondary btn-sm btn-view-job">View Details</button>
                    </div>
                `}
            </div>
        `;
    }

    calculateETA(job) {
        if (job.state !== 'processing' || !job.avg_frame_time_ms || job.avg_frame_time_ms <= 0) {
            return '--';
        }
        const remaining = job.total_frames - job.frames_processed;
        const msRemaining = remaining * job.avg_frame_time_ms;
        const seconds = Math.floor(msRemaining / 1000);

        if (seconds < 60) return `${seconds}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${mins}m`;
    }

    updateJobStats(jobs) {
        const active = jobs.filter(j => j.state === 'processing').length;
        const completed = jobs.filter(j => j.state === 'completed').length;
        const pending = jobs.filter(j => j.state === 'pending').length;
        const failed = jobs.filter(j => j.state === 'failed').length;

        this.updateMetric('active-jobs', active);
        this.updateMetric('completed-jobs', completed);
        this.updateMetric('pending-jobs', pending);
        this.updateMetric('failed-jobs', failed);

        // Total frames processed
        const totalFrames = jobs.reduce((sum, j) => sum + (j.frames_processed || 0), 0);
        this.updateMetric('total-frames', totalFrames.toLocaleString());
    }

    async loadStats() {
        // Load additional stats if needed
    }

    async loadModels() {
        const data = await this.apiCall('/models');
        if (data && data.models) {
            this.renderModels(data.models);
        }
    }

    renderModels(models) {
        const container = document.getElementById('model-list');
        if (!container) return;

        if (!models || models.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <h3>No Models</h3>
                    <p>No models are currently available</p>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <div class="model-grid">
                ${models.map(model => `
                    <div class="model-card">
                        <div class="model-name">${model.name}</div>
                        <div class="model-type">${model.type || 'Unknown type'}</div>
                        <div class="model-info">Scale: ${model.scale || 'N/A'}x</div>
                        <div class="model-status ${model.loaded ? 'loaded' : 'available'}">
                            ${model.loaded ? 'Loaded' : 'Available'}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    updateJob(job) {
        const card = document.querySelector(`.job-card[data-job-id="${job.job_id}"]`);
        if (card) {
            // Update in place
            const progress = job.total_frames > 0
                ? (job.frames_processed / job.total_frames * 100).toFixed(1)
                : 0;

            card.className = `job-card ${job.state}`;
            card.querySelector('.job-status').className = `job-status ${job.state}`;
            card.querySelector('.job-status').textContent = job.state;
            card.querySelector('.progress-fill').style.width = `${progress}%`;
            card.querySelector('.progress-label span:last-child').textContent = `${progress}%`;
        } else {
            // Refresh job list
            this.loadJobs();
        }
    }

    async cancelJob(jobId) {
        if (!confirm('Are you sure you want to cancel this job?')) return;

        const result = await this.apiCall(`/jobs/${jobId}`, { method: 'DELETE' });
        if (result && result.success) {
            this.loadJobs();
        }
    }

    async viewJobDetails(jobId) {
        const data = await this.apiCall(`/jobs/${jobId}`);
        if (data) {
            this.showJobModal(data);
        }
    }

    showJobModal(job) {
        const modal = document.getElementById('job-modal');
        if (!modal) return;

        const content = modal.querySelector('.modal-body');
        content.innerHTML = `
            <div class="form-group">
                <label class="form-label">Job ID</label>
                <div class="form-input" readonly>${job.job_id}</div>
            </div>
            <div class="form-group">
                <label class="form-label">Input</label>
                <div class="form-input" readonly>${job.input_path || 'N/A'}</div>
            </div>
            <div class="form-group">
                <label class="form-label">Output</label>
                <div class="form-input" readonly>${job.output_path || 'N/A'}</div>
            </div>
            <div class="form-group">
                <label class="form-label">Status</label>
                <span class="job-status ${job.state}">${job.state}</span>
            </div>
            <div class="form-group">
                <label class="form-label">Progress</label>
                <div>${job.frames_processed} / ${job.total_frames} frames (${job.progress_percent?.toFixed(1) || 0}%)</div>
            </div>
            ${job.error_message ? `
                <div class="form-group">
                    <label class="form-label">Error</label>
                    <div class="text-error">${job.error_message}</div>
                </div>
            ` : ''}
            <div class="form-group">
                <label class="form-label">Created</label>
                <div>${job.created_at ? new Date(job.created_at).toLocaleString() : 'N/A'}</div>
            </div>
        `;

        modal.querySelector('.modal-overlay').classList.add('active');
    }

    hideModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.querySelector('.modal-overlay').classList.remove('active');
        }
    }

    addLogEntry(log) {
        const viewer = document.getElementById('log-viewer');
        if (!viewer) return;

        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.innerHTML = `
            <span class="log-time">${new Date(log.timestamp).toLocaleTimeString()}</span>
            <span class="log-level ${log.level}">${log.level}</span>
            <span>${log.message}</span>
        `;

        viewer.appendChild(entry);
        viewer.scrollTop = viewer.scrollHeight;

        // Keep only last 100 entries
        while (viewer.children.length > 100) {
            viewer.removeChild(viewer.firstChild);
        }
    }

    async submitJob() {
        const inputPath = document.getElementById('job-input-path')?.value;
        if (!inputPath) {
            alert('Please enter an input path');
            return;
        }

        const preset = document.getElementById('job-preset')?.value || 'balanced';
        const scale = parseInt(document.getElementById('job-scale')?.value || '4');

        const result = await this.apiCall('/jobs', {
            method: 'POST',
            body: JSON.stringify({
                input_path: inputPath,
                preset: preset,
                scale: scale
            })
        });

        if (result && result.job_id) {
            this.hideModal('submit-modal');
            this.loadJobs();
            alert(`Job submitted: ${result.job_id}`);
        }
    }
}

// Icon constants for JS
const ICON_FOLDER = `<svg viewBox="0 0 24 24" fill="currentColor" width="64" height="64"><path d="M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>`;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();

    // Modal close handlers
    document.querySelectorAll('.modal-close').forEach(btn => {
        btn.addEventListener('click', () => {
            btn.closest('.modal-overlay').classList.remove('active');
        });
    });

    document.querySelectorAll('.modal-overlay').forEach(overlay => {
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                overlay.classList.remove('active');
            }
        });
    });

    // New job button
    const newJobBtn = document.getElementById('new-job-btn');
    if (newJobBtn) {
        newJobBtn.addEventListener('click', () => {
            document.querySelector('#submit-modal .modal-overlay').classList.add('active');
        });
    }

    // Submit job button
    const submitJobBtn = document.getElementById('submit-job-btn');
    if (submitJobBtn) {
        submitJobBtn.addEventListener('click', () => {
            window.dashboard.submitJob();
        });
    }

    // Refresh button
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            window.dashboard.loadInitialData();
        });
    }
});
"""


def render_dashboard_page(config: Optional[Dict[str, Any]] = None) -> str:
    """Render the main dashboard HTML page.

    Args:
        config: Optional configuration dictionary

    Returns:
        Complete HTML page as string
    """
    config = config or {}
    title = config.get("title", "FrameWright Dashboard")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{DASHBOARD_CSS}</style>
</head>
<body>
    <div class="app-container">
        <!-- Sidebar -->
        <aside class="sidebar">
            <div class="sidebar-header">
                <div class="logo">Frame<span>Wright</span></div>
            </div>
            <nav class="sidebar-nav">
                <a class="nav-item active" data-page="dashboard">
                    {icon("dashboard")}
                    <span>Dashboard</span>
                </a>
                <a class="nav-item" data-page="jobs">
                    {icon("jobs")}
                    <span>Jobs</span>
                </a>
                <a class="nav-item" data-page="models">
                    {icon("models")}
                    <span>Models</span>
                </a>
                <a class="nav-item" data-page="system">
                    {icon("system")}
                    <span>System</span>
                </a>
                <a class="nav-item" data-page="logs">
                    {icon("logs")}
                    <span>Logs</span>
                </a>
            </nav>
            <div class="sidebar-footer">
                FrameWright v1.0
            </div>
        </aside>

        <!-- Main Content -->
        <main class="main-content">
            <header class="header">
                <button id="menu-toggle" class="btn btn-secondary" style="display: none;">
                    {icon("menu")}
                </button>
                <h1 class="header-title">Dashboard</h1>
                <div class="header-actions">
                    <button id="new-job-btn" class="btn btn-primary">
                        {icon("plus")}
                        New Job
                    </button>
                    <button id="refresh-btn" class="btn btn-secondary">
                        {icon("refresh")}
                    </button>
                    <div class="status-indicator">
                        <div class="status-dot" id="connection-dot"></div>
                        <span id="connection-text">Connected</span>
                    </div>
                </div>
            </header>

            <div class="content">
                <!-- Dashboard Page -->
                <div id="page-dashboard" class="page">
                    <!-- Stats Cards -->
                    <div class="grid grid-4">
                        <div class="card stat-card">
                            <span class="stat-label">Active Jobs</span>
                            <span class="stat-value accent" id="active-jobs">0</span>
                        </div>
                        <div class="card stat-card">
                            <span class="stat-label">Completed</span>
                            <span class="stat-value success" id="completed-jobs">0</span>
                        </div>
                        <div class="card stat-card">
                            <span class="stat-label">Pending</span>
                            <span class="stat-value warning" id="pending-jobs">0</span>
                        </div>
                        <div class="card stat-card">
                            <span class="stat-label">Failed</span>
                            <span class="stat-value error" id="failed-jobs">0</span>
                        </div>
                    </div>

                    <div class="grid grid-2">
                        <!-- System Resources -->
                        <div class="card">
                            <div class="card-header">
                                <h2 class="card-title">System Resources</h2>
                                <span class="text-muted" id="last-update">--</span>
                            </div>
                            <div class="card-body">
                                <div class="progress-container">
                                    <div class="progress-label">
                                        <span>CPU</span>
                                        <span id="cpu-usage">--%</span>
                                    </div>
                                    <div class="progress-bar">
                                        <div class="progress-fill" id="cpu-bar" style="width: 0%"></div>
                                    </div>
                                </div>
                                <div class="progress-container">
                                    <div class="progress-label">
                                        <span>RAM</span>
                                        <span id="ram-usage">-- / -- GB</span>
                                    </div>
                                    <div class="progress-bar">
                                        <div class="progress-fill" id="ram-bar" style="width: 0%"></div>
                                    </div>
                                </div>
                                <div class="progress-container">
                                    <div class="progress-label">
                                        <span>VRAM</span>
                                        <span id="vram-usage">-- / -- GB</span>
                                    </div>
                                    <div class="progress-bar">
                                        <div class="progress-fill" id="vram-bar" style="width: 0%"></div>
                                    </div>
                                </div>
                                <div class="metric-row">
                                    <span class="metric-name">GPU</span>
                                    <span class="metric-value" id="gpu-name">--</span>
                                </div>
                                <div class="metric-row">
                                    <span class="metric-name">Temperature</span>
                                    <span class="metric-value" id="gpu-temp">--</span>
                                </div>
                            </div>
                        </div>

                        <!-- Quick Stats -->
                        <div class="card">
                            <div class="card-header">
                                <h2 class="card-title">Processing Stats</h2>
                            </div>
                            <div class="card-body">
                                <div class="metric-row">
                                    <span class="metric-name">Total Frames Processed</span>
                                    <span class="metric-value" id="total-frames">0</span>
                                </div>
                                <div class="metric-row">
                                    <span class="metric-name">Avg Frame Time</span>
                                    <span class="metric-value" id="avg-frame-time">-- ms</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Recent Jobs -->
                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">Recent Jobs</h2>
                        </div>
                        <div class="card-body">
                            <div class="job-list" id="job-list">
                                <div class="empty-state">
                                    {icon("folder")}
                                    <h3>No Jobs</h3>
                                    <p>Submit a new job to get started</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Jobs Page -->
                <div id="page-jobs" class="page hidden">
                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">All Jobs</h2>
                        </div>
                        <div class="card-body">
                            <div class="job-list" id="all-jobs-list">
                                <!-- Jobs loaded dynamically -->
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Models Page -->
                <div id="page-models" class="page hidden">
                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">Available Models</h2>
                        </div>
                        <div class="card-body" id="model-list">
                            <!-- Models loaded dynamically -->
                        </div>
                    </div>
                </div>

                <!-- System Page -->
                <div id="page-system" class="page hidden">
                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">System Information</h2>
                        </div>
                        <div class="card-body" id="system-info">
                            <!-- System info loaded dynamically -->
                        </div>
                    </div>
                </div>

                <!-- Logs Page -->
                <div id="page-logs" class="page hidden">
                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">Processing Logs</h2>
                        </div>
                        <div class="card-body">
                            <div class="log-viewer" id="log-viewer">
                                <!-- Logs appear here -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <!-- Job Details Modal -->
    <div id="job-modal">
        <div class="modal-overlay">
            <div class="modal">
                <div class="modal-header">
                    <h3 class="modal-title">Job Details</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <!-- Job details loaded dynamically -->
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary modal-close">Close</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Submit Job Modal -->
    <div id="submit-modal">
        <div class="modal-overlay">
            <div class="modal">
                <div class="modal-header">
                    <h3 class="modal-title">Submit New Job</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="form-group">
                        <label class="form-label">Input Video Path</label>
                        <input type="text" class="form-input" id="job-input-path"
                               placeholder="/path/to/video.mp4">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Preset</label>
                        <select class="form-input form-select" id="job-preset">
                            <option value="fast">Fast</option>
                            <option value="balanced" selected>Balanced</option>
                            <option value="quality">Quality</option>
                            <option value="ultra">Ultra</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Scale Factor</label>
                        <select class="form-input form-select" id="job-scale">
                            <option value="2">2x</option>
                            <option value="4" selected>4x</option>
                        </select>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary modal-close">Cancel</button>
                    <button class="btn btn-primary" id="submit-job-btn">Submit Job</button>
                </div>
            </div>
        </div>
    </div>

    <script>{DASHBOARD_JS}</script>
</body>
</html>"""


def render_job_card(job: Dict[str, Any]) -> str:
    """Render a single job card HTML.

    Args:
        job: Job data dictionary

    Returns:
        HTML string for the job card
    """
    job_id = job.get("job_id", "unknown")
    state = job.get("state", "pending")
    input_path = job.get("input_path", "Unknown")
    total_frames = job.get("total_frames", 0)
    frames_processed = job.get("frames_processed", 0)
    progress = (frames_processed / total_frames * 100) if total_frames > 0 else 0

    # Extract filename from path
    input_file = input_path.replace("\\", "/").split("/")[-1]

    return f"""
    <div class="job-card {state}" data-job-id="{job_id}">
        <div class="job-header">
            <div>
                <div class="job-title">{input_file}</div>
                <div class="job-id">{job_id}</div>
            </div>
            <span class="job-status {state}">{state}</span>
        </div>
        <div class="progress-container">
            <div class="progress-label">
                <span>Progress</span>
                <span>{progress:.1f}%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill {'success' if state == 'completed' else ''}"
                     style="width: {progress}%"></div>
            </div>
        </div>
        <div class="job-details">
            <div class="job-detail-item">
                <span class="job-detail-label">Frames</span>
                <span>{frames_processed} / {total_frames}</span>
            </div>
        </div>
    </div>
    """


def render_model_card(model: Dict[str, Any]) -> str:
    """Render a single model card HTML.

    Args:
        model: Model data dictionary

    Returns:
        HTML string for the model card
    """
    name = model.get("name", "Unknown")
    model_type = model.get("type", "Unknown")
    scale = model.get("scale", "N/A")
    loaded = model.get("loaded", False)
    status_class = "loaded" if loaded else "available"
    status_text = "Loaded" if loaded else "Available"

    return f"""
    <div class="model-card">
        <div class="model-name">{name}</div>
        <div class="model-type">{model_type}</div>
        <div class="model-info">Scale: {scale}x</div>
        <div class="model-status {status_class}">{status_text}</div>
    </div>
    """


def render_error_page(error_code: int, error_message: str) -> str:
    """Render an error page HTML.

    Args:
        error_code: HTTP error code
        error_message: Error message to display

    Returns:
        Complete HTML page for the error
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error {error_code} - FrameWright</title>
    <style>{DASHBOARD_CSS}</style>
</head>
<body>
    <div style="min-height: 100vh; display: flex; align-items: center; justify-content: center;">
        <div class="card" style="max-width: 500px; text-align: center;">
            <h1 class="stat-value error" style="font-size: 4rem;">{error_code}</h1>
            <h2 style="margin: 1rem 0;">{error_message}</h2>
            <p class="text-muted">Something went wrong. Please try again.</p>
            <a href="/" class="btn btn-primary" style="margin-top: 1.5rem;">Back to Dashboard</a>
        </div>
    </div>
</body>
</html>"""


def render_login_page() -> str:
    """Render the login page HTML.

    Returns:
        Complete HTML page for login
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - FrameWright</title>
    <style>{DASHBOARD_CSS}</style>
</head>
<body>
    <div style="min-height: 100vh; display: flex; align-items: center; justify-content: center;">
        <div class="card" style="max-width: 400px; width: 100%;">
            <div class="card-header">
                <h2 class="card-title">FrameWright Login</h2>
            </div>
            <div class="card-body">
                <form method="POST" action="/login">
                    <div class="form-group">
                        <label class="form-label">API Key</label>
                        <input type="password" name="api_key" class="form-input"
                               placeholder="Enter your API key" required>
                    </div>
                    <button type="submit" class="btn btn-primary" style="width: 100%;">
                        Login
                    </button>
                </form>
            </div>
        </div>
    </div>
</body>
</html>"""
