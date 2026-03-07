"""Dashboard page HTML template for Code Bridge server management."""

from __future__ import annotations


def render_dashboard_html() -> str:
    """Render the main dashboard HTML page."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Bridge Dashboard</title>
    <style>
        :root {
            --bg-primary: #f5f5f5;
            --bg-secondary: #ffffff;
            --bg-card: #ffffff;
            --text-primary: #333333;
            --text-secondary: #666666;
            --text-muted: #888888;
            --border-color: #e0e0e0;
            --accent-color: #007bff;
            --accent-hover: #0056b3;
            --success-color: #28a745;
            --warning-color: #ffc107;
            --danger-color: #dc3545;
            --shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --bg-primary: #1a1a1a;
                --bg-secondary: #2d2d2d;
                --bg-card: #2d2d2d;
                --text-primary: #e0e0e0;
                --text-secondary: #b0b0b0;
                --text-muted: #808080;
                --border-color: #404040;
                --shadow: 0 2px 8px rgba(0,0,0,0.3);
            }
        }

        * {
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
        }

        .header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 16px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header h1 {
            margin: 0;
            font-size: 1.5rem;
            font-weight: 600;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 500;
        }

        .status-badge.online {
            background: rgba(40, 167, 69, 0.15);
            color: var(--success-color);
        }

        .status-badge.offline {
            background: rgba(220, 53, 69, 0.15);
            color: var(--danger-color);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 24px;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 24px;
        }

        .card {
            background: var(--bg-card);
            border-radius: 12px;
            box-shadow: var(--shadow);
            overflow: hidden;
        }

        .card.full-width {
            grid-column: 1 / -1;
        }

        .card-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .card-header h2 {
            margin: 0;
            font-size: 1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .card-content {
            padding: 16px 20px;
        }

        .card-actions {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .btn-primary {
            background: var(--accent-color);
            color: white;
        }

        .btn-primary:hover {
            background: var(--accent-hover);
        }

        .btn-secondary {
            background: var(--bg-primary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }

        .btn-secondary:hover {
            background: var(--border-color);
        }

        .btn-danger {
            background: var(--danger-color);
            color: white;
        }

        .btn-danger:hover {
            background: #c82333;
        }

        .btn-success {
            background: var(--success-color);
            color: white;
        }

        .btn-success:hover {
            background: #218838;
        }

        .btn-sm {
            padding: 4px 10px;
            font-size: 0.75rem;
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .info-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid var(--border-color);
        }

        .info-row:last-child {
            border-bottom: none;
        }

        .info-label {
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        .info-value {
            font-weight: 500;
            font-size: 0.875rem;
        }

        .info-value.mono {
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            background: var(--bg-primary);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
        }

        .info-value a {
            color: var(--accent-color);
            text-decoration: none;
        }

        .info-value a:hover {
            text-decoration: underline;
        }

        .list-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid var(--border-color);
        }

        .list-item:last-child {
            border-bottom: none;
        }

        .list-item-info {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .list-item-name {
            font-weight: 500;
        }

        .list-item-meta {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .select-wrapper {
            position: relative;
        }

        select {
            appearance: none;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 8px 32px 8px 12px;
            font-size: 0.875rem;
            color: var(--text-primary);
            cursor: pointer;
            width: 100%;
        }

        select:focus {
            outline: none;
            border-color: var(--accent-color);
        }

        .select-wrapper::after {
            content: '';
            position: absolute;
            right: 12px;
            top: 50%;
            transform: translateY(-50%);
            border: 4px solid transparent;
            border-top-color: var(--text-secondary);
        }

        .empty-state {
            text-align: center;
            padding: 24px;
            color: var(--text-muted);
        }

        .section-desc {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin: 0 0 12px 0;
            line-height: 1.4;
        }

        .connected-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }

        .connected-indicator.active {
            background: var(--success-color);
        }

        .connected-indicator.inactive {
            background: var(--text-muted);
        }

        /* Checkbox */
        .checkbox-wrapper {
            position: relative;
            cursor: pointer;
            display: flex;
            align-items: center;
        }

        .checkbox-wrapper input {
            position: absolute;
            opacity: 0;
            cursor: pointer;
            height: 0;
            width: 0;
        }

        .checkmark {
            position: relative;
            height: 20px;
            width: 20px;
            background-color: var(--bg-primary);
            border: 2px solid var(--border-color);
            border-radius: 4px;
            transition: all 0.2s;
        }

        .checkbox-wrapper:hover input ~ .checkmark {
            border-color: var(--accent-color);
        }

        .checkbox-wrapper input:checked ~ .checkmark {
            background-color: var(--success-color);
            border-color: var(--success-color);
        }

        .checkmark:after {
            content: "";
            position: absolute;
            display: none;
            left: 5px;
            top: 1px;
            width: 5px;
            height: 10px;
            border: solid white;
            border-width: 0 2px 2px 0;
            transform: rotate(45deg);
        }

        .checkbox-wrapper input:checked ~ .checkmark:after {
            display: block;
        }

        /* Server card sections */
        .server-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
        }

        .server-section {
            padding: 16px;
            background: var(--bg-primary);
            border-radius: 8px;
        }

        .server-section h3 {
            margin: 0 0 12px 0;
            font-size: 0.875rem;
            color: var(--text-secondary);
            font-weight: 600;
        }

        /* Modal */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }

        .modal-overlay.show {
            display: flex;
        }

        .modal {
            background: var(--bg-card);
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            max-width: 440px;
            width: 90%;
            max-height: 90vh;
            overflow-y: auto;
        }

        .modal-header {
            padding: 20px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .modal-header h3 {
            margin: 0;
            font-size: 1.125rem;
        }

        .modal-close {
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: var(--text-muted);
            padding: 0;
            line-height: 1;
        }

        .modal-close:hover {
            color: var(--text-primary);
        }

        .modal-body {
            padding: 24px;
            text-align: center;
        }

        .modal-body p {
            color: var(--text-secondary);
            margin: 0 0 16px 0;
        }

        .qr-code {
            margin: 16px 0;
            background: white;
            padding: 16px;
            border-radius: 12px;
            display: inline-block;
        }

        .qr-code img {
            max-width: 240px;
            width: 100%;
            height: auto;
            display: block;
        }

        .timer {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--warning-color);
            margin: 16px 0;
        }

        .timer.expired {
            color: var(--danger-color);
        }

        /* LLM Settings Modal */
        .form-group {
            margin-bottom: 16px;
            text-align: left;
        }

        .form-group label {
            display: block;
            font-size: 0.875rem;
            font-weight: 500;
            margin-bottom: 6px;
            color: var(--text-secondary);
        }

        .form-group select {
            width: 100%;
        }

        .modal-footer {
            padding: 16px 24px;
            border-top: 1px solid var(--border-color);
            display: flex;
            justify-content: flex-end;
            gap: 8px;
        }

        /* Toast notification */
        .toast {
            position: fixed;
            bottom: 24px;
            right: 24px;
            background: var(--bg-card);
            border-radius: 8px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            padding: 16px 20px;
            display: flex;
            align-items: center;
            gap: 12px;
            z-index: 2000;
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.3s;
        }

        .toast.show {
            transform: translateY(0);
            opacity: 1;
        }

        .toast.success {
            border-left: 4px solid var(--success-color);
        }

        .toast.error {
            border-left: 4px solid var(--danger-color);
        }

        /* Loading spinner */
        .loading {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid var(--border-color);
            border-top-color: var(--accent-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        .loading-lg {
            width: 32px;
            height: 32px;
            border-width: 3px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Add button in header */
        .btn-add {
            width: 22px;
            height: 22px;
            padding: 0;
            border: 1px dashed var(--text-muted);
            border-radius: 4px;
            background: transparent;
            color: var(--text-muted);
            font-size: 1rem;
            font-weight: 300;
            line-height: 1;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-add:hover {
            border-color: var(--accent-color);
            color: var(--accent-color);
            background: rgba(0, 123, 255, 0.1);
        }

        /* Excluded folder style */
        .list-item.excluded .list-item-name {
            color: var(--text-muted);
            text-decoration: line-through;
        }

        .btn-icon-toggle {
            background: none;
            border: none;
            font-size: 1.1rem;
            color: var(--text-muted);
            cursor: pointer;
            padding: 4px 8px;
        }

        .btn-icon-toggle:hover {
            color: var(--danger-color);
        }

        .btn-icon-toggle.excluded {
            color: var(--text-muted);
        }

        .btn-icon-toggle.excluded:hover {
            color: var(--success-color);
        }

        /* Tabs for modal */
        .modal-tabs {
            display: flex;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 16px;
        }

        .modal-tab {
            flex: 1;
            padding: 12px 16px;
            background: none;
            border: none;
            border-bottom: 2px solid transparent;
            cursor: pointer;
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-secondary);
            transition: all 0.2s;
        }

        .modal-tab:hover {
            color: var(--text-primary);
        }

        .modal-tab.active {
            color: var(--accent-color);
            border-bottom-color: var(--accent-color);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        /* Code input */
        .code-input-wrapper {
            display: flex;
            gap: 8px;
            justify-content: center;
            margin: 24px 0;
        }

        .code-display-wrapper {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin: 20px 0;
        }

        .code-digit-display {
            width: 48px;
            height: 56px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.75rem;
            font-weight: 700;
            border: 2px solid var(--accent-color);
            border-radius: 8px;
            background: var(--bg-secondary);
            color: var(--text-primary);
        }

        /* Folder Browser Modal */
        .folder-browser-modal {
            max-width: 560px;
        }

        .folder-browser-header {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 12px 16px;
            background: var(--bg-primary);
            border-radius: 8px;
            margin-bottom: 12px;
        }

        .folder-browser-header .drive-select {
            flex-shrink: 0;
            padding: 6px 10px;
            font-size: 0.875rem;
        }

        .folder-browser-path {
            flex: 1;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            font-size: 0.8rem;
            color: var(--text-secondary);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .folder-browser-nav {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
        }

        .folder-browser-list {
            max-height: 320px;
            overflow-y: auto;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background: var(--bg-primary);
        }

        .folder-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 14px;
            cursor: pointer;
            transition: background 0.15s;
            border-bottom: 1px solid var(--border-color);
        }

        .folder-item:last-child {
            border-bottom: none;
        }

        .folder-item:hover {
            background: var(--bg-secondary);
        }

        .folder-item.selected {
            background: rgba(0, 123, 255, 0.15);
        }

        .folder-item.disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .folder-item-icon {
            font-size: 1.25rem;
        }

        .folder-item-name {
            flex: 1;
            font-size: 0.9rem;
        }

        .folder-item-arrow {
            color: var(--text-muted);
            font-size: 0.875rem;
        }

        .folder-browser-selection {
            margin-top: 12px;
            padding: 10px 14px;
            background: var(--bg-primary);
            border-radius: 8px;
            font-size: 0.8rem;
        }

        .folder-browser-selection-label {
            color: var(--text-muted);
            margin-bottom: 4px;
        }

        .folder-browser-selection-path {
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            color: var(--accent-color);
            word-break: break-all;
        }

        .quick-access-list {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 12px;
        }

        .quick-access-btn {
            padding: 6px 12px;
            font-size: 0.75rem;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            cursor: pointer;
            color: var(--text-secondary);
            transition: all 0.2s;
        }

        .quick-access-btn:hover {
            background: var(--bg-secondary);
            border-color: var(--accent-color);
            color: var(--text-primary);
        }

        .folder-browser-loading {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 200px;
        }

        .folder-browser-empty {
            text-align: center;
            padding: 40px;
            color: var(--text-muted);
        }

        .log-toolbar {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }

        .log-lines-select {
            width: auto;
            min-width: 88px;
            padding-right: 24px;
        }

        .log-autoscroll-label {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        .log-autoscroll-label input {
            accent-color: var(--accent-color);
        }

        .log-meta {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 10px;
        }

        .log-view {
            margin: 0;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            font-size: 0.75rem;
            line-height: 1.45;
            min-height: 180px;
            max-height: 360px;
            overflow: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .log-view.log-empty {
            color: var(--text-muted);
        }

        /* Login Page */
        .login-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: var(--bg-primary);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        }

        .login-overlay.hidden {
            display: none;
        }

        .login-card {
            background: var(--bg-card);
            border-radius: 16px;
            box-shadow: var(--shadow);
            padding: 40px;
            max-width: 400px;
            width: 90%;
            text-align: center;
        }

        .login-card h2 {
            margin: 0 0 8px 0;
            font-size: 1.5rem;
        }

        .login-card .subtitle {
            color: var(--text-muted);
            margin: 0 0 24px 0;
            font-size: 0.875rem;
        }

        .login-form {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .login-form input[type="password"] {
            width: 100%;
            padding: 12px 16px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 1rem;
            background: var(--bg-primary);
            color: var(--text-primary);
        }

        .login-form input[type="password"]:focus {
            outline: none;
            border-color: var(--accent-color);
        }

        .login-error {
            color: var(--danger-color);
            font-size: 0.875rem;
            margin: 0;
            min-height: 1.25em;
        }

        /* Password Status Badge */
        .password-status-badge {
            font-size: 0.7rem;
            padding: 2px 8px;
            border-radius: 12px;
            margin-top: 2px;
        }

        .password-status-badge.enabled {
            background: rgba(40, 167, 69, 0.15);
            color: var(--success-color);
        }

        .password-status-badge.disabled {
            background: rgba(136, 136, 136, 0.15);
            color: var(--text-muted);
        }

        /* Access Mode Cards (Radio-style) */
        .access-mode-cards {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .access-mode-card {
            display: block;
            padding: 14px 16px;
            border: 2px solid var(--border-color);
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s;
            background: var(--bg-primary);
        }

        .access-mode-card:hover:not(.disabled) {
            border-color: var(--accent-color);
        }

        .access-mode-card.active {
            border-color: var(--success-color);
            background: rgba(40, 167, 69, 0.08);
        }

        .access-mode-card.disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .access-mode-card.is-loading {
            opacity: 0.7;
            pointer-events: none;
        }

        .access-mode-card.is-loading .access-mode-title::after {
            content: '';
            display: inline-block;
            width: 12px;
            height: 12px;
            margin-left: 8px;
            border: 2px solid var(--border-color);
            border-top-color: var(--accent-color);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        .access-mode-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 4px;
        }

        /* Real radio button - hidden native, custom visual */
        .access-mode-radio {
            appearance: none;
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border: 2px solid var(--border-color);
            border-radius: 50%;
            position: relative;
            flex-shrink: 0;
            transition: all 0.2s;
            cursor: pointer;
            margin: 0;
            padding: 0;
            background: var(--bg-secondary);
        }

        .access-mode-radio:checked {
            border-color: var(--success-color);
        }

        .access-mode-radio:checked::after {
            content: '';
            position: absolute;
            top: 3px;
            left: 3px;
            width: 8px;
            height: 8px;
            background: var(--success-color);
            border-radius: 50%;
        }

        .access-mode-card.active .access-mode-radio {
            border-color: var(--success-color);
        }

        .access-mode-card.active .access-mode-radio::after {
            content: '';
            position: absolute;
            top: 3px;
            left: 3px;
            width: 8px;
            height: 8px;
            background: var(--success-color);
            border-radius: 50%;
        }

        .access-mode-title {
            font-weight: 600;
            font-size: 0.9rem;
        }

        .access-mode-desc {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-left: 28px;
        }

        .access-mode-card.active .access-mode-title {
            color: var(--success-color);
        }

        /* Tunnel Action Section inside Access Mode Card */
        .tunnel-action-section {
            margin-top: 10px;
            margin-left: 28px;
            display: none;
        }

        .access-mode-card.active .tunnel-action-section {
            display: block;
        }

        /* Language Selector */
        .lang-selector {
            display: flex;
            align-items: center;
            gap: 4px;
            background: var(--bg-primary);
            border-radius: 6px;
            padding: 2px;
        }

        .lang-btn {
            padding: 4px 10px;
            border: none;
            background: transparent;
            color: var(--text-muted);
            font-size: 0.75rem;
            font-weight: 500;
            cursor: pointer;
            border-radius: 4px;
            transition: all 0.2s;
        }

        .lang-btn:hover {
            color: var(--text-primary);
        }

        .lang-btn.active {
            background: var(--accent-color);
            color: white;
        }

        /* Empty State with Action */
        .empty-state-action {
            text-align: center;
            padding: 24px 16px;
            color: var(--text-muted);
        }

        .empty-state-action p {
            margin: 0 0 12px 0;
        }

        .empty-state-action .btn {
            margin-top: 8px;
        }

        /* Pairing Banner */
        .pairing-banner {
            background: linear-gradient(135deg, var(--accent-color) 0%, #0056b3 100%);
            color: white;
            padding: 16px 24px;
            display: none;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 24px;
            border-radius: 12px;
            box-shadow: var(--shadow);
        }

        .pairing-banner.visible {
            display: flex;
        }

        .pairing-banner-content {
            display: flex;
            align-items: center;
            gap: 12px;
            flex: 1;
        }

        .pairing-banner-icon {
            font-size: 1.5rem;
        }

        .pairing-banner-text h3 {
            margin: 0;
            font-size: 1rem;
            font-weight: 600;
        }

        .pairing-banner-text p {
            margin: 4px 0 0 0;
            font-size: 0.875rem;
            opacity: 0.9;
        }

        .pairing-banner-actions {
            display: flex;
            align-items: center;
            gap: 12px;
            flex-shrink: 0;
        }

        .pairing-banner .btn {
            background: white;
            color: var(--accent-color);
            border: none;
            white-space: nowrap;
        }

        .pairing-banner .btn:hover {
            background: rgba(255,255,255,0.9);
        }

        .app-store-links {
            display: flex;
            gap: 8px;
        }

        .app-store-link {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            background: rgba(255,255,255,0.15);
            border-radius: 6px;
            color: white;
            text-decoration: none;
            font-size: 0.75rem;
            font-weight: 500;
            transition: background 0.2s;
        }

        .app-store-link:hover {
            background: rgba(255,255,255,0.25);
        }

        .app-store-link svg {
            flex-shrink: 0;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .header {
                flex-direction: column;
                gap: 12px;
                text-align: center;
            }

            .container {
                padding: 16px;
            }

            .grid {
                grid-template-columns: 1fr;
            }

            .server-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <!-- Login Overlay -->
    <div id="loginOverlay" class="login-overlay hidden">
        <div class="login-card">
            <h2>🔐 Dashboard Login</h2>
            <p class="subtitle">Enter password to access the dashboard</p>
            <form class="login-form" onsubmit="handleLogin(event)">
                <input type="password" id="loginPassword" placeholder="Password" autocomplete="current-password" required>
                <p id="loginError" class="login-error"></p>
                <button type="submit" class="btn btn-primary" style="width: 100%;">Login</button>
            </form>
        </div>
    </div>

    <header class="header">
        <h1>Code Bridge Dashboard</h1>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div class="lang-selector">
                <button class="lang-btn active" onclick="setLanguage('en')">EN</button>
                <button class="lang-btn" onclick="setLanguage('ko')">한국어</button>
            </div>
            <div id="serverStatus" class="status-badge online">
                <span class="status-dot"></span>
                <span data-i18n="online">Online</span>
            </div>
            <button id="logoutBtn" class="btn btn-secondary btn-sm" onclick="handleLogout()" style="display: none;" data-i18n="logout">Logout</button>
        </div>
    </header>

    <main class="container">
        <!-- Pairing Banner (shown when no devices paired) -->
        <div id="pairingBanner" class="pairing-banner">
            <div class="pairing-banner-content">
                <span class="pairing-banner-icon">📱</span>
                <div class="pairing-banner-text">
                    <h3 data-i18n="get_started">Get Started</h3>
                    <p data-i18n="pairing_banner_desc">Connect your device by scanning the QR code with the Code Bridge app.</p>
                </div>
            </div>
            <div class="pairing-banner-actions">
                <div class="app-store-links">
                    <a href="https://play.google.com/store/apps/details?id=com.mkideabox.codeBridge" target="_blank" class="app-store-link" title="Google Play">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M3,20.5V3.5C3,2.91 3.34,2.39 3.84,2.15L13.69,12L3.84,21.85C3.34,21.6 3,21.09 3,20.5M16.81,15.12L6.05,21.34L14.54,12.85L16.81,15.12M20.16,10.81C20.5,11.08 20.75,11.5 20.75,12C20.75,12.5 20.53,12.9 20.18,13.18L17.89,14.5L15.39,12L17.89,9.5L20.16,10.81M6.05,2.66L16.81,8.88L14.54,11.15L6.05,2.66Z"/></svg>
                        Android
                    </a>
                    <a href="https://apps.apple.com/app/code-bridge/id6740008029" target="_blank" class="app-store-link" title="App Store">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M18.71,19.5C17.88,20.74 17,21.95 15.66,21.97C14.32,22 13.89,21.18 12.37,21.18C10.84,21.18 10.37,21.95 9.1,22C7.79,22.05 6.8,20.68 5.96,19.47C4.25,17 2.94,12.45 4.7,9.39C5.57,7.87 7.13,6.91 8.82,6.88C10.1,6.86 11.32,7.75 12.11,7.75C12.89,7.75 14.37,6.68 15.92,6.84C16.57,6.87 18.39,7.1 19.56,8.82C19.47,8.88 17.39,10.1 17.41,12.63C17.44,15.65 20.06,16.66 20.09,16.67C20.06,16.74 19.67,18.11 18.71,19.5M13,3.5C13.73,2.67 14.94,2.04 15.94,2C16.07,3.17 15.6,4.35 14.9,5.19C14.21,6.04 13.07,6.7 11.95,6.61C11.8,5.46 12.36,4.26 13,3.5Z"/></svg>
                        iOS
                    </a>
                </div>
                <button class="btn" onclick="showPairingModal()" data-i18n="open_qr_pairing">Open QR Pairing</button>
            </div>
        </div>

        <div class="grid">
            <!-- Server Card -->
            <div class="card">
                <div class="card-header">
                    <h2>Server</h2>
                    <button class="btn btn-secondary btn-sm" onclick="checkUpdate()">Check Update</button>
                </div>
                <div class="card-content">
                    <div class="info-row">
                        <span class="info-label">Version</span>
                        <span id="serverVersion" class="info-value">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Dashboard</span>
                        <span id="dashboardUrl" class="info-value mono">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">API (Local)</span>
                        <span id="apiLocalUrl" class="info-value mono">-</span>
                    </div>
                    <div class="info-row" id="tunnelUrlRow" style="display: none;">
                        <span class="info-label">API (Tunnel)</span>
                        <span id="serverTunnelUrl" class="info-value mono">-</span>
                    </div>
                </div>
            </div>

            <!-- Security Card -->
            <div class="card">
                <div class="card-header">
                    <h2>🔒 <span data-i18n="security">Security</span></h2>
                </div>
                <div class="card-content">
                    <p class="section-desc" data-i18n="security_desc">Select access mode. QR pairing is required by default.</p>
                    <!-- Access Mode Selection (Mutually Exclusive Radio Buttons) -->
                    <div class="access-mode-cards">
                        <!-- IP Login Option -->
                        <label class="access-mode-card" id="ipLoginCard">
                            <div class="access-mode-header">
                                <input type="radio" name="accessMode" value="ip" class="access-mode-radio" onchange="handleAccessModeChange('ip')">
                                <span class="access-mode-title" data-i18n="ip_login">IP Login</span>
                            </div>
                            <span class="access-mode-desc" data-i18n="ip_login_desc">⚠️ Direct access from local network without QR pairing</span>
                        </label>

                        <!-- External Access Option (Default) -->
                        <label class="access-mode-card" id="externalAccessCard">
                            <div class="access-mode-header">
                                <input type="radio" name="accessMode" value="external" class="access-mode-radio" onchange="handleAccessModeChange('external')" checked>
                                <span class="access-mode-title" data-i18n="external_access">External Access</span>
                            </div>
                            <span class="access-mode-desc" data-i18n="external_access_desc">Access via tunnel URL from anywhere</span>
                            <div id="tunnelActionSection" class="tunnel-action-section">
                                <button id="refreshTunnelBtn" class="btn btn-secondary btn-sm" onclick="refreshTunnel(); event.stopPropagation();" data-i18n="refresh_url">Refresh URL</button>
                            </div>
                        </label>
                    </div>

                    <!-- Dashboard Password -->
                    <div class="info-row" style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border-color);">
                        <div style="display: flex; flex-direction: column; gap: 2px;">
                            <span class="info-label" style="font-weight: 500;">Dashboard Password</span>
                            <span id="passwordStatusBadge" class="password-status-badge disabled">Not Set</span>
                        </div>
                        <button class="btn btn-secondary btn-sm" onclick="showPasswordModal()">Settings</button>
                    </div>
                </div>
            </div>

            <!-- Paired Devices Card -->
            <div class="card">
                <div class="card-header">
                    <h2>Paired Devices</h2>
                    <button class="btn-add" onclick="showPairingModal()" title="Add Device">+</button>
                </div>
                <div class="card-content">
                    <div id="clientList"></div>
                </div>
            </div>

            <!-- Available LLM Card -->
            <div class="card">
                <div class="card-header">
                    <h2>Available LLM</h2>
                </div>
                <div class="card-content">
                    <p class="section-desc">Detects installed CLI: claude, codex</p>
                    <div id="llmList"></div>
                </div>
            </div>

            <!-- Accessible Folders Card -->
            <div class="card">
                <div class="card-header">
                    <h2>Accessible Folders</h2>
                    <button class="btn-add" onclick="showAddFolderModal()" title="Add Folder">+</button>
                </div>
                <div class="card-content">
                    <div id="folderList"></div>
                </div>
            </div>

            <!-- Available Devices Card -->
            <div class="card">
                <div class="card-header">
                    <h2>Available Devices</h2>
                </div>
                <div class="card-content">
                    <div id="deviceList"></div>
                </div>
            </div>

            <!-- Server Logs Card (Bottom Full Width) -->
            <div class="card full-width">
                <div class="card-header">
                    <h2>Server Logs</h2>
                    <div class="card-actions">
                        <select id="serverLogLines" class="log-lines-select" onchange="refreshServerLog(false)">
                            <option value="100">100 lines</option>
                            <option value="200" selected>200 lines</option>
                            <option value="400">400 lines</option>
                            <option value="800">800 lines</option>
                        </select>
                        <label class="log-autoscroll-label">
                            <input id="serverLogAutoScroll" type="checkbox" checked>
                            Auto-scroll
                        </label>
                        <button class="btn btn-secondary btn-sm" onclick="refreshServerLog(true)">Refresh</button>
                    </div>
                </div>
                <div class="card-content">
                    <div id="serverLogMeta" class="log-meta">Loading log metadata...</div>
                    <pre id="serverLogContent" class="log-view log-empty">Loading logs...</pre>
                </div>
            </div>
        </div>
    </main>

    <!-- Pairing Modal (QR + Code) -->
    <div id="pairingModal" class="modal-overlay" onclick="closePairingModal(event)">
        <div class="modal" onclick="event.stopPropagation()">
            <div class="modal-header">
                <h3>Pair New Device</h3>
                <button class="modal-close" onclick="closePairingModal()">&times;</button>
            </div>
            <div class="modal-body">
                <div class="modal-tabs">
                    <button class="modal-tab active" onclick="switchPairingTab('qr')">QR Code</button>
                    <button class="modal-tab" onclick="switchPairingTab('code')">Enter Code</button>
                </div>

                <!-- QR Tab -->
                <div id="pairingTabQr" class="tab-content active">
                    <p>Scan this QR code with the Code Bridge app</p>
                    <div id="qrTimer" class="timer">5:00</div>
                    <div id="qrCode" class="qr-code">
                        <div class="loading loading-lg"></div>
                    </div>
                    <div style="margin-top: 16px;">
                        <button class="btn btn-secondary" onclick="refreshQr()">Refresh QR</button>
                    </div>
                </div>

                <!-- Code Tab -->
                <div id="pairingTabCode" class="tab-content">
                    <p>Enter this code in the Code Bridge app</p>
                    <div id="codeTimer" class="timer">5:00</div>
                    <div id="pairingCodeDisplay" class="code-display-wrapper">
                        <span class="code-digit-display">-</span>
                        <span class="code-digit-display">-</span>
                        <span class="code-digit-display">-</span>
                        <span class="code-digit-display">-</span>
                        <span class="code-digit-display">-</span>
                        <span class="code-digit-display">-</span>
                    </div>
                    <div style="margin-top: 16px;">
                        <button class="btn btn-secondary" onclick="refreshQr()">Refresh Code</button>
                    </div>
                    <p style="color: var(--text-secondary); margin-top: 12px; font-size: 0.875rem;">
                        Open Code Bridge app → Settings → Connect to Server → Enter Code
                    </p>
                </div>
            </div>
        </div>
    </div>

    <!-- Folder Browser Modal -->
    <div id="folderBrowserModal" class="modal-overlay" onclick="closeFolderBrowserModal(event)">
        <div class="modal folder-browser-modal" onclick="event.stopPropagation()">
            <div class="modal-header">
                <h3>Select Folder</h3>
                <button class="modal-close" onclick="closeFolderBrowserModal()">&times;</button>
            </div>
            <div class="modal-body" style="text-align: left;">
                <!-- Current Path -->
                <div class="folder-browser-header">
                    <select id="folderDriveSelect" class="drive-select" style="display: none;" onchange="onDriveChange(this.value)"></select>
                    <span class="folder-browser-path" id="folderCurrentPath">Loading...</span>
                </div>

                <!-- Quick Access -->
                <div id="quickAccessList" class="quick-access-list"></div>

                <!-- Navigation -->
                <div class="folder-browser-nav">
                    <button class="btn btn-secondary btn-sm" id="folderUpBtn" onclick="navigateUp()">↑ Parent</button>
                    <button class="btn btn-secondary btn-sm" onclick="navigateHome()">Home</button>
                </div>

                <!-- Folder List -->
                <div class="folder-browser-list" id="folderBrowserList">
                    <div class="folder-browser-loading">
                        <div class="loading loading-lg"></div>
                    </div>
                </div>

                <!-- Selection Display -->
                <div class="folder-browser-selection" id="folderSelection" style="display: none;">
                    <div class="folder-browser-selection-label">Selected:</div>
                    <div class="folder-browser-selection-path" id="folderSelectedPath"></div>
                </div>
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="closeFolderBrowserModal()">Cancel</button>
                <button class="btn btn-primary" id="folderAddBtn" onclick="confirmFolderSelection()" disabled>Add This Folder</button>
            </div>
        </div>
    </div>


    <!-- Password Settings Modal -->
    <div id="passwordModal" class="modal-overlay" onclick="closePasswordModal(event)">
        <div class="modal" style="max-width: 380px;" onclick="event.stopPropagation()">
            <div class="modal-header">
                <h3>🔐 Dashboard Password</h3>
                <button class="modal-close" onclick="closePasswordModal()">&times;</button>
            </div>
            <div class="modal-body" style="text-align: left;">
                <!-- Set Password (when no password) -->
                <div id="setPasswordSection" style="display: none;">
                    <p class="section-desc">Protect dashboard access with a password. Only applies to local network access.</p>
                    <div class="form-group">
                        <label>New Password</label>
                        <input type="password" id="newPassword" placeholder="Min 4 characters" style="width: 100%; padding: 10px 12px; border: 1px solid var(--border-color); border-radius: 6px; background: var(--bg-primary); color: var(--text-primary);">
                    </div>
                    <div class="form-group">
                        <label>Confirm Password</label>
                        <input type="password" id="confirmPassword" placeholder="Re-enter password" style="width: 100%; padding: 10px 12px; border: 1px solid var(--border-color); border-radius: 6px; background: var(--bg-primary); color: var(--text-primary);">
                    </div>
                </div>
                <!-- Change/Remove Password (when password set) -->
                <div id="changePasswordSection" style="display: none;">
                    <p class="section-desc">Change or remove the dashboard password.</p>
                    <div class="form-group">
                        <label>Current Password</label>
                        <input type="password" id="currentPasswordForChange" placeholder="Enter current password" style="width: 100%; padding: 10px 12px; border: 1px solid var(--border-color); border-radius: 6px; background: var(--bg-primary); color: var(--text-primary);">
                    </div>
                    <div class="form-group">
                        <label>New Password (optional)</label>
                        <input type="password" id="newPasswordForChange" placeholder="Leave empty to remove" style="width: 100%; padding: 10px 12px; border: 1px solid var(--border-color); border-radius: 6px; background: var(--bg-primary); color: var(--text-primary);">
                    </div>
                </div>
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="closePasswordModal()">Cancel</button>
                <button id="setPasswordBtn" class="btn btn-primary" onclick="setPassword()" style="display: none;">Set Password</button>
                <button id="updatePasswordBtn" class="btn btn-primary" onclick="changePassword()" style="display: none;">Update</button>
                <button id="removePasswordBtn" class="btn btn-danger" onclick="removePassword()" style="display: none;">Remove</button>
            </div>
        </div>
    </div>

    <!-- Toast -->
    <div id="toast" class="toast"></div>

    <script>
        // i18n Translations
        const i18n = {
            en: {
                // Header
                online: 'Online',
                offline: 'Offline',
                logout: 'Logout',
                // Security
                security: 'Security',
                security_desc: 'Select access mode. QR pairing is required by default.',
                ip_login: 'IP Login',
                ip_login_desc: '⚠️ Direct access from local network without QR pairing',
                external_access: 'External Access',
                external_access_desc: 'Access via tunnel URL from anywhere',
                refresh_url: 'Refresh URL',
                // Pairing Banner
                get_started: 'Get Started',
                pairing_banner_desc: 'Connect your device by scanning the QR code with the Code Bridge app.',
                // Paired Devices
                no_paired_devices: 'No paired devices',
                no_paired_devices_desc: 'Scan the QR code with the Code Bridge app to connect.',
                open_qr_pairing: 'Open QR Pairing',
                connected: 'Connected',
                offline: 'Offline',
                // Common
                remove: 'Remove',
                cancel: 'Cancel',
                save: 'Save',
                settings: 'Settings',
                check_update: 'Check Update'
            },
            ko: {
                // Header
                online: '온라인',
                offline: '오프라인',
                logout: '로그아웃',
                // Security
                security: '보안',
                security_desc: '접속 모드를 선택하세요. 기본적으로 QR 페어링이 필요합니다.',
                ip_login: 'IP 로그인 허용',
                ip_login_desc: '⚠️ QR 페어링 없이 로컬 네트워크에서 바로 접속',
                external_access: '외부접속 허용',
                external_access_desc: '인터넷 어디서든 터널 URL로 접속 가능',
                refresh_url: 'URL 새로고침',
                // Pairing Banner
                get_started: '시작하기',
                pairing_banner_desc: 'Code Bridge 앱으로 QR 코드를 스캔하여 디바이스를 연결하세요.',
                // Paired Devices
                no_paired_devices: '연결된 디바이스 없음',
                no_paired_devices_desc: 'Code Bridge 앱으로 QR 코드를 스캔하여 연결하세요.',
                open_qr_pairing: 'QR 페어링 열기',
                connected: '연결됨',
                offline: '오프라인',
                // Common
                remove: '삭제',
                cancel: '취소',
                save: '저장',
                settings: '설정',
                check_update: '업데이트 확인'
            }
        };

        let currentLang = localStorage.getItem('dashboard_lang') || 'en';

        function setLanguage(lang) {
            currentLang = lang;
            localStorage.setItem('dashboard_lang', lang);
            applyTranslations();
            updateLangButtons();
            // Re-render dynamic content
            if (dashboardData) {
                renderClients(dashboardData.clients || []);
            }
        }

        function t(key) {
            return i18n[currentLang]?.[key] || i18n['en']?.[key] || key;
        }

        function applyTranslations() {
            document.querySelectorAll('[data-i18n]').forEach(el => {
                const key = el.getAttribute('data-i18n');
                el.textContent = t(key);
            });
        }

        function updateLangButtons() {
            document.querySelectorAll('.lang-btn').forEach(btn => {
                const lang = btn.textContent === 'EN' ? 'en' : 'ko';
                btn.classList.toggle('active', lang === currentLang);
            });
        }

        // State
        let dashboardData = null;
        let accessibleFolders = [];
        let qrExpiresAt = null;
        let qrTimerInterval = null;
        let pairingPollInterval = null;
        let currentPairingCode = null;
        let serverLogPollInterval = null;
        let authState = {
            passwordEnabled: false,
            authenticated: false
        };

        // API calls
        async function fetchApi(endpoint, options = {}) {
            try {
                const response = await fetch(endpoint, {
                    headers: { 'Content-Type': 'application/json' },
                    ...options
                });
                if (response.status === 401) {
                    // Only show login if password is enabled
                    authState.authenticated = false;
                    if (authState.passwordEnabled) {
                        showLoginOverlay();
                    }
                    return null;
                }
                return await response.json();
            } catch (error) {
                console.error('API error:', error);
                return null;
            }
        }

        // Authentication functions
        async function checkAuthStatus() {
            try {
                const response = await fetch('/api/dashboard/auth/status');
                const data = await response.json();
                authState.passwordEnabled = data.password_enabled;
                authState.authenticated = data.authenticated;
                updateAuthUI();
                return data;
            } catch (error) {
                console.error('Auth check error:', error);
                return null;
            }
        }

        function updateAuthUI() {
            const loginOverlay = document.getElementById('loginOverlay');
            const logoutBtn = document.getElementById('logoutBtn');
            const passwordStatusBadge = document.getElementById('passwordStatusBadge');

            // Show/hide login overlay
            if (authState.passwordEnabled && !authState.authenticated) {
                loginOverlay.classList.remove('hidden');
            } else {
                loginOverlay.classList.add('hidden');
            }

            // Show/hide logout button
            if (authState.passwordEnabled && authState.authenticated) {
                logoutBtn.style.display = 'inline-block';
            } else {
                logoutBtn.style.display = 'none';
            }

            // Update password status badge
            if (authState.passwordEnabled) {
                passwordStatusBadge.textContent = 'Enabled';
                passwordStatusBadge.className = 'password-status-badge enabled';
            } else {
                passwordStatusBadge.textContent = 'Not Set';
                passwordStatusBadge.className = 'password-status-badge disabled';
            }
        }

        function showPasswordModal() {
            const modal = document.getElementById('passwordModal');
            const setPasswordSection = document.getElementById('setPasswordSection');
            const changePasswordSection = document.getElementById('changePasswordSection');
            const setPasswordBtn = document.getElementById('setPasswordBtn');
            const updatePasswordBtn = document.getElementById('updatePasswordBtn');
            const removePasswordBtn = document.getElementById('removePasswordBtn');

            // Clear form fields
            document.getElementById('newPassword').value = '';
            document.getElementById('confirmPassword').value = '';
            document.getElementById('currentPasswordForChange').value = '';
            document.getElementById('newPasswordForChange').value = '';

            // Show appropriate section based on password state
            if (authState.passwordEnabled) {
                setPasswordSection.style.display = 'none';
                changePasswordSection.style.display = 'block';
                setPasswordBtn.style.display = 'none';
                updatePasswordBtn.style.display = 'inline-block';
                removePasswordBtn.style.display = 'inline-block';
            } else {
                setPasswordSection.style.display = 'block';
                changePasswordSection.style.display = 'none';
                setPasswordBtn.style.display = 'inline-block';
                updatePasswordBtn.style.display = 'none';
                removePasswordBtn.style.display = 'none';
            }

            modal.classList.add('show');
        }

        function closePasswordModal(event) {
            if (event && event.target !== event.currentTarget) return;
            document.getElementById('passwordModal').classList.remove('show');
        }

        function showLoginOverlay() {
            document.getElementById('loginOverlay').classList.remove('hidden');
            document.getElementById('loginPassword').focus();
        }

        async function handleLogin(event) {
            event.preventDefault();
            const password = document.getElementById('loginPassword').value;
            const errorEl = document.getElementById('loginError');

            try {
                const response = await fetch('/api/dashboard/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ password })
                });
                const result = await response.json();

                if (result.success) {
                    authState.authenticated = true;
                    document.getElementById('loginOverlay').classList.add('hidden');
                    document.getElementById('loginPassword').value = '';
                    errorEl.textContent = '';
                    updateAuthUI();
                    await fetchDashboard();
                    await refreshServerLog(false);
                } else {
                    errorEl.textContent = result.error || 'Login failed';
                }
            } catch (error) {
                errorEl.textContent = 'Connection error';
            }
        }

        async function handleLogout() {
            try {
                await fetch('/api/dashboard/auth/logout', { method: 'POST' });
                authState.authenticated = false;
                updateAuthUI();
                showToast('Logged out', 'success');
            } catch (error) {
                showToast('Logout failed', 'error');
            }
        }

        async function setPassword() {
            const newPassword = document.getElementById('newPassword').value;
            const confirmPassword = document.getElementById('confirmPassword').value;

            if (newPassword.length < 4) {
                showToast('Password must be at least 4 characters', 'error');
                return;
            }

            if (newPassword !== confirmPassword) {
                showToast('Passwords do not match', 'error');
                return;
            }

            try {
                const response = await fetch('/api/dashboard/auth/password', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ new_password: newPassword })
                });
                const result = await response.json();

                if (result.success) {
                    closePasswordModal();
                    showToast('Password set successfully', 'success');
                    await checkAuthStatus();
                } else {
                    showToast(result.error || result.message || 'Failed to set password', 'error');
                }
            } catch (error) {
                showToast('Connection error', 'error');
            }
        }

        async function changePassword() {
            const currentPassword = document.getElementById('currentPasswordForChange').value;
            const newPassword = document.getElementById('newPasswordForChange').value;

            if (!currentPassword) {
                showToast('Current password required', 'error');
                return;
            }

            if (newPassword && newPassword.length < 4) {
                showToast('New password must be at least 4 characters', 'error');
                return;
            }

            // If new password is empty, confirm removal
            if (!newPassword) {
                if (!confirm('No new password entered. Do you want to remove the password?')) {
                    return;
                }
                await removePasswordWithCurrent(currentPassword);
                return;
            }

            try {
                const response = await fetch('/api/dashboard/auth/password', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        current_password: currentPassword,
                        new_password: newPassword
                    })
                });
                const result = await response.json();

                if (result.success) {
                    closePasswordModal();
                    showToast('Password changed. Please login again.', 'success');
                    await checkAuthStatus();
                } else {
                    showToast(result.error || result.message || 'Failed to change password', 'error');
                }
            } catch (error) {
                showToast('Connection error', 'error');
            }
        }

        async function removePassword() {
            const currentPassword = document.getElementById('currentPasswordForChange').value;
            if (!currentPassword) {
                showToast('Enter current password to remove', 'error');
                return;
            }

            if (!confirm('Are you sure you want to remove the password? Dashboard will be accessible without authentication.')) {
                return;
            }

            await removePasswordWithCurrent(currentPassword);
        }

        async function removePasswordWithCurrent(currentPassword) {
            try {
                const response = await fetch('/api/dashboard/auth/password', {
                    method: 'DELETE',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ current_password: currentPassword })
                });
                const result = await response.json();

                if (result.success) {
                    closePasswordModal();
                    showToast('Password removed', 'success');
                    await checkAuthStatus();
                } else {
                    showToast(result.error || result.message || 'Failed to remove password', 'error');
                }
            } catch (error) {
                showToast('Connection error', 'error');
            }
        }

        // Access mode state
        let accessModeState = {
            ipLogin: false,
            externalAccess: false
        };

        async function fetchDashboard() {
            const [data, foldersData, ipLoginData] = await Promise.all([
                fetchApi('/api/system/overview'),
                fetchApi('/api/filesystem/accessible-folders'),
                fetchApi('/api/system/ip-login')
            ]);
            if (data) {
                dashboardData = data;
            }
            if (foldersData && foldersData.folders) {
                accessibleFolders = foldersData.folders;
            }
            if (ipLoginData) {
                accessModeState.ipLogin = ipLoginData.allow_ip_login === true;
            }
            // External access is determined by tunnel running state
            if (dashboardData && dashboardData.tunnel) {
                accessModeState.externalAccess = dashboardData.tunnel.running === true;
            }
            updateUI();
        }

        async function handleAccessModeChange(mode) {
            const ipLoginCard = document.getElementById('ipLoginCard');
            const externalAccessCard = document.getElementById('externalAccessCard');
            const selectedCard = mode === 'ip' ? ipLoginCard : externalAccessCard;

            // Show loading state
            selectedCard.classList.add('is-loading');
            ipLoginCard.style.pointerEvents = 'none';
            externalAccessCard.style.pointerEvents = 'none';

            try {
                // Mutual exclusivity: only one mode can be active at a time
                if (mode === 'ip') {
                    // Enable IP login, disable external access (stop tunnel)
                    const result = await fetchApi('/api/system/ip-login', {
                        method: 'PUT',
                        body: JSON.stringify({ allow_ip_login: true })
                    });
                    if (result && result.message) {
                        accessModeState.ipLogin = true;
                        showToast(result.message, 'success');
                    } else if (result && result.error) {
                        showToast(result.error, 'error');
                        // Revert radio selection
                        document.querySelector('input[name="accessMode"][value="external"]').checked = true;
                        return;
                    }
                    // Stop tunnel if running (auto)
                    if (accessModeState.externalAccess) {
                        await stopTunnel();
                    }
                    accessModeState.externalAccess = false;
                } else if (mode === 'external') {
                    // Enable external access (start tunnel), disable IP login
                    // First disable IP login
                    const result = await fetchApi('/api/system/ip-login', {
                        method: 'PUT',
                        body: JSON.stringify({ allow_ip_login: false })
                    });
                    if (result && result.message) {
                        accessModeState.ipLogin = false;
                    }
                    // Start tunnel (auto)
                    await startTunnel();
                    accessModeState.externalAccess = true;
                }
            } finally {
                // Remove loading state
                selectedCard.classList.remove('is-loading');
                ipLoginCard.style.pointerEvents = '';
                externalAccessCard.style.pointerEvents = '';
                updateAccessModeUI();
            }
        }

        async function refreshTunnel() {
            const refreshBtn = document.getElementById('refreshTunnelBtn');
            const originalText = refreshBtn.textContent;

            // Show loading state
            refreshBtn.disabled = true;
            refreshBtn.textContent = 'Refreshing...';

            try {
                await fetchApi('/api/system/tunnel/stop', { method: 'POST' });
                const result = await fetchApi('/api/system/tunnel/start', { method: 'POST' });
                await fetchDashboard();
                if (result && result.url) {
                    showToast('New tunnel URL: ' + result.url, 'success');
                } else if (result && result.error) {
                    showToast('Failed: ' + result.error, 'error');
                }
            } finally {
                refreshBtn.disabled = false;
                refreshBtn.textContent = originalText;
            }
        }

        // Legacy function for compatibility
        async function setAccessMode(mode) {
            handleAccessModeChange(mode);
        }

        function updateAccessModeUI() {
            const ipLoginCard = document.getElementById('ipLoginCard');
            const externalAccessCard = document.getElementById('externalAccessCard');
            const refreshTunnelBtn = document.getElementById('refreshTunnelBtn');
            const ipRadio = document.querySelector('input[name="accessMode"][value="ip"]');
            const externalRadio = document.querySelector('input[name="accessMode"][value="external"]');

            // Update state from dashboard data
            const tunnelRunning = dashboardData?.tunnel?.running;

            // Sync radio buttons with actual state
            if (accessModeState.ipLogin) {
                ipRadio.checked = true;
                ipLoginCard.classList.add('active');
                externalAccessCard.classList.remove('active');
            } else {
                ipLoginCard.classList.remove('active');
            }

            if (tunnelRunning) {
                externalRadio.checked = true;
                externalAccessCard.classList.add('active');
                ipLoginCard.classList.remove('active');
                accessModeState.externalAccess = true;
            } else {
                accessModeState.externalAccess = false;
            }
        }

        async function refreshServerLog(notify = false) {
            const linesSelect = document.getElementById('serverLogLines');
            const logMeta = document.getElementById('serverLogMeta');
            const logContent = document.getElementById('serverLogContent');
            const autoScroll = document.getElementById('serverLogAutoScroll');
            if (!linesSelect || !logMeta || !logContent) return;

            const lines = Number.parseInt(linesSelect.value || '200', 10);
            const result = await fetchApi(`/api/system/server-log?lines=${lines}`);
            if (!result) {
                logMeta.textContent = 'Failed to load server logs';
                logContent.textContent = 'Cannot reach log endpoint.';
                logContent.classList.add('log-empty');
                if (notify) showToast('Failed to load logs', 'error');
                return;
            }

            const logText = result.text || '';
            if (logText) {
                logContent.textContent = logText;
                logContent.classList.remove('log-empty');
            } else if (result.exists) {
                logContent.textContent = 'No logs yet.';
                logContent.classList.add('log-empty');
            } else {
                logContent.textContent = 'Log file not found yet. Trigger some requests and refresh.';
                logContent.classList.add('log-empty');
            }

            const pathText = result.path ? `Path: ${result.path}` : 'Path: -';
            const updatedText = result.updated_at ? `Updated: ${result.updated_at}` : 'Updated: -';
            logMeta.textContent = `${pathText} | ${updatedText}`;

            if (autoScroll && autoScroll.checked) {
                logContent.scrollTop = logContent.scrollHeight;
            }

            if (notify) {
                showToast('Logs refreshed', 'success');
            }
        }

        function updateUI() {
            if (!dashboardData) return;

            // Server status
            const statusBadge = document.getElementById('serverStatus');
            statusBadge.className = 'status-badge online';
            statusBadge.innerHTML = '<span class="status-dot"></span><span>Online</span>';

            // Server info
            document.getElementById('serverVersion').textContent = dashboardData.server?.version || '-';

            // Dashboard URL (current page location)
            const dashboardUrl = window.location.origin;
            document.getElementById('dashboardUrl').innerHTML = `<a href="${dashboardUrl}/dashboard" style="color: var(--accent-color); text-decoration: none;">${dashboardUrl}</a>`;

            // API Local URL (from server data)
            const apiLocalUrl = dashboardData.server?.local_url || '-';
            document.getElementById('apiLocalUrl').innerHTML = `<a href="${apiLocalUrl}" target="_blank" style="color: var(--accent-color); text-decoration: none;">${apiLocalUrl}</a>`;

            // Update Tunnel URL in Server card
            const tunnelUrlRow = document.getElementById('tunnelUrlRow');
            const serverTunnelUrl = document.getElementById('serverTunnelUrl');
            const tunnelUrl = dashboardData.tunnel?.url;
            if (tunnelUrl && dashboardData.tunnel?.running) {
                tunnelUrlRow.style.display = 'flex';
                serverTunnelUrl.innerHTML = `<a href="${tunnelUrl}" target="_blank" style="color: var(--accent-color); text-decoration: none;">${tunnelUrl}</a>`;
            } else {
                tunnelUrlRow.style.display = 'none';
            }

            // Update access mode UI (Security section)
            updateAccessModeUI();

            // Clients
            const clients = dashboardData.pairing?.clients || [];
            renderClients(clients);

            // Available LLM
            const companies = dashboardData.llm?.companies || [];
            renderLlmList(companies);

            // Accessible Folders (from dedicated API)
            renderFolders(accessibleFolders);

            // Available Devices
            const devices = dashboardData.devices?.items || [];
            renderDevices(devices);
        }

        function renderLlmList(companies) {
            const container = document.getElementById('llmList');
            const availableCompanies = companies.filter(c => c.connected || c.selectable);
            if (availableCompanies.length === 0) {
                container.innerHTML = '<div class="empty-state">No LLM available</div>';
                return;
            }
            container.innerHTML = availableCompanies.map(company => `
                <div class="list-item">
                    <div class="list-item-info">
                        <span class="list-item-name">
                            <span class="connected-indicator ${company.connected ? 'active' : 'inactive'}"></span>
                            ${company.name}
                        </span>
                        <span class="list-item-meta">${company.connected ? 'Connected' : 'Not connected'}</span>
                    </div>
                    <label class="checkbox-wrapper">
                        <input type="checkbox"
                               ${company.enabled !== false ? 'checked' : ''}
                               onchange="toggleLlmAccess('${company.id}', this.checked)">
                        <span class="checkmark"></span>
                    </label>
                </div>
            `).join('');
        }

        async function toggleLlmAccess(companyId, enabled) {
            await fetchApi('/api/system/llm/access', {
                method: 'PUT',
                body: JSON.stringify({ company_id: companyId, enabled: enabled })
            });
            showToast(enabled ? 'LLM enabled' : 'LLM disabled', 'success');
        }

        function renderClients(clients) {
            const container = document.getElementById('clientList');
            const banner = document.getElementById('pairingBanner');

            if (clients.length === 0) {
                // Show banner at top
                banner.classList.add('visible');
                container.innerHTML = `<div class="empty-state">${t('no_paired_devices')}</div>`;
                return;
            }

            // Hide banner when devices are paired
            banner.classList.remove('visible');
            container.innerHTML = clients.map(client => {
                const isConnected = client.is_connected === true;
                const statusClass = isConnected ? 'active' : 'inactive';
                const statusText = isConnected ? t('connected') : t('offline');
                const firebaseEmail = client.firebase_user?.email;
                const userLine = firebaseEmail
                    ? `<span class="list-item-meta" style="color: var(--accent-color);">${escapeHtml(firebaseEmail)}</span>`
                    : '';
                return `
                    <div class="list-item">
                        <div class="list-item-info">
                            <span class="list-item-name">
                                <span class="connected-indicator ${statusClass}"></span>
                                ${escapeHtml(client.device_name || 'Unknown Device')}
                                <span style="font-weight: normal; font-size: 0.75rem; color: var(--text-muted); margin-left: 8px;">${statusText}</span>
                            </span>
                            ${userLine}
                            <span class="list-item-meta">Last: ${formatTimeAgo(client.last_used)}</span>
                        </div>
                        <button class="btn btn-danger btn-sm" onclick="revokeClient('${client.client_id}')">${t('remove')}</button>
                    </div>
                `;
            }).join('');
        }

        function renderFolders(folders) {
            const container = document.getElementById('folderList');
            if (folders.length === 0) {
                container.innerHTML = '<div class="empty-state">No accessible folders</div>';
                return;
            }
            container.innerHTML = folders.map((folder) => `
                <div class="list-item">
                    <div class="list-item-info">
                        <span class="list-item-name">${escapeHtml(folder.name)}</span>
                        <span class="list-item-meta mono">${escapeHtml(folder.path)}</span>
                    </div>
                    <button class="btn-icon-toggle"
                            onclick="removeAccessibleFolder('${escapeHtml(folder.path)}')"
                            title="Remove folder">
                        ✕
                    </button>
                </div>
            `).join('');
        }

        async function removeAccessibleFolder(path) {
            const result = await fetchApi(`/api/filesystem/accessible-folders?path=${encodeURIComponent(path)}`, {
                method: 'DELETE'
            });
            if (result && result.success) {
                showToast('Folder removed', 'success');
                await fetchDashboard();
            } else {
                showToast(result?.detail || 'Failed to remove folder', 'error');
            }
        }


        function renderDevices(devices) {
            const container = document.getElementById('deviceList');
            if (devices.length === 0) {
                container.innerHTML = '<div class="empty-state">No devices available</div>';
                return;
            }
            container.innerHTML = devices.map(device => `
                <div class="list-item">
                    <div class="list-item-info">
                        <span class="list-item-name">${device.name || device.model || 'Unknown'}</span>
                        <span class="list-item-meta">${device.id}</span>
                    </div>
                    <label class="checkbox-wrapper">
                        <input type="checkbox"
                               ${device.enabled !== false ? 'checked' : ''}
                               onchange="toggleDeviceAccess('${device.id}', this.checked)">
                        <span class="checkmark"></span>
                    </label>
                </div>
            `).join('');
        }

        async function toggleDeviceAccess(deviceId, enabled) {
            await fetchApi('/api/devices/access', {
                method: 'PUT',
                body: JSON.stringify({ device_id: deviceId, enabled: enabled })
            });
            showToast(enabled ? 'Device enabled' : 'Device disabled', 'success');
        }

        // Actions
        async function checkUpdate() {
            showToast('Checking for updates...', 'success');
            const result = await fetchApi('/api/system/check-update', { method: 'POST' });
            if (result) {
                if (result.updated) {
                    showToast('Updated! Restart server to apply changes.', 'success');
                } else if (result.message) {
                    showToast(result.message, 'success');
                } else {
                    showToast('Already up to date', 'success');
                }
            } else {
                showToast('Update check failed', 'error');
            }
        }

        async function startTunnel() {
            showToast('Starting tunnel...', 'success');
            const result = await fetchApi('/api/system/tunnel/start', { method: 'POST' });
            await fetchDashboard();
            if (result && result.url) {
                showToast('Tunnel started: ' + result.url, 'success');
            } else if (result && result.error) {
                showToast('Failed: ' + result.error, 'error');
            }
        }

        async function stopTunnel() {
            showToast('Stopping tunnel...', 'success');
            await fetchApi('/api/system/tunnel/stop', { method: 'POST' });
            await fetchDashboard();
            showToast('Tunnel stopped', 'success');
        }

        async function revokeClient(clientId) {
            await fetchApi(`/api/pair/clients/${clientId}`, { method: 'DELETE' });
            await fetchDashboard();
            showToast('Device removed', 'success');
        }

        // Pairing Modal
        async function showPairingModal() {
            document.getElementById('pairingModal').classList.add('show');
            switchPairingTab('qr');
            await loadQrCode();
        }

        function closePairingModal(event) {
            if (event && event.target !== event.currentTarget) return;
            document.getElementById('pairingModal').classList.remove('show');
            if (qrTimerInterval) {
                clearInterval(qrTimerInterval);
                qrTimerInterval = null;
            }
            if (pairingPollInterval) {
                clearInterval(pairingPollInterval);
                pairingPollInterval = null;
            }
            currentPairingCode = null;
        }

        function switchPairingTab(tab) {
            document.querySelectorAll('.modal-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            if (tab === 'qr') {
                document.querySelector('.modal-tab:first-child').classList.add('active');
                document.getElementById('pairingTabQr').classList.add('active');
            } else {
                document.querySelector('.modal-tab:last-child').classList.add('active');
                document.getElementById('pairingTabCode').classList.add('active');
                updatePairingCodeDisplay();
            }
        }

        async function loadQrCode() {
            const qrContainer = document.getElementById('qrCode');
            qrContainer.innerHTML = '<div class="loading loading-lg"></div>';

            const data = await fetchApi('/api/pair/qr');
            if (data && data.qr_url) {
                qrExpiresAt = Date.now() + (data.expires_in_seconds || 300) * 1000;
                qrContainer.innerHTML = `<img src="/api/pair/qr-image" alt="QR Code" />`;
                currentPairingCode = data.pairing_code;
                updatePairingCodeDisplay();
                startQrTimer();
                pollForPairing(data.payload?.pair_token);
            } else {
                qrContainer.innerHTML = '<p style="color: var(--danger-color);">Failed to load QR code</p>';
            }
        }

        function updatePairingCodeDisplay() {
            const codeDigits = document.querySelectorAll('.code-digit-display');
            if (currentPairingCode && currentPairingCode.length === 6) {
                codeDigits.forEach((digit, i) => {
                    digit.textContent = currentPairingCode[i];
                });
            } else {
                codeDigits.forEach(digit => {
                    digit.textContent = '-';
                });
            }
        }

        async function refreshQr() {
            await loadQrCode();
        }

        function startQrTimer() {
            if (qrTimerInterval) clearInterval(qrTimerInterval);

            const qrTimerEl = document.getElementById('qrTimer');
            const codeTimerEl = document.getElementById('codeTimer');

            qrTimerInterval = setInterval(() => {
                const remaining = Math.max(0, qrExpiresAt - Date.now());
                const minutes = Math.floor(remaining / 60000);
                const seconds = Math.floor((remaining % 60000) / 1000);

                if (remaining <= 0) {
                    [qrTimerEl, codeTimerEl].forEach(el => {
                        if (el) {
                            el.textContent = 'Expired';
                            el.classList.add('expired');
                        }
                    });
                    clearInterval(qrTimerInterval);
                } else {
                    const timeText = `${minutes}:${seconds.toString().padStart(2, '0')}`;
                    [qrTimerEl, codeTimerEl].forEach(el => {
                        if (el) {
                            el.textContent = timeText;
                            el.classList.remove('expired');
                        }
                    });
                }
            }, 1000);
        }

        function pollForPairing(token) {
            if (!token) return;
            if (pairingPollInterval) clearInterval(pairingPollInterval);

            pairingPollInterval = setInterval(async () => {
                if (!document.getElementById('pairingModal').classList.contains('show')) {
                    clearInterval(pairingPollInterval);
                    return;
                }
                const status = await fetchApi(`/api/pair/token-status/${token}`);
                if (status && status.used) {
                    clearInterval(pairingPollInterval);
                    closePairingModal();
                    await fetchDashboard();
                    showToast('Device paired successfully!', 'success');
                }
            }, 2000);
        }

        // Folder Browser Modal
        let folderBrowserState = {
            currentPath: null,
            selectedPath: null,
            quickAccessPaths: [],
            drives: null,
            platform: null
        };

        async function showAddFolderModal() {
            document.getElementById('folderBrowserModal').classList.add('show');
            folderBrowserState.selectedPath = null;
            updateFolderSelection();
            await loadQuickAccess();
            await browseTo(null);  // Start at home directory
        }

        function closeFolderBrowserModal(event) {
            if (event && event.target !== event.currentTarget) return;
            document.getElementById('folderBrowserModal').classList.remove('show');
            folderBrowserState.selectedPath = null;
        }

        async function loadQuickAccess() {
            const result = await fetchApi('/api/filesystem/quick-access');
            if (result && result.paths) {
                folderBrowserState.quickAccessPaths = result.paths;
                renderQuickAccess();
            }
        }

        function renderQuickAccess() {
            const container = document.getElementById('quickAccessList');
            container.innerHTML = folderBrowserState.quickAccessPaths.map(p =>
                `<button class="quick-access-btn" onclick="browseTo('${escapeHtml(p.path)}')">${escapeHtml(p.name)}</button>`
            ).join('');
        }

        async function browseTo(path) {
            const listContainer = document.getElementById('folderBrowserList');
            listContainer.innerHTML = '<div class="folder-browser-loading"><div class="loading loading-lg"></div></div>';

            // Use unrestricted browse for adding new accessible folders
            const url = path
                ? `/api/filesystem/browse-unrestricted?path=${encodeURIComponent(path)}`
                : '/api/filesystem/browse-unrestricted';
            const result = await fetchApi(url);

            if (!result) {
                listContainer.innerHTML = '<div class="folder-browser-empty">Failed to load directory</div>';
                return;
            }

            folderBrowserState.currentPath = result.current_path;
            folderBrowserState.platform = result.platform;
            folderBrowserState.drives = result.drives;

            // Update path display
            document.getElementById('folderCurrentPath').textContent = result.current_path;

            // Update up button
            const upBtn = document.getElementById('folderUpBtn');
            upBtn.disabled = result.is_root;

            // Update drives dropdown (Windows only)
            const driveSelect = document.getElementById('folderDriveSelect');
            if (result.drives && result.drives.length > 0) {
                driveSelect.style.display = 'block';
                driveSelect.innerHTML = result.drives.map(d => {
                    const selected = result.current_path.startsWith(d) ? 'selected' : '';
                    return `<option value="${d}" ${selected}>${d}</option>`;
                }).join('');
            } else {
                driveSelect.style.display = 'none';
            }

            // Render folders
            if (result.folders.length === 0) {
                listContainer.innerHTML = '<div class="folder-browser-empty">No folders found</div>';
            } else {
                listContainer.innerHTML = result.folders.map(f => {
                    const disabledClass = f.is_accessible ? '' : 'disabled';
                    const selectedClass = f.path === folderBrowserState.selectedPath ? 'selected' : '';
                    return `
                        <div class="folder-item ${disabledClass} ${selectedClass}"
                             onclick="selectFolder('${escapeHtml(f.path)}', ${f.is_accessible})"
                             ondblclick="enterFolder('${escapeHtml(f.path)}', ${f.is_accessible})">
                            <span class="folder-item-icon">📁</span>
                            <span class="folder-item-name">${escapeHtml(f.name)}</span>
                            ${f.is_accessible ? '<span class="folder-item-arrow">›</span>' : '<span class="folder-item-arrow">🔒</span>'}
                        </div>
                    `;
                }).join('');
            }
        }

        function selectFolder(path, isAccessible) {
            if (!isAccessible) return;

            folderBrowserState.selectedPath = path;
            updateFolderSelection();

            // Update visual selection
            document.querySelectorAll('.folder-item').forEach(item => {
                item.classList.remove('selected');
            });
            event.currentTarget.classList.add('selected');
        }

        function enterFolder(path, isAccessible) {
            if (!isAccessible) return;
            browseTo(path);
        }

        function updateFolderSelection() {
            const selectionDiv = document.getElementById('folderSelection');
            const pathDisplay = document.getElementById('folderSelectedPath');
            const addBtn = document.getElementById('folderAddBtn');

            if (folderBrowserState.selectedPath) {
                selectionDiv.style.display = 'block';
                pathDisplay.textContent = folderBrowserState.selectedPath;
                addBtn.disabled = false;
            } else {
                selectionDiv.style.display = 'none';
                pathDisplay.textContent = '';
                addBtn.disabled = true;
            }
        }

        function navigateUp() {
            if (folderBrowserState.currentPath) {
                // Get parent path
                const pathParts = folderBrowserState.currentPath.split(/[\\/]/);
                pathParts.pop();
                const parentPath = pathParts.join(folderBrowserState.platform === 'windows' ? '\\\\' : '/') || '/';
                browseTo(parentPath);
            }
        }

        function navigateHome() {
            browseTo(null);
        }

        function onDriveChange(drive) {
            browseTo(drive + '\\\\');
        }

        async function confirmFolderSelection() {
            if (!folderBrowserState.selectedPath) {
                showToast('Please select a folder', 'error');
                return;
            }

            const result = await fetchApi('/api/filesystem/accessible-folders', {
                method: 'POST',
                body: JSON.stringify({ path: folderBrowserState.selectedPath })
            });

            if (result && result.success) {
                closeFolderBrowserModal();
                await fetchDashboard();
                showToast('Folder added successfully!', 'success');
            } else {
                showToast(result?.detail || 'Failed to add folder', 'error');
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Toast
        function showToast(message, type = 'success') {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = `toast ${type} show`;
            setTimeout(() => {
                toast.classList.remove('show');
            }, 3000);
        }

        // Utilities
        function formatTimeAgo(timestamp) {
            if (!timestamp) return 'Never';
            const diff = Date.now() - timestamp * 1000;
            const minutes = Math.floor(diff / 60000);
            if (minutes < 1) return 'Just now';
            if (minutes < 60) return `${minutes}m ago`;
            const hours = Math.floor(minutes / 60);
            if (hours < 24) return `${hours}h ago`;
            const days = Math.floor(hours / 24);
            return `${days}d ago`;
        }

        // Initialize
        async function init() {
            // Apply i18n translations
            applyTranslations();
            updateLangButtons();

            // Check auth status first
            await checkAuthStatus();

            // If authenticated (or no password), load dashboard
            if (authState.authenticated || !authState.passwordEnabled) {
                await fetchDashboard();
                await refreshServerLog(false);
                setInterval(fetchDashboard, 5000);
                serverLogPollInterval = setInterval(() => {
                    refreshServerLog(false);
                }, 3000);
            }
        }

        init();
    </script>
</body>
</html>
"""
