// Same-origin shell integration. Existing forms and their save gates stay intact.
(() => {
    const management = document.body.dataset.embedded === 'management';
    // The shell is Korean; keep the embedded controls in the same language.
    if (typeof setLanguage === 'function') setLanguage('ko');
    if (management) {
        const groups = {
            connection: ['linkSpine', 'pairingBanner', 'dashboardUrl', 'clientList'],
            environment: ['llmList', 'deviceList'],
            access: ['ipLoginCard', 'folderList'],
            diagnostics: ['serverVersion', 'serverLogContent'],
        };
        Object.entries(groups).forEach(([group, ids]) => ids.forEach(id => {
            const node = document.getElementById(id);
            if (node) (node.closest('.card') || node).dataset.management = group;
        }));
        const overview = document.getElementById('agentStatStrip');
        if (overview) overview.closest('.card').id = 'agentOverviewCard';
    }
    window.dashboardNavigate = async ({agentId, section} = {}) => {
        if (management) {
            document.querySelectorAll('[data-management]').forEach(node => {
                node.toggleAttribute('data-management-hidden', !!section && node.dataset.management !== section);
            });
            window.scrollTo(0, 0);
            return;
        }
        if (window.dashboardReady) await window.dashboardReady;
        if (agentId) {
            goManage();
            await selectAgent(agentId);
        }
        if (section === 'create' && currentView !== 'create') goCreate();
        else if (['setup', 'approvals', 'cli-sweep'].includes(section)) goSetup(section);
        else if (section === 'runs' || section === 'schedule') {
            goManage();
            showDetail(section);
        }
    };
    document.addEventListener('click', event => {
        const link = event.target.closest('a[href]');
        if (!link || link.target === '_blank' || event.ctrlKey || event.metaKey) return;
        const url = new URL(link.href, location.href);
        if (url.origin !== location.origin || !['/dashboard', '/agents', '/settings', '/inbox', '/projects'].includes(url.pathname)) return;
        if (window.parent === window) return;
        event.preventDefault();
        window.parent.postMessage({type: 'dashboard:navigate', path: url.pathname + url.search}, location.origin);
    });
})();
