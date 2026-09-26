"""Behavior gates for the shipped approval renderer and decision controls."""
import json
from pathlib import Path
import subprocess
import sys

SERVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(Path(__file__).parent))
from dashboard_js import COMMON_STUBS, js_function, run_js


def test_full_content_and_scope_contract():
    subprocess.run([
        'node', str(Path(__file__).with_name('check_approval_review.cjs')),
        str(SERVER / 'dashboard/templates/approval_review.js'),
    ], check=True, capture_output=True, text=True)


def test_shipped_page_embeds_review_module():
    from dashboard.dashboard_page import render_agents_html
    html = render_agents_html()
    assert 'const ApprovalReview =' in html
    assert '/* APPROVAL_REVIEW_MODULE */' not in html
    assert 'ApprovalReview.render(approval, currentLang)' in html


def test_decisions_require_complete_content_and_standing_rule_confirmation():
    module = (SERVER / 'dashboard/templates/approval_review.js').read_text()
    result = run_js('\n'.join([
        module, js_function('decideApproval'),
        """
        const calls = [], prompts = [];
        const approvalDecisionsInFlight = new Set();
        const currentLang = 'ko';
        const ruleScopePhrase = scope => scope;
        const approvals = [{id:'apr',operation:'feedback.reply.send',details:{recipient:'a@b.c',project_name:'demo'}}];
        let answer = false;
        const confirm = value => {prompts.push(value); return answer;};
        const renderApprovals = () => {};
        const toast = () => {};
        const reloadAll = async () => {};
        const api = async (path, options) => {calls.push(JSON.parse(options.body)); return {};};
        (async () => {
          await decideApproval('apr','approve_once');
          await decideApproval('apr','approve_rule');
          if (calls.length) throw Error('missing body must not be approved');
          approvals[0].details.proposed_reply = 'reply';
          await decideApproval('apr','approve_rule');
          if (calls.length) throw Error('cancelled confirmation must not write');
          if (!prompts[0].includes('project:demo') || !prompts[0].includes('만료가 없습니다')) throw Error('scope and expiry must be visible');
          answer = true;
          await decideApproval('apr','approve_rule');
          approvals[0].details = {};
          await decideApproval('apr','deny');
          console.log(JSON.stringify(calls));
        })();
        """,
    ]))
    calls = json.loads(result)
    assert calls[0]['rule_scope'] == 'project:demo'
    assert calls[0]['decision'] == 'approve_rule'
    assert calls[1]['decision'] == 'deny'
    assert len(calls) == 2


def test_card_has_no_truncated_target_or_direct_approval_outside_review():
    module = (SERVER / 'dashboard/templates/approval_review.js').read_text()
    html = run_js('\n'.join([
        COMMON_STUBS, module,
        """
        const currentLang='ko';
        const ruleScopePhrase = scope => scope;
        const approvalDecisionsInFlight = new Set();
        const host={innerHTML:'',querySelectorAll:()=>[]};
        const document={getElementById:()=>host};
        const approvals=[{id:'apr',operation:'feedback.reply.send',details:{
          recipient:'reader@example.com', proposed_reply:'full reply',
          display:{action:'send_feedback_reply',target:'long target last word'}
        }}];
        """,
        js_function('approvalActionSentence'), js_function('renderApprovals'),
        "renderApprovals(); console.log(host.innerHTML);",
    ]))
    assert 'long target last word' in html
    assert 'text-overflow:ellipsis' not in html
    assert 'full reply' in html
    assert html.index('data-approval-id=') < html.index('data-decision="approve_once"')
    assert '내용과 승인 범위 확인' in html
