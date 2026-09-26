const ApprovalReview = (() => {
  function esc(s) {
    return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  function render(approval, lang) {
    const isKo = lang === 'ko';
    const t = (ko, en) => isKo ? ko : en;
    const d = approval.details || {};
    const op = approval.operation;

    let h = '<div class="approval-review">';

    if (op === 'feedback.reply.send') {
      h += `<h3>${t('피드백 답장을 보내려 합니다', 'The agent wants to send a feedback reply.')}</h3>`;
      h += '<dl>';
      const fields = ['recipient', 'subject', 'agent_name', 'project_name', 'summary'];
      for (const f of fields) {
        const val = d[f];
        const label = f === 'recipient' ? t('수신자', 'Recipient') :
                      f === 'subject' ? t('제목', 'Subject') :
                      f === 'agent_name' ? t('에이전트', 'Agent') :
                      f === 'project_name' ? t('프로젝트', 'Project') :
                      t('요약', 'Summary');
        if (typeof val !== 'string' || val.trim() === '') {
          h += `<dt>${label}</dt><dd style="white-space:pre-wrap;overflow-wrap:anywhere;">${t('요청에 제공되지 않았습니다', 'Not provided in this request')}</dd>`;
        } else {
          h += `<dt>${label}</dt><dd style="white-space:pre-wrap;overflow-wrap:anywhere;">${esc(val)}</dd>`;
        }
      }

      const orig = d.original_message;
      if (typeof orig !== 'string' || orig.trim() === '') {
        h += `<dt>${t('원본 문의', 'Original Message')}</dt><dd style="white-space:pre-wrap;overflow-wrap:anywhere;">${t('요청에 제공되지 않았습니다', 'Not provided in this request')}</dd>`;
      } else {
        h += `<dt>${t('원본 문의', 'Original Message')}</dt><dd style="white-space:pre-wrap;overflow-wrap:anywhere;">${esc(orig)}</dd>`;
      }

      const reply = d.proposed_reply;
      if (typeof reply !== 'string' || reply.trim() === '') {
        h += `<dt>${t('보낼 답장', 'Reply to send')}</dt><dd style="white-space:pre-wrap;overflow-wrap:anywhere;">${t('요청에 제공되지 않았습니다', 'Not provided in this request')}</dd>`;
      } else {
        h += `<dt>${t('보낼 답장', 'Reply to send')}</dt><dd style="white-space:pre-wrap;overflow-wrap:anywhere;">${esc(reply)}</dd>`;
      }

      h += '</dl>';

      h += `<p class="approval-risk">${t('외부 수신자에게 답장이 발송됩니다. 발송 후에는 여기서 되돌릴 수 없습니다.', 'This reply will be sent to an external recipient. Sending cannot be undone here.')}</p>`;
      h += `<h3>${t('승인 후', 'After approval')}</h3>`;
      h += `<p>${t('승인은 발송 권한을 기록합니다. 외부 피드백 에이전트가 다음 처리 시 답장을 발송하며, 승인 자체가 발송 완료를 뜻하지는 않습니다.', 'Approval records permission. The external feedback agent sends the reply on its next processing cycle; this is not a delivery confirmation.')}</p>`;

      if (typeof d.recipient !== 'string' || d.recipient.trim() === '' || typeof d.proposed_reply !== 'string' || d.proposed_reply.trim() === '') {
         h += `<p>${t('수신자 또는 답장이 누락되었습니다. 완전하게 재제출할 때까지 승인하지 마십시오.', 'Missing recipient or reply. Do not approve until resubmitted complete.')}</p>`;
      }
    } else {
      const target = d.display && d.display.target;
      const reason = d.reason;
      const summary = d.summary;
      let input = d.input;

      h += '<dl>';
      if (typeof target === 'string') h += `<dt>${t('대상', 'Target')}</dt><dd style="white-space:pre-wrap;overflow-wrap:anywhere;">${esc(target)}</dd>`;
      if (typeof reason === 'string') h += `<dt>${t('이유', 'Reason')}</dt><dd style="white-space:pre-wrap;overflow-wrap:anywhere;">${esc(reason)}</dd>`;
      if (typeof summary === 'string') h += `<dt>${t('요약', 'Summary')}</dt><dd style="white-space:pre-wrap;overflow-wrap:anywhere;">${esc(summary)}</dd>`;

      if (input !== undefined && input !== null) {
        let inputStr;
        try { inputStr = typeof input === 'string' ? input : JSON.stringify(input); } catch { inputStr = String(input); }
        h += `<dt>${t('입력', 'Input')}</dt><dd style="white-space:pre-wrap;overflow-wrap:anywhere;">${esc(inputStr)}</dd>`;
      } else {
        input = d.tool_input ?? d.arguments ?? d.args ?? d.payload;
        if (input !== undefined && input !== null) {
          let inputStr;
          try { inputStr = typeof input === 'string' ? input : JSON.stringify(input); } catch { inputStr = String(input); }
          h += `<dt>${t('입력', 'Input')}</dt><dd style="white-space:pre-wrap;overflow-wrap:anywhere;">${esc(inputStr)}</dd>`;
        } else {
          h += `<dt>${t('입력', 'Input')}</dt><dd style="white-space:pre-wrap;overflow-wrap:anywhere;">${t('요청에 제공되지 않았습니다', 'Not provided')}</dd>`;
        }
      }
      h += '</dl>';

      h += `<p>${t('승인하면 요청한 작업이 허용됩니다. 대기 중인 실행이 재개될 수 있으며, 결과는 별도로 확인해야 합니다.', 'After approval, the requested operation is allowed. Pending execution may resume, and results should be checked separately.')}</p>`;
    }

    h += '</div>';
    return h;
  }

  function canReview(approval) {
    if (approval.operation === 'feedback.reply.send') {
      const d = approval.details || {};
      return typeof d.recipient === 'string' && d.recipient.trim() !== '' &&
             typeof d.proposed_reply === 'string' && d.proposed_reply.trim() !== '';
    }
    return true;
  }

  function scope(approval) {
    const d = approval.details || {};
    const pn = d.project_name;
    if (typeof pn === 'string' && pn.trim() !== '' && pn.trim() !== '__global__') return 'project:' + pn.trim();
    const wi = d.workspace_id;
    if (typeof wi === 'string' && wi.trim() !== '') return 'workspace:' + wi.trim();
    const ai = d.agent_id;
    if (typeof ai === 'string' && ai.trim() !== '') return 'agent:' + ai.trim();
    const rid = approval.run_id;
    if (typeof rid === 'string' && rid.trim() !== '') return 'run:' + rid.trim();
    return 'global';
  }

  return { render, canReview, scope };
})();
