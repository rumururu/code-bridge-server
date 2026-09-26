const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');
const mod = vm.runInNewContext(fs.readFileSync(process.argv[2], 'utf8') + '\n;ApprovalReview');
const body = 'First line\n' + 'long content '.repeat(400) + '\nLAST-LINE';
const request = {id:'apr_test', operation:'feedback.reply.send', risk_level:'high', details: {
  recipient:'reader@example.com', subject:'A <script>bad()</script> title',
  original_message:'original question', proposed_reply:body, summary:'summary here',
  agent_name:'Feedback agent', project_name:'demo'
}};
const html = mod.render(request, 'ko');
assert.ok(html.includes('되돌릴 수 없습니다'), 'explain sending risk');
for (const content of ['reader@example.com','original question','LAST-LINE','summary here','Feedback agent','demo','보낼 답장','원본 문의','승인 후']) assert.ok(html.includes(content), content);
assert.ok(html.includes(body), 'complete multiline body');
assert.ok(!html.includes('<script>bad()'), 'escape untrusted content');
assert.ok(html.includes('&lt;script&gt;'), 'show escaped text, not delete it');
assert.equal(mod.canReview(request), true);
assert.equal(mod.canReview({...request, details:{recipient:'a@b.c'}}), false);
assert.equal(mod.canReview({...request, details:{recipient:23, proposed_reply:'x'}}), false);
const missing = mod.render({...request,details:{}}, 'ko');
assert.ok(missing.includes('제공되지'), 'missing content is explicit');
assert.match(mod.render(request,'en'), /<dt>Reply to send<\/dt>/, 'reply heading must label the actual body');
assert.ok(mod.render(request,'ko').includes('승인은 발송 권한을 기록합니다'), 'accurate permission wording, not existing authority');
const blank = mod.render({...request,details:{recipient:'a@b.c', original_message:' ',proposed_reply:' '}},'ko');
assert.ok((blank.match(/요청에 제공되지 않았습니다/g) || []).length >= 4, 'blank strings count as missing');
assert.equal(mod.scope(request), 'project:demo');
assert.equal(mod.scope({details:{project_name:'__global__',agent_id:'agent_1'}}), 'agent:agent_1');
assert.equal(mod.scope({run_id:'run_1',details:{workspace_id:'w'}}), 'workspace:w');
assert.equal(mod.scope({run_id:'run_1',details:{}}), 'run:run_1');
assert.equal(mod.scope({details:{}}), 'global');
const generic=mod.render({operation:'process.terminal',details:{input:{command:body},reason:'needed reason'}},'en');
assert.ok(generic.includes('LAST-LINE') && generic.includes('needed reason'), 'generic complete input');
console.log('PASS approval review web contract');
