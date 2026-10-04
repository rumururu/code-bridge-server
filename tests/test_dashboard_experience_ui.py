"""Behavior checks for the Dashboard experience shell without starting services."""

import json
from pathlib import Path
import re
import subprocess

TEMPLATE = Path(__file__).resolve().parents[1] / "dashboard/templates/experience.html"
REVIEW = TEMPLATE.with_name("approval_review.js")


def script():
    html = TEMPLATE.read_text(encoding="utf-8")
    source = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    return source.replace("/* APPROVAL_REVIEW_MODULE */", REVIEW.read_text(encoding="utf-8")).replace(
        "/* BROWSER_HANDOFF_MODULE */", TEMPLATE.with_name("browser_handoff.js").read_text(encoding="utf-8")
    )


def run_node(source):
    process = subprocess.run(["node", "-"], input=source, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


def harness(body):
    source = script()
    source = source[:source.rfind("renderRoute();")]
    prefix = """
const nodes = {};
const documentListeners = {};
const document = {
  querySelector: selector => nodes[selector] ||= {
    innerHTML: '', textContent: '', value: '', disabled: false, isConnected: true,
    classList: {toggle() {}, add() {}, remove() {}},
    addEventListener() {}, querySelector() { return null; },
  },
  querySelectorAll: () => [],
  addEventListener(name, handler) { documentListeners[name] = handler; },
};
const listeners = {};
const window = {addEventListener: (name, callback) => {listeners[name] = callback}};
const location = {origin: 'http://localhost:8000', pathname: '/dashboard', search: ''};
const history = {pushState() {}, replaceState() {}};
const crypto = {randomUUID: () => 'attempt-123'};
const confirm = () => true;
"""
    return run_node(prefix + source + "\n" + body)


def test_script_parses_and_initial_partial_failure_is_not_empty():
    source = script()
    subprocess.run(["node", "--check", "-"], input=source, capture_output=True, text=True, check=True)
    result = harness("""
api = async path => path.includes('system') ? {server: {status:'online'}, llm: {connected:true}} :
  {running:[], recent:[], action_count:null, next_schedules:[], section_errors:['running','recent','action_count','next_schedules']};
(async () => {
  await loadOverview(state.generation);
  console.log(JSON.stringify({html:nodes['#content'].innerHTML, time:state.overviewTime}));
})();
""")
    assert "조회 실패" in result["html"]
    assert "항목 없음" not in result["html"]
    assert "예약 없음" not in result["html"]
    assert result["time"] is None


def test_partial_failure_keeps_previous_data_and_server_failure_is_separate():
    result = harness("""
state.overview = {running:[{id:'old',title:'이전 작업'}],recent:[],action_count:4,next_schedules:[]};
state.overviewTime = 1234;
api = async path => {
  if (path.includes('system')) throw Error('server unavailable');
  return {running:[],recent:[],action_count:null,next_schedules:[],section_errors:['running','action_count']};
};
(async () => {
  await loadOverview(state.generation);
  console.log(JSON.stringify({html:nodes['#content'].innerHTML,count:state.overview.action_count,
    running:state.overview.running[0].id,time:state.overviewTime}));
})();
""")
    assert result["count"] == 4
    assert result["running"] == "old"
    assert result["time"] == 1234
    assert "상태 조회 실패" in result["html"]
    assert "이전 작업" in result["html"]


def test_project_attempt_reconciles_by_exact_metadata_and_keeps_project_drafts():
    result = harness("""
state.path='/projects'; state.project='alpha';
draftFor('alpha').title='초안 A'; draftFor('beta').title='초안 B';
$('#taskTitle').value='초안 A'; $('#taskDescription').value='';
state.taskAttempts.alpha={id:'attempt-123',status:'uncertain',payload:{title:'초안 A',description:''}};
reconcileAttempt('alpha',[{id:'other',title:'같은 제목',metadata:{dashboard_request_id:'other'}}]);
const absent=state.taskAttempts.alpha.status;
reconcileAttempt('alpha',[{id:'created',title:'다른 제목',metadata:{dashboard_request_id:'attempt-123'}}]);
console.log(JSON.stringify({absent,confirmed:state.taskAttempts.alpha.status,
  taskId:state.taskAttempts.alpha.taskId,alpha:draftFor('alpha').title,beta:draftFor('beta').title}));
""")
    assert result == {"absent": "absent", "confirmed": "confirmed", "taskId": "created", "alpha": "", "beta": "초안 B"}


def test_bounded_or_failed_reconciliation_blocks_retry():
    result = harness("""
state.taskAttempts.alpha={id:'attempt-123',status:'uncertain'};
reconcileAttempt('alpha',Array.from({length:200},(_,index)=>({id:String(index),metadata:{}})));
const bounded=taskFormState('alpha');
state.taskAttempts.alpha.status='query_failed';
const failed=taskFormState('alpha');
console.log(JSON.stringify({bounded,failed}));
""")
    assert result["bounded"]["disabled"] is True
    assert result["failed"]["disabled"] is True


def test_late_post_does_not_render_other_project():
    result = harness("""
state.path='/projects'; state.project='alpha'; state.generation=7;
let resolvePost;
api=()=>new Promise(resolve=>{resolvePost=resolve});
let rendered=0;
renderProjects=()=>{rendered++};
loadProjectItems=async()=>{};
const attempt={id:'attempt-123',status:'pending',payload:{title:'A',project_name:'alpha',metadata:{dashboard_request_id:'attempt-123'}}};
state.taskAttempts.alpha=attempt;
(async()=>{
  const pending=postAttempt('alpha',attempt,7);
  state.project='beta'; state.generation=8;
  resolvePost({task:{id:'created'}});
  await pending;
  console.log(JSON.stringify({rendered,alpha:attempt.status,beta:draftFor('beta').title}));
})();
""")
    assert result == {"rendered": 0, "alpha": "confirmed", "beta": ""}


def test_navigation_rejects_other_origin_and_other_window():
    result = harness("""
const child={};
nodes['#agentsFrame iframe']={contentWindow:child};
const seen=[];
navigate=path=>seen.push(path);
listeners.message({origin:'http://other',source:child,data:{type:'dashboard:navigate',path:'/inbox'}});
listeners.message({origin:location.origin,source:{},data:{type:'dashboard:navigate',path:'/inbox'}});
listeners.message({origin:location.origin,source:child,data:{type:'dashboard:navigate',path:'/inbox'}});
console.log(JSON.stringify(seen));
""")
    assert result == ["/inbox"]


def test_automation_return_keeps_child_view_and_management_keeps_section():
    result = harness("""
const calls=[];
show=()=>{};
frame=()=>({});
sendFrameSection=(_frame,section)=>calls.push(section);
location.pathname='/agents';
renderRoute();
state.managementSection='diagnostics';
location.pathname='/settings';
renderRoute();
console.log(JSON.stringify({calls,section:state.managementSection}));
""")
    assert result == {"calls": ["diagnostics"], "section": "diagnostics"}


def test_run_status_uses_korean_for_known_values_and_preserves_unknown():
    result = harness("""
console.log(JSON.stringify({known:runStatus('completed'),unknown:runStatus('custom_state'),
  link:runLink({id:'run_123456789012345',title:'자료 정리',status:'completed'})}));
""")
    assert result["known"] == "완료"
    assert result["unknown"] == "custom_state"
    assert "자료 정리" in result["link"]
    assert "run_1234567" in result["link"]


def test_uncertain_post_cannot_send_second_request_after_absent_recheck():
    result = harness("""
state.path='/projects'; state.project='alpha'; state.generation=3;
state.taskAttempts.alpha={id:'attempt-123',status:'uncertain',payload:{title:'A'}};
renderProjects=()=>{}; loadProjectItems=async()=>{};
let posts=0;
api=async path=>{if(path.endsWith('/tasks')) posts++; return {tasks:[]}};
(async()=>{
  await recheckAttempt('alpha');
  await submitTask();
  console.log(JSON.stringify({status:state.taskAttempts.alpha.status,
    disabled:taskFormState('alpha').disabled,posts}));
})();
""")
    assert result == {"status": "absent", "disabled": True, "posts": 0}


def test_run_partial_failures_display_error_states():
    result = harness("""
api=async path=>{
  if(path.endsWith('/summary')) return {run:{id:'run-1',title:'자료 정리',status:'completed'},
    artifacts:[],section_errors:['artifacts'],verification:{status:'unknown'}};
  if(path.endsWith('/events')) return {events:[]};
  throw Error('checkpoint unavailable');
};
(async()=>{
  await loadRun('run-1',state.generation);
  console.log(JSON.stringify({html:nodes['#content'].innerHTML,title:nodes['#title'].textContent,
    status:nodes['#status'].innerHTML}));
})();
""")
    assert result["title"] == "자료 정리"
    assert "결과물 조회 실패" in result["html"]
    assert "체크포인트 조회 실패" in result["html"]
    assert "실행 일부 정보 조회 실패" in result["status"]


def test_late_artifact_response_cannot_replace_new_selection():
    result = harness("""
state.run={run:{id:'run-1'}};
const pending={};
api=path=>new Promise(resolve=>{pending[path.split('/').at(-2)]=resolve});
(async()=>{
  const first=showArtifact('artifact-a');
  const second=showArtifact('artifact-b');
  pending['artifact-b']({readable:true,content:'B 내용'});
  await second;
  pending['artifact-a']({readable:true,content:'A 내용'});
  await first;
  console.log(JSON.stringify(nodes['#artifactContent'].textContent));
})();
""")
    assert result == "B 내용"


def test_checkpoint_draft_survives_rerender():
    result = harness("""
state.checkpointDrafts['run-1:step-1']='작성 중인 답변';
const host={innerHTML:''};
renderCheckpoint(host,{checkpoint:{prompt:'확인'},step:{id:'step-1'}},
  {id:'step-1',run_id:'run-1',task_id:'task-1',title:'응답 요청'});
console.log(JSON.stringify(host.innerHTML));
""")
    assert "작성 중인 답변" in result


HANDOFF_HOST = """
const handoffNodes = {};
const host = {innerHTML:'', querySelectorAll:()=>[], querySelector: selector =>
  handoffNodes[selector] ||= {textContent:'',hidden:false,disabled:false,value:'',srcObject:null}};
const handoffData = {run:{id:'run-browser',status:'running'},task:{id:'task-browser'},
  checkpoint:{reason:'login_required',browser_session_id:'session-browser',prompt:'internal prompt'}};
const handoffItem = {run_id:'run-browser',task_id:'task-browser'};
"""


def test_browser_checkpoint_uses_handoff_controls_not_json_or_generic_response():
    result = harness(HANDOFF_HOST + """
renderCheckpoint(host,handoffData,handoffItem);
console.log(JSON.stringify({html:host.innerHTML,guide:handoffNodes['[data-handoff-guide]'].textContent,
  disabled:handoffNodes['[data-handoff-done]'].disabled}));
""")
    assert 'data-handoff-video' in result['html']
    assert 'data-handoff-done disabled' in result['html']
    assert 'checkpointAnswer' not in result['html']
    assert '<pre>' not in result['html']
    assert 'internal prompt' not in result['html']
    assert '같은 단계' in result['guide']


def test_terminal_browser_request_does_not_offer_login_or_resume():
    result = harness(HANDOFF_HOST + """
handoffData.run.status='failed';
renderCheckpoint(host,handoffData,handoffItem);
console.log(JSON.stringify({guide:handoffNodes['[data-handoff-guide]'].textContent,
  openHidden:handoffNodes['[data-handoff-open]'].hidden,
  doneHidden:handoffNodes['[data-handoff-done]'].hidden}));
""")
    assert '이미 종료된 실행' in result['guide']
    assert result['openHidden'] and result['doneHidden']


def test_browser_request_changed_during_open_is_not_streamed():
    result = harness(HANDOFF_HOST + """
api=async()=>({run:{id:'different-run'},browser_session:{id:'different-session'}});
renderCheckpoint(host,handoffData,handoffItem);
(async()=>{
  await handoffNodes['[data-handoff-open]'].onclick();
  console.log(JSON.stringify({message:handoffNodes['[data-handoff-status]'].textContent,
    doneDisabled:handoffNodes['[data-handoff-done]'].disabled}));
})();
""")
    assert '요청이 변경' in result['message']
    assert result['doneDisabled']


def test_leaving_or_closing_browser_before_fetch_finishes_cannot_start_stream():
    result = harness(HANDOFF_HOST + """
let resolve;
api=()=>new Promise(r=>{resolve=r});
renderCheckpoint(host,handoffData,handoffItem);
(async()=>{
  const pending=handoffNodes['[data-handoff-open]'].onclick();
  handoffNodes['[data-handoff-close]'].onclick();
  resolve({run:{id:'run-browser'},browser_session:{id:'session-browser'}});
  await pending;
  console.log(JSON.stringify({message:handoffNodes['[data-handoff-status]'].textContent,
    hidden:handoffNodes['[data-handoff-player]'].hidden}));
})();
""")
    assert result['hidden']
    assert '화면을 닫았습니다' in result['message']


def test_disconnected_input_is_not_cleared_without_delivery():
    result = harness(HANDOFF_HOST + """
renderCheckpoint(host,handoffData,handoffItem);
handoffNodes['[data-handoff-text]'].value='dummy fixture input';
handoffNodes['[data-handoff-send]'].onclick();
console.log(JSON.stringify({value:handoffNodes['[data-handoff-text]'].value}));
""")
    assert result['value'] == 'dummy fixture input'


def test_old_offer_rejection_cannot_close_reconnected_peer():
    result = harness(HANDOFF_HOST + """
location.href='http://localhost:8000/inbox'; location.protocol='http:';
let rejectOffer;
const peers=[],sockets=[];
global.RTCPeerConnection=class {
  constructor(){this.closed=false;peers.push(this);}
  addTransceiver(){}
  createDataChannel(){return {readyState:'connecting',close(){}};}
  createOffer(){return new Promise((resolve,reject)=>{rejectOffer=reject});}
  close(){this.closed=true;}
};
global.WebSocket=class {
  static OPEN=1;
  constructor(){this.readyState=1;sockets.push(this);}
  send(){} close(){}
};
api=async()=>({run:{id:'run-browser'},browser_session:{id:'session-browser'}});
renderCheckpoint(host,handoffData,handoffItem);
(async()=>{
  await handoffNodes['[data-handoff-open]'].onclick();
  const oldSignal=sockets[0].onmessage({data:JSON.stringify({type:'browser_rtc.ready',browser_session:{id:'session-browser'}})});
  handoffNodes['[data-handoff-close]'].onclick();
  await handoffNodes['[data-handoff-open]'].onclick();
  rejectOffer(Error('old peer closed'));
  await oldSignal;
  const kept=!peers[1].closed;
  closeBrowserHandoff();
  console.log(JSON.stringify({kept}));
})();
""")
    assert result['kept']


def test_project_switch_snapshots_live_form_before_replacing_dom():
    result = harness("""
state.path='/projects'; state.project='QA Alpha';
state.projects=[{name:'QA Alpha'},{name:'QA Beta'}];
nodes['#taskTitle']={value:'Alpha 초안'};
nodes['#taskDescription']={value:'Alpha 내용'};
saveVisibleDraft(state.project);
state.project='QA Beta';
renderProjects(false);
const beta=nodes['#content'].innerHTML;
$('#taskTitle').value='Beta 초안';
saveVisibleDraft(state.project);
state.project='QA Alpha';
renderProjects(false);
console.log(JSON.stringify({alpha:nodes['#content'].innerHTML.includes('Alpha 초안'),
  betaBlank:beta.includes('value=""'), alphaDraft:draftFor('QA Alpha'), betaDraft:draftFor('QA Beta')}));
""")
    assert result["alpha"] is True
    assert result["betaBlank"] is True
    assert result["alphaDraft"] == {"title": "Alpha 초안", "description": "Alpha 내용"}
    assert result["betaDraft"]["title"] == "Beta 초안"


def test_project_read_response_updates_lists_without_recreating_form():
    result = harness("""
state.path='/projects'; state.project='QA Alpha'; state.generation=2;
$('#taskTitle').value='첫 제목';
$('#taskDescription').value='첫 내용';
let renders=0;
renderProjects=()=>{renders++};
api=async path=>path.includes('/history') ? {runs:[],total_count:0,next_cursor:null} : {tasks:[]};
(async()=>{
  const titleNode=nodes['#taskTitle'];
  await loadProjectItems('QA Alpha',2);
  console.log(JSON.stringify({renders,sameNode:nodes['#taskTitle']===titleNode,title:titleNode.value}));
})();
""")
    assert result == {"renders": 0, "sameNode": True, "title": "첫 제목"}


def test_confirmed_post_does_not_clear_newer_edit():
    result = harness("""
state.path='/projects'; state.project='QA Alpha';
state.projectDrafts['QA Alpha']={title:'새 초안',description:'새 내용'};
$('#taskTitle').value='새 초안'; $('#taskDescription').value='새 내용';
clearSubmittedDraft('QA Alpha',{payload:{title:'이전 요청',description:'이전 내용'}});
console.log(JSON.stringify({draft:draftFor('QA Alpha'),title:nodes['#taskTitle'].value}));
""")
    assert result == {"draft": {"title": "새 초안", "description": "새 내용"}, "title": "새 초안"}


def test_overview_and_server_permission_failures_are_distinct():
    result = harness("""
api=async path=>{const error=new Error('forbidden');error.status=path.includes('system')?401:403;throw error};
(async()=>{
  await loadOverview(state.generation);
  console.log(JSON.stringify({content:nodes['#content'].innerHTML,status:nodes['#status'].innerHTML}));
})();
""")
    assert "접근 권한이 없습니다" in result["content"]
    assert "개요 조회 실패: 접근 권한이 없습니다" in result["status"]
    assert "서버 상태 조회 실패: 접근 권한이 없습니다" in result["status"]


def test_approval_resume_success_refreshes_without_checkpoint_reference():
    result = harness("""
state.path='/inbox'; state.selected='recovery';
state.checkpointDrafts['unrelated']='keep';
let refreshes=0;
api=async()=>({resume_status:'pending'});
loadInbox=async()=>{refreshes++};
const button={dataset:{resume:'approval-1'},disabled:false};
const event={target:{id:'',closest:selector=>selector==='[data-resume]'?button:null}};
(async()=>{
  await documentListeners.click(event);
  console.log(JSON.stringify({refreshes,status:nodes['#status'].innerHTML,
    error:nodes['#actionResult']?.textContent||'',draft:state.checkpointDrafts.unrelated}));
})();
""")
    assert result['refreshes'] == 1
    assert 'pending' in result['status']
    assert result['error'] == ''
    assert result['draft'] == 'keep'


def test_reload_restores_uncertain_task_and_blocks_duplicate_post():
    result = harness("""
let saved=null;
const sessionStorage={getItem:()=>saved,setItem:(_key,value)=>{saved=value}};
state.path='/projects'; state.project='alpha';
state.taskAttempts.alpha={id:'lost-request',status:'pending',payload:{title:'same'}};
persistTaskAttempts();
state.taskAttempts=restoreTaskAttempts();
state.projectDrafts.alpha={title:'same',description:''};
let posts=0;
api=async()=>{posts++;return {}};
(async()=>{
  await submitTask();
  console.log(JSON.stringify({posts,status:state.taskAttempts.alpha.status}));
})();
""")
    assert result == {'posts': 0, 'status': 'uncertain'}


def test_connection_summary_does_not_treat_registered_clients_as_online_devices():
    result = harness("""
console.log(JSON.stringify({html:connectionOverview({status:'fulfilled',value:{
  server:{server_name:'<img src=x onerror=alert(1)>',api_listening:false},
  pairing:{active_clients:2},devices:{total:0},tunnel:{running:false}
}})}));
""")
    assert '페어링 등록</dt><dd>2대' in result['html']
    assert '개발 기기</dt><dd>0대' in result['html']
    assert 'API 연결 확인 필요' in result['html']
    assert '터널 중지됨' in result['html']
    assert '<img src=x' not in result['html']
    assert '온라인' not in result['html']


def test_connection_summary_missing_status_is_unknown_not_zero_or_connected():
    result = harness("""
console.log(JSON.stringify({empty:connectionOverview({status:'fulfilled',value:{}}),
  failed:connectionOverview({status:'rejected',reason:Error('unavailable')})}));
""")
    assert 'API 상태 미확인' in result['empty']
    assert '터널 상태 미확인' in result['empty']
    assert '0대' not in result['empty']
    assert '서버 상태를 불러오지 못했습니다' in result['failed']
    assert '연결됨' not in result['failed']


def test_browser_resume_refresh_waits_for_repeated_login_checkpoint():
    result = harness("""
state.path = '/inbox'; state.selected = 'step';
let calls = 0, refreshed = 0, notice;
global.setTimeout = callback => callback();
api = async () => {
  calls++;
  return {run:{status:calls === 2 ? 'running' : 'waiting_for_user'},
    checkpoint:{created_at:calls < 3 ? 'old' : 'new'}};
};
loadInbox = async () => {refreshed++};
status = text => {notice = text};
(async () => {
  await refreshBrowserResume({id:'step',run_id:'run'}, 'old');
  console.log(JSON.stringify({calls,refreshed,notice}));
})();
""")
    assert result['calls'] == 3
    assert result['refreshed'] == 1
    assert '아직 확인이 필요합니다' in result['notice']


def test_browser_resume_refresh_does_not_replace_another_selected_request():
    result = harness("""
state.path = '/inbox'; state.selected = 'step';
let refreshed = 0;
global.setTimeout = callback => callback();
api = async () => {state.selected = 'another'; return {run:{status:'waiting_for_user'}}};
loadInbox = async () => {refreshed++};
(async () => {
  await refreshBrowserResume({id:'step',run_id:'run'});
  console.log(JSON.stringify({refreshed}));
})();
""")
    assert result['refreshed'] == 0
