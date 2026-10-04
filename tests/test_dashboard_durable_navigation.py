"""Exercise lost responses and Canvas refresh while a draft is being edited."""
import json
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))
from tests.dashboard_js import js_function, run_js


def test_schedule_response_loss_reuses_exact_task_and_schedule():
    source = js_function('saveScheduleAttempt')
    result = json.loads(run_js('''
const scheduleAttempts = new Map();
const currentLang = 'ko';
const crypto = {randomUUID: () => 'attempt-one'};
let task = null, schedules = [], taskPosts = 0, schedulePosts = 0;
async function api(path, options) {
    if (path === '/tasks' && options) {
        taskPosts++;
        task = {id:'task_one', metadata:JSON.parse(options.body).metadata};
        return {error:'Lost task response'};
    }
    if (path === '/tasks?limit=200') return {tasks:[{id:'wrong', title:'same'}, task]};
    if (options) {
        schedulePosts++;
        schedules = [{id:'schedule_one'}];
        return {error:'Lost schedule response'};
    }
    return {schedules};
}
''' + source + '''
(async () => {
    const args = ['form', {id:'agent_one'}, 'same', {kind:'interval',seconds:60}];
    const first = await saveScheduleAttempt(...args);
    const second = await saveScheduleAttempt(...args);
    const third = await saveScheduleAttempt(...args);
    console.log(JSON.stringify({first,second,third,taskPosts,schedulePosts}));
})();
'''))
    assert result['first']['error']
    assert result['second']['error']
    assert result['third']['schedule']['id'] == 'schedule_one'
    assert result['taskPosts'] == result['schedulePosts'] == 1


def test_uncertain_task_missing_from_bounded_list_does_not_repost():
    result = json.loads(run_js('''
const scheduleAttempts = new Map();
const currentLang = 'en';
const crypto = {randomUUID: () => 'attempt-two'};
let posts = 0;
async function api(path, options) {
    if (options) { posts++; return {error:'Disconnected'}; }
    return {tasks:[]};
}
''' + js_function('saveScheduleAttempt') + '''
(async () => {
    const args = ['form', {id:'a'}, 'goal', {kind:'interval',seconds:60}];
    await saveScheduleAttempt(...args);
    await saveScheduleAttempt(...args);
    const changed = await saveScheduleAttempt('form', {id:'a'}, 'different', args[3]);
    console.log(JSON.stringify({posts,blocked:!!changed.error}));
})();
'''))
    assert result == {'posts': 1, 'blocked': True}


def test_canvas_refresh_cannot_replace_a_draft_opened_during_fetch():
    result = json.loads(run_js('''
let currentView = 'manage', selectedAgentId = 'a';
async function api() { currentView = 'create'; return {agent:{id:'a',name:'server'}}; }
const document = {getElementById: () => { throw Error('Draft was touched'); }};
''' + js_function('selectAgent') + '''
(async () => {
    const result = await selectAgent('a', {onlyWhileManaging:true});
    console.log(JSON.stringify({ignored:result === undefined,currentView}));
})();
'''))
    assert result == {'ignored': True, 'currentView': 'create'}


def test_invalid_schedule_never_creates_a_task():
    result = json.loads(run_js('''
const currentLang = 'en';
const scheduleAttempts = new Map();
async function api() { throw Error('Invalid schedule performed a write'); }
''' + js_function('saveScheduleAttempt') + '''
(async () => {
    const invalid = [
      {kind:'interval',seconds:NaN}, {kind:'daily_at',time:'24:00'},
      {kind:'interval',seconds:0}, {kind:'weekly',day:'Monday'}
    ];
    const results = await Promise.all(invalid.map(expression =>
      saveScheduleAttempt('form', {id:'a'}, 'goal', expression)));
    console.log(JSON.stringify(results.map(result => !!result.error)));
})();
'''))
    assert result == [True, True, True, True]


def test_uncertain_schedule_survives_reload_and_waits_for_late_commit():
    result = json.loads(run_js('''
let saved = null;
const sessionStorage = {getItem:()=>saved, setItem:(_key,value)=>{saved=value}};
const scheduleAttempts = new Map();
const currentLang = 'en';
const crypto = {randomUUID: () => 'reload-attempt'};
let posts = 0, committed = false;
async function api(path, options) {
    if (path === '/tasks') return {task:{id:'task-one'}};
    if (options) { posts++; return {error:'Lost response'}; }
    return {schedules:committed ? [{id:'late-schedule'}] : []};
}
''' + js_function('restoreScheduleAttempts') + js_function('saveScheduleAttempt') + '''
(async () => {
    const args = ['form', {id:'a'}, 'goal', {kind:'interval',seconds:60}];
    await saveScheduleAttempt(...args);
    const recovered = restoreScheduleAttempts();
    scheduleAttempts.clear();
    recovered.forEach(([key,value]) => scheduleAttempts.set(key,value));
    const pending = await saveScheduleAttempt(...args);
    committed = true;
    const completed = await saveScheduleAttempt(...args);
    console.log(JSON.stringify({posts,blocked:!!pending.error,completed,saved:JSON.parse(saved)}));
})();
'''))
    assert result['posts'] == 1
    assert result['blocked'] is True
    assert result['completed']['schedule']['id'] == 'late-schedule'
    assert result['saved'] == []
