/* Same server browser session as the mobile handoff; no credentials in drafts. */
const BrowserHandoff = (() => {
  function mount(host, data, item, {api, base, esc, linkRun, onComplete}) {
    const cp = data.checkpoint || {};
    const runId = data.run?.id || item.run_id;
    const taskId = data.task?.id || item.task_id;
    const sessionId = cp.browser_session_id;
    let socket, peer, channel, timer, closed = false, connected = false, attempt = 0;
    const terminal = ["completed", "done", "failed", "blocked", "skipped", "cancelled", "canceled", "aborted"].includes(data.run?.status);
    host.innerHTML = `<h2>${terminal ? "종료된 브라우저 요청" : cp.reason === "login_required" ? "로그인이 필요합니다" : "브라우저에서 확인이 필요합니다"}</h2>
      <p>${linkRun(runId)}</p><p data-handoff-guide></p>
      <p data-handoff-location></p><p role="status" data-handoff-status></p>
      <button data-handoff-open>로그인 화면 열기</button>
      <button data-handoff-close hidden>화면 닫기</button>
      <div data-handoff-player hidden>
        <video data-handoff-video autoplay muted playsinline tabindex="0" aria-label="원격 로그인 브라우저" style="display:block;width:100%;height:auto;background:#111"></video>
        <p>화면의 입력란을 클릭해 입력하세요. 한글·붙여넣기는 아래 입력란을 사용하세요.</p>
        <label>선택한 브라우저 입력란에 보낼 내용 <input data-handoff-text type="password" autocomplete="off" aria-label="브라우저 입력 내용"></label>
        <button data-handoff-send disabled>입력 보내기</button>
        <button data-handoff-key="Tab">다음 입력란</button>
        <button data-handoff-key="Enter">Enter</button>
        <button data-handoff-key="Backspace">한 글자 지우기</button>
      </div><button data-handoff-done disabled>로그인 완료 · 계속하기</button>`;
    const find = selector => host.querySelector(selector);
    const status = find('[data-handoff-status]');
    const open = find('[data-handoff-open]');
    const done = find('[data-handoff-done]');
    const video = find('video');
    const textInput = find('[data-handoff-text]');
    const guide = find('[data-handoff-guide]');
    if (cp.reason !== 'login_required') {
      open.textContent = '브라우저 열기';
      done.textContent = '확인 완료 · 계속하기';
    }
    guide.textContent = terminal
      ? "이미 종료된 실행의 이전 로그인 요청입니다. 이 요청으로 로그인하거나 실행을 재개할 수 없습니다. 작업의 실행 기록을 확인하세요."
      : !sessionId || !taskId
      ? "이 요청에 연결된 브라우저 세션이 없습니다. 실행 기록에서 작업을 확인하세요."
      : "서버에서 실행 중인 브라우저를 열어 직접 로그인한 뒤 계속하기를 누르세요. 자동화가 같은 단계에서 로그인 상태를 다시 확인합니다.";
    if (terminal || !sessionId || !taskId) {
      open.hidden = true; done.hidden = true;
      return {close() {}};
    }
    function showLocation(url, title) {
      const node = find('[data-handoff-location]');
      try {
        const parsed = new URL(url);
        node.textContent = parsed.protocol === 'file:' ? '로컬 테스트 페이지' : parsed.origin;
      } catch { node.textContent = title || '브라우저 위치 확인 중'; }
    }
    showLocation(data.step?.output?.browser_action?.current_url || data.step?.input?.target_url);
    function disconnect() {
      attempt++;
      clearTimeout(timer); connected = false; done.disabled = true;
      find('[data-handoff-send]').disabled = true;
      if (socket) { socket.onclose = null; socket.onerror = null; socket.close(); }
      channel?.close(); peer?.close(); socket = peer = channel = null;
      video.srcObject = null; textInput.value = '';
    }
    function fail(message) {
      if (closed) return;
      status.textContent = message;
      disconnect(); open.disabled = false; open.textContent = '다시 연결';
    }
    function ready() {
      if (closed || !connected || channel?.readyState !== 'open') return;
      clearTimeout(timer); done.disabled = false;
      find('[data-handoff-send]').disabled = false;
      status.textContent = '브라우저가 연결됐습니다. 로그인을 완료하세요.';
    }
    function send(message) {
      if (channel?.readyState !== 'open') return false;
      try { channel.send(JSON.stringify({...message, sensitive: true})); return true; }
      catch { return false; }
    }
    open.onclick = async () => {
      disconnect(); open.disabled = true; status.textContent = '브라우저에 연결 중…';
      const currentAttempt = attempt;
      find('[data-handoff-player]').hidden = false;
      find('[data-handoff-close]').hidden = false;
      try {
        const payload = await api(`${base}/tasks/${encodeURIComponent(taskId)}/browser-handoff`);
        if (closed || attempt !== currentAttempt) return;
        if (payload.browser_session?.id !== sessionId || payload.run?.id !== runId) {
          fail('요청이 변경됐습니다. 수신함을 새로고침해 현재 요청을 선택하세요.'); return;
        }
        showLocation(payload.browser_session.current_url, payload.browser_session.title);
        peer = new RTCPeerConnection({iceServers: []});
        const localPeer = peer;
        const url = new URL(`/ws/dashboard/agent/tasks/${encodeURIComponent(taskId)}/browser-handoff/rtc`, location.href);
        url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        url.searchParams.set('expected_browser_session_id', sessionId);
        url.searchParams.set('expected_run_id', runId);
        socket = new WebSocket(url); const localSocket = socket;
        const signal = value => { if (localSocket.readyState === WebSocket.OPEN) localSocket.send(JSON.stringify(value)); };
        localPeer.onicecandidate = event => { if (event.candidate) signal({type:'candidate', candidate:event.candidate.toJSON()}); };
        localPeer.ontrack = event => {
          if (closed || peer !== localPeer) return;
          video.srcObject = event.streams[0] || new MediaStream([event.track]);
          video.play().catch(() => {
            if (peer === localPeer && attempt === currentAttempt) fail('영상 재생을 시작하지 못했습니다. 다시 연결하세요.');
          });
        };
        video.onplaying = () => { connected = true; ready(); };
        localPeer.onconnectionstatechange = () => {
          if (peer === localPeer && ['failed','disconnected'].includes(localPeer.connectionState)) fail('브라우저 연결이 끊어졌습니다. 다시 연결하세요.');
        };
        localPeer.addTransceiver('video', {direction:'recvonly'});
        channel = localPeer.createDataChannel('browser-input', {ordered:true});
        channel.onopen = ready;
        channel.onmessage = event => {
          if (closed || peer !== localPeer) return;
          try {
            const message = JSON.parse(event.data);
            if (message.url) showLocation(message.url, message.title);
            if (message.ok === false) status.textContent = '브라우저 입력을 처리하지 못했습니다. 연결 상태를 확인하세요.';
          } catch { /* Non-JSON control messages do not change the UI. */ }
        };
        localSocket.onmessage = async event => {
          if (closed || socket !== localSocket) return;
          try {
            const message = JSON.parse(event.data);
            if (message.type === 'browser_rtc.ready') {
              if (message.browser_session?.id !== sessionId) { fail('브라우저 요청이 변경됐습니다. 수신함을 새로고침하세요.'); return; }
              const offer = await localPeer.createOffer();
              await localPeer.setLocalDescription(offer);
              signal({type:'offer', sdp:offer.sdp, sdpType:offer.type});
            } else if (message.type === 'answer') {
              await localPeer.setRemoteDescription({type:message.sdpType || 'answer', sdp:message.sdp});
            } else if (message.type === 'candidate') {
              await localPeer.addIceCandidate(message.candidate);
            } else if (message.type === 'browser_rtc.error') {
              fail('브라우저 세션을 열 수 없습니다. 만료되었거나 실행이 종료됐을 수 있습니다. 수신함을 새로고침하세요.');
            }
          } catch {
            if (peer === localPeer && attempt === currentAttempt) fail('브라우저 연결 처리에 실패했습니다. 다시 연결하세요.');
          }
        };
        localSocket.onerror = () => { if (socket === localSocket) fail('브라우저에 연결할 수 없습니다. 서버 상태를 확인하세요.'); };
        localSocket.onclose = () => { if (socket === localSocket) fail('브라우저 연결이 종료됐습니다. 다시 연결하세요.'); };
        timer = setTimeout(() => fail('브라우저 연결 시간이 초과됐습니다. 다시 연결하세요.'), 25000);
      } catch (error) {
        if (closed || attempt !== currentAttempt) return;
        fail(error.status === 409 ? '이 브라우저 요청은 만료되었거나 종료됐습니다. 수신함을 새로고침하세요.' : '브라우저 요청을 불러오지 못했습니다. 다시 연결하세요.');
      }
    };
    video.onclick = event => {
      const rect = video.getBoundingClientRect(); video.focus();
      send({type:'tap',x:(event.clientX-rect.left)/rect.width,y:(event.clientY-rect.top)/rect.height,normalized:true});
    };
    video.onwheel = event => { event.preventDefault(); send({type:'scroll',delta_x:event.deltaX,delta_y:event.deltaY}); };
    video.onkeydown = event => {
      if (event.isComposing || event.metaKey || event.ctrlKey || event.altKey) return;
      event.preventDefault();
      if (event.key.length === 1) send({type:'type_text',text:event.key});
      else if (['Enter','Tab','Backspace','Delete','ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Escape','Home','End'].includes(event.key)) send({type:'key',key:event.key});
    };
    find('[data-handoff-send]').onclick = () => {
      if (send({type:'type_text',text:textInput.value})) { textInput.value = ''; video.focus(); }
    };
    host.querySelectorAll('[data-handoff-key]').forEach(button => button.onclick = () => send({type:'key',key:button.dataset.handoffKey}));
    find('[data-handoff-close]').onclick = () => {
      disconnect(); find('[data-handoff-player]').hidden = true;
      find('[data-handoff-close]').hidden = true; open.disabled = false;
      status.textContent = '화면을 닫았습니다. 작업은 로그인 대기 상태로 유지됩니다.';
    };
    done.onclick = async () => {
      done.disabled = true; status.textContent = '같은 실행을 계속하도록 요청 중…';
      try {
        const result = await api(`${base}/tasks/${encodeURIComponent(taskId)}/browser-handoff/complete`, {
          method:'POST',body:JSON.stringify({message:'Browser handoff completed.',resume:true,expected_browser_session_id:sessionId,expected_run_id:runId}),
        });
        disconnect(); if (!closed) await onComplete(result);
      } catch (error) {
        fail(error.status === 409 ? '실행 상태가 바뀌어 재개하지 않았습니다. 수신함을 새로고침하세요.' : '재개 결과를 확인하지 못했습니다. 실행 기록을 확인하세요.');
      }
    };
    return {close() { closed = true; disconnect(); }};
  }
  return {mount};
})();
