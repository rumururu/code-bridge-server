import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeControlMessage, forwardControlMessage, isAllowedUdid } from './control_decoder.mjs';

test('configured device and QA allowlist intersect without changing default access', () => {
    assert.equal(isAllowedUdid('device', {}), true);
    assert.equal(isAllowedUdid('device', { CODEBRIDGE_AGENT_ANDROID_DEVICE_ID: 'device' }), true);
    assert.equal(isAllowedUdid('other', { CODEBRIDGE_AGENT_ANDROID_DEVICE_ID: 'device' }), false);
    assert.equal(isAllowedUdid('qa', { CODEBRIDGE_AGENT_ANDROID_DEVICE_ID: 'device', CODEBRIDGE_SCRCPY_ALLOWED_UDID: 'qa' }), false);
    assert.equal(isAllowedUdid('device', { CODEBRIDGE_AGENT_ANDROID_DEVICE_ID: 'device', CODEBRIDGE_SCRCPY_ALLOWED_UDID: 'qa' }), false);
    assert.equal(isAllowedUdid('device', { CODEBRIDGE_AGENT_ANDROID_DEVICE_ID: 'device', CODEBRIDGE_SCRCPY_ALLOWED_UDID: 'device' }), true);
});

test('decodes client key, text, and touch packets', () => {
    const key = Buffer.alloc(17);
    key.writeUInt32BE(67, 5);
    assert.deepEqual(decodeControlMessage(key), {
        type: 'key', value: { action: 0, keyCode: 67, repeat: 0, metaState: 0 },
    });
    const value = Buffer.from('한글');
    const text = Buffer.alloc(5 + value.length);
    text[0] = 1;
    text.writeUInt32BE(value.length, 1);
    value.copy(text, 5);
    assert.deepEqual(decodeControlMessage(text), { type: 'text', value: '한글' });
    const touch = Buffer.alloc(28);
    touch[0] = 2;
    touch.writeUInt32BE(40, 10);
    touch.writeUInt32BE(60, 14);
    touch.writeUInt16BE(100, 18);
    touch.writeUInt16BE(100, 20);
    touch.writeUInt16BE(65535, 22);
    touch.writeUInt32BE(1, 24);
    assert.equal(decodeControlMessage(touch).value.pointerX, 40);
});

test('rejects malformed and unsupported packets before controller calls', () => {
    for (const packet of [Buffer.alloc(0), Buffer.alloc(16), Buffer.from([1, 0, 0, 0, 8, 65]), Buffer.from([9])]) {
        assert.throws(() => decodeControlMessage(packet));
    }
    const touch = Buffer.alloc(28);
    touch[0] = 2;
    touch.writeUInt16BE(100, 18);
    touch.writeUInt16BE(100, 20);
    touch.writeUInt32BE(101, 10);
    touch.writeUInt32BE(1, 24);
    assert.throws(() => decodeControlMessage(touch), /Invalid touch/);
});

test('preserves pasted UTF-8 text larger than 4096 bytes', async () => {
    const value = '한글🙂'.repeat(700);
    const encoded = Buffer.from(value);
    const packet = Buffer.alloc(5 + encoded.length);
    packet[0] = 1;
    packet.writeUInt32BE(encoded.length, 1);
    encoded.copy(packet, 5);
    let received;
    await forwardControlMessage({ injectText: async text => { received = text; } }, packet);
    assert.equal(received, value);
});

test('dispatches only valid packets to the matching controller method', async () => {
    const calls = [];
    const controller = {
        injectKeyCode: async value => calls.push(['key', value]),
        injectText: async value => calls.push(['text', value]),
        injectTouch: async value => calls.push(['touch', value]),
    };
    const key = Buffer.alloc(17);
    key.writeUInt32BE(66, 5);
    await forwardControlMessage(controller, key);
    const text = Buffer.from([1, 0, 0, 0, 1, 65]);
    await forwardControlMessage(controller, text);
    const touch = Buffer.alloc(28);
    touch[0] = 2;
    touch.writeUInt16BE(100, 18);
    touch.writeUInt16BE(100, 20);
    touch.writeUInt32BE(1, 24);
    await forwardControlMessage(controller, touch);
    await assert.rejects(forwardControlMessage(controller, Buffer.from([9])), /Unsupported/);
    assert.deepEqual(calls.map(([type]) => type), ['key', 'text', 'touch']);
    assert.equal(calls[0][1].keyCode, 66);
    assert.equal(calls[1][1], 'A');
});
