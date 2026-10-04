const decoder = new TextDecoder('utf-8', { fatal: true });

export function isAllowedUdid(udid, env = process.env) {
    const configured = env.CODEBRIDGE_AGENT_ANDROID_DEVICE_ID || env.ANDROID_SERIAL;
    const qa = env.CODEBRIDGE_SCRCPY_ALLOWED_UDID;
    return Boolean(udid) && (!configured || udid === configured) && (!qa || udid === qa);
}

export function decodeControlMessage(data) {
    if (!Buffer.isBuffer(data)) throw new Error('Binary control message required');
    const type = data[0];
    if (type === 0) {
        if (data.length !== 17) throw new Error('Invalid key message length');
        const action = data.readUInt32BE(1);
        const keyCode = data.readUInt32BE(5);
        if (action > 1 || keyCode > 288) throw new Error('Invalid key message');
        return { type: 'key', value: { action, keyCode, repeat: data.readUInt32BE(9), metaState: data.readUInt32BE(13) } };
    }
    if (type === 1) {
        if (data.length < 5) throw new Error('Invalid text message length');
        const length = data.readUInt32BE(1);
        if (!length || data.length !== 5 + length) throw new Error('Invalid text message length');
        return { type: 'text', value: decoder.decode(data.subarray(5)) };
    }
    if (type === 2) {
        if (data.length !== 28) throw new Error('Invalid touch message length');
        const action = data[1];
        const pointerId = data.readBigUInt64BE(2);
        const pointerX = data.readUInt32BE(10);
        const pointerY = data.readUInt32BE(14);
        const videoWidth = data.readUInt16BE(18);
        const videoHeight = data.readUInt16BE(20);
        const pressure = data.readUInt16BE(22) / 65535;
        const buttons = data.readUInt32BE(24);
        if (![0, 1, 2].includes(action) || !videoWidth || !videoHeight || pointerX > videoWidth || pointerY > videoHeight || pointerId !== 0n || buttons !== 1) {
            throw new Error('Invalid touch message');
        }
        return { type: 'touch', value: { action, pointerId, pointerX, pointerY, videoWidth, videoHeight, pressure, actionButton: 0, buttons } };
    }
    throw new Error('Unsupported control message');
}

export async function forwardControlMessage(controller, data) {
    const message = decodeControlMessage(data);
    if (message.type === 'key') await controller.injectKeyCode(message.value);
    else if (message.type === 'text') await controller.injectText(message.value);
    else await controller.injectTouch(message.value);
}
