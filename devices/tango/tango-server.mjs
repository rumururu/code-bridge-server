#!/usr/bin/env node

/**
 * Tango Scrcpy Unified Server
 * HTTP + WebSocket server for all Android devices
 *
 * Usage:
 *   node tango-server.mjs [port]
 *
 * HTTP:
 *   http://localhost:PORT/ - Web UI
 *
 * WebSocket endpoints:
 *   ws://localhost:PORT/stream?udid=DEVICE_SERIAL&displayId=0&maxSize=720&maxFps=30
 *   ws://localhost:PORT/devices
 *   ws://localhost:PORT/displays?udid=DEVICE_SERIAL
 */

import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { WebSocketServer, WebSocket } from 'ws';
import { TangoScrcpySession, getDevices, getDisplays, assertAllowedUdid } from './src/server/goog-device/tango/TangoScrcpyService.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = parseInt(process.argv[2]) || 8000;
const TAG = '[TangoServer]';
const PUBLIC_DIR = path.join(__dirname, 'public');

// MIME types
const MIME_TYPES = {
    '.html': 'text/html',
    '.css': 'text/css',
    '.js': 'application/javascript',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
    '.wasm': 'application/wasm',
};

// Active sessions
const sessions = new Map();

// Create HTTP server
const httpServer = http.createServer((req, res) => {
    // API endpoints (REST)
    if (req.url.startsWith('/api/')) {
        handleApiRequest(req, res);
        return;
    }

    // Static file serving
    let filePath = req.url === '/' ? '/index.html' : req.url;

    // Remove query string
    filePath = filePath.split('?')[0];

    const fullPath = path.join(PUBLIC_DIR, filePath);

    // Security: prevent directory traversal
    if (!fullPath.startsWith(PUBLIC_DIR)) {
        res.writeHead(403);
        res.end('Forbidden');
        return;
    }

    fs.readFile(fullPath, (err, data) => {
        if (err) {
            if (err.code === 'ENOENT') {
                res.writeHead(404);
                res.end('Not Found');
            } else {
                res.writeHead(500);
                res.end('Internal Server Error');
            }
            return;
        }

        const ext = path.extname(fullPath).toLowerCase();
        const contentType = MIME_TYPES[ext] || 'application/octet-stream';

        res.writeHead(200, { 'Content-Type': contentType });
        res.end(data);
    });
});

// REST API handler
async function handleApiRequest(req, res) {
    const url = new URL(req.url, `http://localhost:${PORT}`);

    res.setHeader('Content-Type', 'application/json');

    try {
        if (url.pathname === '/api/devices') {
            const devices = await getDevices();
            res.writeHead(200);
            res.end(JSON.stringify({
                type: 'devices',
                data: devices.map(d => ({
                    serial: d.serial,
                    model: d.model || 'Unknown',
                })),
            }));
            return;
        }

        if (url.pathname === '/api/displays') {
            const udid = url.searchParams.get('udid');
            if (!udid) {
                res.writeHead(400);
                res.end(JSON.stringify({ error: 'Missing udid parameter' }));
                return;
            }

            assertAllowedUdid(udid);

            const displays = await getDisplays(udid);
            res.writeHead(200);
            res.end(JSON.stringify({
                type: 'displays',
                data: displays.map(d => ({
                    id: d.id,
                    resolution: d.resolution || 'Unknown',
                })),
            }));
            return;
        }

        res.writeHead(404);
        res.end(JSON.stringify({ error: 'Unknown API endpoint' }));
    } catch (error) {
        console.error(`${TAG} API error:`, error.message);
        res.writeHead(500);
        res.end(JSON.stringify({ error: error.message }));
    }
}

// Create WebSocket server attached to HTTP server
const wss = new WebSocketServer({ server: httpServer });

wss.on('connection', async (ws, req) => {
    const url = new URL(req.url, `http://localhost:${PORT}`);
    const pathname = url.pathname;

    console.log(`${TAG} WebSocket connection: ${pathname}`);

    try {
        if (pathname === '/devices' || pathname === '/api/devices') {
            const devices = await getDevices();
            ws.send(JSON.stringify({
                type: 'devices',
                data: devices.map(d => ({
                    serial: d.serial,
                    model: d.model || 'Unknown',
                })),
            }));
            ws.close();
            return;
        }

        if (pathname === '/displays' || pathname === '/api/displays') {
            const udid = url.searchParams.get('udid');
            if (!udid) {
                ws.send(JSON.stringify({ type: 'error', message: 'Missing udid parameter' }));
                ws.close();
                return;
            }

            assertAllowedUdid(udid);

            const displays = await getDisplays(udid);
            ws.send(JSON.stringify({
                type: 'displays',
                data: displays.map(d => ({
                    id: d.id,
                    resolution: d.resolution || 'Unknown',
                })),
            }));
            ws.close();
            return;
        }

        if (pathname === '/stream' || pathname === '/api/stream' || pathname.startsWith('/stream-tango')) {
            const udid = url.searchParams.get('udid');
            if (!udid) {
                ws.send(JSON.stringify({ type: 'error', message: 'Missing udid parameter' }));
                ws.close();
                return;
            }

            assertAllowedUdid(udid);

            const displayId = parseInt(url.searchParams.get('displayId') || '0', 10);
            const maxSize = parseInt(url.searchParams.get('maxSize') || '720', 10);
            const maxFps = parseInt(url.searchParams.get('maxFps') || '30', 10);
            const bitrate = parseInt(url.searchParams.get('bitrate') || '2000000', 10);

            const session = new TangoScrcpySession(udid, {
                displayId,
                maxSize,
                maxFps,
                bitrate,
            });

            const sessionId = `${udid}-${displayId}-${Date.now()}`;
            sessions.set(sessionId, session);

            session.onVideoPacket = (packet) => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(packet.data);
                }
            };

            session.onMetadata = (metadata) => {
                if (ws.readyState === WebSocket.OPEN) {
                    const initialInfo = buildInitialInfo(metadata, displayId);
                    ws.send(initialInfo);
                }
            };

            session.onError = (error) => {
                console.error(`${TAG} Session error:`, error.message);
                if (ws.readyState === WebSocket.OPEN) {
                    ws.close(4005, error.message);
                }
            };

            session.onClose = () => {
                console.log(`${TAG} Session closed: ${sessionId}`);
                sessions.delete(sessionId);
            };

            ws.on('close', () => {
                console.log(`${TAG} Client disconnected: ${sessionId}`);
                session.stop();
                sessions.delete(sessionId);
            });

            ws.on('error', (err) => {
                console.error(`${TAG} WebSocket error:`, err.message);
                session.stop();
                sessions.delete(sessionId);
            });

            ws.on('message', (data) => {
                session.sendControlMessage(data).catch((error) => {
                    console.warn(`${TAG} Rejected control message: ${error.message}`);
                });
            });

            console.log(`${TAG} Starting session: ${sessionId}`);
            await session.start();
            return;
        }

        ws.send(JSON.stringify({
            type: 'error',
            message: `Unknown endpoint: ${pathname}`,
            endpoints: ['/devices', '/displays?udid=SERIAL', '/stream?udid=SERIAL'],
        }));
        ws.close();

    } catch (error) {
        console.error(`${TAG} Error:`, error.message);
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'error', message: error.message }));
            ws.close();
        }
    }
});

/**
 * Build initial info buffer compatible with ws-scrcpy client
 */
function buildInitialInfo(metadata, displayId) {
    const magicBytes = Buffer.from('scrcpy_initial');

    const deviceNameBuffer = Buffer.alloc(64);
    const deviceName = metadata.deviceName || 'Unknown';
    deviceNameBuffer.write(deviceName, 0, 'utf8');

    const displayCountBuffer = Buffer.alloc(4);
    displayCountBuffer.writeInt32BE(1, 0);

    const displayInfoBuffer = Buffer.alloc(24);
    let offset = 0;

    displayInfoBuffer.writeInt32BE(displayId, offset);
    offset += 4;

    displayInfoBuffer.writeUInt16BE(metadata.width || 720, offset);
    offset += 2;

    displayInfoBuffer.writeUInt16BE(metadata.height || 1280, offset);
    offset += 2;

    displayInfoBuffer.writeUInt8(0, offset);
    offset += 1;

    offset += 3; // Padding

    displayInfoBuffer.writeInt32BE(1, offset);
    offset += 4;

    displayInfoBuffer.writeInt32BE(0, offset);
    offset += 4;

    displayInfoBuffer.writeInt32BE(0, offset);
    offset += 4;

    const encoderName = 'OMX.qcom.video.encoder.avc';
    const encoderNameBuffer = Buffer.from(encoderName, 'utf8');
    const encodersBuffer = Buffer.alloc(4 + 4 + encoderNameBuffer.length);
    let encoderOffset = 0;
    encodersBuffer.writeInt32BE(1, encoderOffset);
    encoderOffset += 4;
    encodersBuffer.writeInt32BE(encoderNameBuffer.length, encoderOffset);
    encoderOffset += 4;
    encoderNameBuffer.copy(encodersBuffer, encoderOffset);

    const clientIdBuffer = Buffer.alloc(4);
    clientIdBuffer.writeInt32BE(1, 0);

    return Buffer.concat([
        magicBytes,
        deviceNameBuffer,
        displayCountBuffer,
        displayInfoBuffer,
        encodersBuffer,
        clientIdBuffer
    ]);
}

// Handle server errors
wss.on('error', (error) => {
    console.error(`${TAG} WebSocket server error:`, error.message);
});

httpServer.on('error', (error) => {
    console.error(`${TAG} HTTP server error:`, error.message);
});

// Graceful shutdown
process.on('SIGINT', async () => {
    console.log(`\n${TAG} Shutting down...`);

    for (const [id, session] of sessions) {
        console.log(`${TAG} Stopping session: ${id}`);
        await session.stop();
    }

    wss.close(() => {
        httpServer.close(() => {
            console.log(`${TAG} Server closed`);
            process.exit(0);
        });
    });
});

// Start server
httpServer.listen(PORT, '127.0.0.1', () => {
    console.log(`${TAG} Tango Scrcpy Server started on port ${PORT}`);
    console.log(`${TAG} ========================================`);
    console.log(`${TAG} HTTP:`);
    console.log(`${TAG}   http://localhost:${PORT}/ - Web UI`);
    console.log(`${TAG}   http://localhost:${PORT}/api/devices - Device list (REST)`);
    console.log(`${TAG}   http://localhost:${PORT}/api/displays?udid=SERIAL - Display list (REST)`);
    console.log(`${TAG} WebSocket:`);
    console.log(`${TAG}   ws://localhost:${PORT}/devices`);
    console.log(`${TAG}   ws://localhost:${PORT}/displays?udid=SERIAL`);
    console.log(`${TAG}   ws://localhost:${PORT}/stream?udid=SERIAL&displayId=0&maxSize=720&maxFps=30`);
});
