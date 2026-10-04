/**
 * Tango Scrcpy Service - JavaScript module
 * Uses @yume-chan/adb-scrcpy for Android 12+ support
 * This is a JavaScript module to avoid TypeScript version conflicts
 */

import { AdbServerClient } from '@yume-chan/adb';
import { AdbServerNodeTcpConnector } from '@yume-chan/adb-server-node-tcp';
import { AdbScrcpyClient, AdbScrcpyOptionsLatest } from '@yume-chan/adb-scrcpy';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { forwardControlMessage, isAllowedUdid } from './control_decoder.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const TAG = '[TangoScrcpyService]';
const adbPort = 5037;

export function assertAllowedUdid(udid) {
    if (!isAllowedUdid(udid)) throw new Error('Device not allowed');
}

// Scrcpy server paths
const SCRCPY_SERVER_PATHS = [
    path.join(__dirname, '../../../vendor/Genymobile/scrcpy/scrcpy-server-3.3.3.jar'),
    '/tmp/ya-webadb-test/scrcpy-server-v3.3.3',
    path.join(process.cwd(), 'vendor/Genymobile/scrcpy/scrcpy-server-3.3.3.jar'),
];
const DEVICE_SERVER_PATH = '/data/local/tmp/scrcpy-server.jar';

/**
 * TangoScrcpySession - manages a single scrcpy streaming session
 */
export class TangoScrcpySession {
    constructor(udid, options = {}) {
        this.udid = udid;
        this.displayId = options.displayId || 0;
        this.maxSize = options.maxSize || 720;
        this.maxFps = options.maxFps || 30;
        this.bitrate = options.bitrate || 2000000;

        this.scrcpyClient = null;
        this.adbInstance = null;
        this.isRunning = false;
        this.videoReader = null;

        this.onVideoPacket = null;
        this.onMetadata = null;
        this.onError = null;
        this.onClose = null;
    }

    async start() {
        assertAllowedUdid(this.udid);
        try {
            console.log(`${TAG} Starting session for ${this.udid}, display ${this.displayId}`);

            // Create ADB connector
            const connector = new AdbServerNodeTcpConnector({
                host: '127.0.0.1',
                port: adbPort,
            });

            const client = new AdbServerClient(connector);

            // Get device
            const devices = await client.getDevices();
            const targetDevice = devices.find(d => d.serial === this.udid);

            if (!targetDevice) {
                throw new Error(`Device ${this.udid} not found`);
            }

            // Connect to device
            this.adbInstance = await client.createAdb({ serial: this.udid });
            console.log(`${TAG} Connected to ${this.adbInstance.banner.product}`);

            // Push scrcpy server
            await this.pushScrcpyServer();

            // Create scrcpy options
            const options = new AdbScrcpyOptionsLatest({
                video: true,
                audio: false,
                control: true,
                maxSize: this.maxSize,
                maxFps: this.maxFps,
                videoBitRate: this.bitrate,
                displayId: this.displayId,
                tunnelForward: true,
            });

            console.log(`${TAG} Options:`, options.serialize());

            // Start scrcpy
            this.scrcpyClient = await AdbScrcpyClient.start(
                this.adbInstance,
                DEVICE_SERVER_PATH,
                options
            );

            this.isRunning = true;
            this.scrcpyClient.output.pipeTo(new WritableStream({
                write: line => console.log(`${TAG} Device: ${line}`),
            })).catch(error => console.debug(`${TAG} Device output ended: ${error.message}`));
            console.log(`${TAG} Scrcpy started`);

            // Get video stream
            const videoStream = await this.scrcpyClient.videoStream;
            if (!videoStream) {
                throw new Error('No video stream available');
            }

            // Notify metadata
            if (this.onMetadata) {
                this.onMetadata(videoStream.metadata);
            }

            // Start streaming
            this.streamVideo(videoStream);

            return {
                metadata: videoStream.metadata,
                deviceName: videoStream.metadata.deviceName,
                width: videoStream.metadata.width,
                height: videoStream.metadata.height,
            };

        } catch (error) {
            console.error(`${TAG} Error:`, error.message);
            if (this.onError) {
                this.onError(error);
            }
            await this.stop();
            throw error;
        }
    }

    async pushScrcpyServer() {
        if (!this.adbInstance) return;

        console.log(`${TAG} Pushing scrcpy server...`);

        let serverBinary = null;
        for (const serverPath of SCRCPY_SERVER_PATHS) {
            if (fs.existsSync(serverPath)) {
                serverBinary = fs.readFileSync(serverPath);
                console.log(`${TAG} Using: ${serverPath}`);
                break;
            }
        }

        if (!serverBinary) {
            throw new Error(`Scrcpy server not found: ${SCRCPY_SERVER_PATHS.join(', ')}`);
        }

        const sync = await this.adbInstance.sync();
        await sync.write({
            filename: DEVICE_SERVER_PATH,
            file: new ReadableStream({
                start(controller) {
                    controller.enqueue(serverBinary);
                    controller.close();
                },
            }),
        });
        await sync.dispose();

        console.log(`${TAG} Server pushed (${serverBinary.length} bytes)`);
    }

    async streamVideo(videoStream) {
        this.videoReader = videoStream.stream.getReader();

        console.log(`${TAG} Streaming video...`);

        try {
            while (this.isRunning) {
                const { value, done } = await this.videoReader.read();
                if (done) break;

                if (value && value.data && this.onVideoPacket) {
                    this.onVideoPacket({
                        type: value.type,
                        data: value.data,
                    });
                }
            }
        } catch (error) {
            if (this.isRunning) {
                console.error(`${TAG} Stream error:`, error.message);
                if (this.onError) {
                    this.onError(error);
                }
            }
        } finally {
            if (this.videoReader) {
                this.videoReader.releaseLock();
            }
        }

        console.log(`${TAG} Stream ended`);
        if (this.onClose) {
            this.onClose();
        }
    }

    async stop() {
        if (!this.isRunning) return;

        this.isRunning = false;
        console.log(`${TAG} Stopping session...`);

        try {
            if (this.videoReader) {
                await this.videoReader.cancel();
                this.videoReader = null;
            }

            if (this.scrcpyClient) {
                await this.scrcpyClient.close();
                this.scrcpyClient = null;
            }

            if (this.adbInstance) {
                await this.adbInstance.close();
                this.adbInstance = null;
            }
        } catch (error) {
            console.error(`${TAG} Stop error:`, error.message);
        }

        console.log(`${TAG} Session stopped`);
    }

    async sendControlMessage(data) {
        const controller = this.scrcpyClient?.controller;
        if (!this.isRunning || !controller) return;
        await forwardControlMessage(controller, data);
    }
}

/**
 * Get list of connected devices
 */
export async function getDevices() {
    const connector = new AdbServerNodeTcpConnector({
        host: '127.0.0.1',
        port: adbPort,
    });

    const client = new AdbServerClient(connector);
    return (await client.getDevices()).filter(device => {
        try { assertAllowedUdid(device.serial); return true; } catch { return false; }
    });
}

/**
 * Get displays for a device
 */
export async function getDisplays(udid) {
    assertAllowedUdid(udid);
    const connector = new AdbServerNodeTcpConnector({
        host: '127.0.0.1',
        port: adbPort,
    });

    const client = new AdbServerClient(connector);
    const adb = await client.createAdb({ serial: udid });

    // Push server first
    let serverBinary = null;
    for (const serverPath of SCRCPY_SERVER_PATHS) {
        if (fs.existsSync(serverPath)) {
            serverBinary = fs.readFileSync(serverPath);
            break;
        }
    }

    if (serverBinary) {
        const sync = await adb.sync();
        await sync.write({
            filename: DEVICE_SERVER_PATH,
            file: new ReadableStream({
                start(controller) {
                    controller.enqueue(serverBinary);
                    controller.close();
                },
            }),
        });
        await sync.dispose();
    }

    const displays = await AdbScrcpyClient.getDisplays(
        adb,
        DEVICE_SERVER_PATH,
        new AdbScrcpyOptionsLatest({
            video: true,
            audio: false,
            control: false,
        })
    );

    await adb.close();
    return displays;
}
