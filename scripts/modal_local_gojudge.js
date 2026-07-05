import fs from 'fs/promises';
import path from 'path';
import os from 'os';
import { randomUUID } from 'crypto';
import { spawn } from 'child_process';

const CACHE_ROOT = '/tmp/local-gojudge-cache';

async function ensureDir(dir) {
    await fs.mkdir(dir, { recursive: true });
}

function parseEnv(envList = []) {
    const env = { ...process.env };
    for (const item of envList || []) {
        const split = String(item).indexOf('=');
        if (split >= 0) {
            env[String(item).slice(0, split)] = String(item).slice(split + 1);
        }
    }
    return env;
}

function limitText(value, maxBytes) {
    const text = value.toString('utf8');
    if (!maxBytes || Buffer.byteLength(text) <= maxBytes) return text;
    return text.slice(0, maxBytes);
}

async function cachedPath(fileId) {
    return path.join(CACHE_ROOT, fileId);
}

async function putCachedFile(sourcePath) {
    await ensureDir(CACHE_ROOT);
    const fileId = randomUUID();
    const target = await cachedPath(fileId);
    await fs.copyFile(sourcePath, target);
    await fs.chmod(target, 0o755).catch(() => {});
    return fileId;
}

async function putCachedContent(content, executable = false) {
    await ensureDir(CACHE_ROOT);
    const fileId = randomUUID();
    const target = await cachedPath(fileId);
    await fs.writeFile(target, content);
    if (executable) await fs.chmod(target, 0o755).catch(() => {});
    return fileId;
}

async function materializeCopyIn(copyIn = {}, workDir) {
    for (const [name, spec] of Object.entries(copyIn || {})) {
        const target = path.join(workDir, name);
        await ensureDir(path.dirname(target));
        if (spec && typeof spec === 'object' && Object.prototype.hasOwnProperty.call(spec, 'content')) {
            await fs.writeFile(target, spec.content ?? '');
        } else if (spec && typeof spec === 'object' && spec.fileId) {
            await fs.copyFile(await cachedPath(spec.fileId), target);
            await fs.chmod(target, 0o755).catch(() => {});
        } else {
            throw new Error(`Unsupported copyIn spec for ${name}`);
        }
    }
}

function executableFor(args, workDir) {
    const exe = args[0];
    if (!exe || exe.includes('/')) return exe;
    return path.join(workDir, exe);
}

async function runProcess(cmd, workDir) {
    const args = (cmd.args || []).map(String);
    if (!args.length) throw new Error('missing command args');

    const stdin = cmd.files?.[0]?.content ?? '';
    const stdoutMax = cmd.files?.[1]?.max ?? 64 * 1024 * 1024;
    const stderrMax = cmd.files?.[2]?.max ?? 64 * 1024 * 1024;
    const timeoutMs = Math.max(1000, Math.min(Number(cmd.clockLimit || cmd.cpuLimit || 300e9) / 1e6, 300000));
    const started = Date.now();

    return await new Promise((resolve) => {
        const child = spawn(executableFor(args, workDir), args.slice(1), {
            cwd: workDir,
            env: parseEnv(cmd.env),
            stdio: ['pipe', 'pipe', 'pipe'],
        });

        let stdout = Buffer.alloc(0);
        let stderr = Buffer.alloc(0);
        let timedOut = false;

        const timer = setTimeout(() => {
            timedOut = true;
            child.kill('SIGKILL');
        }, timeoutMs);

        child.stdout.on('data', (chunk) => {
            stdout = Buffer.concat([stdout, chunk]).subarray(0, stdoutMax);
        });
        child.stderr.on('data', (chunk) => {
            stderr = Buffer.concat([stderr, chunk]).subarray(0, stderrMax);
        });
        child.on('error', (error) => {
            clearTimeout(timer);
            resolve({
                status: 'Internal Error',
                exitStatus: 0,
                error: String(error),
                files: { stdout: '', stderr: String(error) },
                time: 0,
                memory: 0,
                runTime: 0,
            });
        });
        child.on('close', (code, signal) => {
            clearTimeout(timer);
            const runTime = (Date.now() - started) * 1e6;
            const files = {
                stdout: limitText(stdout, stdoutMax),
                stderr: limitText(stderr, stderrMax),
            };
            if (timedOut) {
                resolve({ status: 'Time Limit Exceeded', exitStatus: code ?? 0, error: signal || 'SIGKILL', files, time: runTime, memory: 0, runTime });
            } else if (signal) {
                resolve({ status: 'Signalled', exitStatus: code ?? 0, error: signal, files, time: runTime, memory: 0, runTime });
            } else if (code === 0) {
                resolve({ status: 'Accepted', exitStatus: 0, files, time: runTime, memory: 0, runTime });
            } else {
                resolve({ status: 'Runtime Error', exitStatus: code ?? 1, files, time: runTime, memory: 0, runTime });
            }
        });

        child.stdin.end(stdin);
    });
}

export class GoJudgeClient {
    constructor(baseURL = 'local-subprocess') {
        this.baseURL = baseURL;
    }

    async runOne(cmd) {
        const workDir = await fs.mkdtemp(path.join(os.tmpdir(), 'local-gojudge-'));
        try {
            await materializeCopyIn(cmd.copyIn || {}, workDir);
            const result = await runProcess(cmd, workDir);
            result.fileIds = result.fileIds || {};

            for (const name of cmd.copyOut || []) {
                try {
                    result.files = result.files || {};
                    result.files[name] = await fs.readFile(path.join(workDir, name), 'utf8');
                } catch {
                    if (name === 'stdout' || name === 'stderr') continue;
                }
            }

            for (const name of cmd.copyOutCached || []) {
                try {
                    result.fileIds[name] = await putCachedFile(path.join(workDir, name));
                } catch (error) {
                    result.status = 'Internal Error';
                    result.error = `copyOutCached failed for ${name}: ${error}`;
                }
            }
            return result;
        } finally {
            await fs.rm(workDir, { recursive: true, force: true }).catch(() => {});
        }
    }

    async run(cmds) {
        const commands = Array.isArray(cmds?.cmd) ? cmds.cmd : [];
        const results = [];
        for (const command of commands) {
            results.push(await this.runOne(command));
        }
        return results;
    }

    async deleteFile(fileId) {
        if (!fileId) return;
        await fs.rm(await cachedPath(fileId), { force: true }).catch(() => {});
    }

    async getFileContent(fileId) {
        return await fs.readFile(await cachedPath(fileId));
    }

    async copyInFile(filePath) {
        return await putCachedFile(filePath);
    }

    async cacheSingleFile(name, content) {
        return await putCachedContent(content, false);
    }

    async prepareProgram({ lang, code, mainName = null }) {
        if (lang === 'cpp') return await this._prepareCpp(code, mainName);
        if (lang === 'java') return await this._prepareJava(code, mainName);
        if (['py', 'pypy', 'python', 'python3'].includes(lang)) return await this._preparePython(code, mainName, lang);
        throw new Error('unsupported lang');
    }

    async _prepareCpp(code, mainName) {
        const srcName = mainName || 'main.cpp';
        const outName = 'a';
        const res = await this.runOne({
            args: ['/usr/bin/g++', srcName, '-O2', '-pipe', '-std=gnu++17', '-o', outName],
            env: ['PATH=/usr/bin:/bin'],
            files: [{ content: '' }, { name: 'stdout', max: 1024 * 1024 }, { name: 'stderr', max: 1024 * 1024 }],
            copyIn: { [srcName]: { content: code } },
            copyOut: ['stdout', 'stderr'],
            copyOutCached: [outName],
            cpuLimit: 10e9,
            memoryLimit: 512 << 20,
            procLimit: 50,
        });
        if (res.status !== 'Accepted') throw new Error(`compile failed: ${res.files?.stderr || res.error || res.status}`);
        const exeId = res.fileIds[outName];
        return { runArgs: [outName], preparedCopyIn: { [outName]: { fileId: exeId } }, cleanupIds: [exeId] };
    }

    async _prepareJava(code, mainName) {
        const srcName = mainName || 'Main.java';
        const mainClass = (srcName.replace(/\.java$/, '') || 'Main');
        const res = await this.runOne({
            args: ['/usr/bin/javac', srcName],
            env: ['PATH=/usr/bin:/bin'],
            files: [{ content: '' }, { name: 'stdout', max: 65536 }, { name: 'stderr', max: 65536 }],
            copyIn: { [srcName]: { content: code } },
            copyOut: ['stdout', 'stderr'],
            copyOutCached: [`${mainClass}.class`],
            cpuLimit: 10e9,
            memoryLimit: 1024 << 20,
            procLimit: 50,
        });
        if (res.status !== 'Accepted') throw new Error(`javac failed: ${res.files?.stderr || res.error || res.status}`);
        const clsId = res.fileIds[`${mainClass}.class`];
        return { runArgs: ['/usr/bin/java', mainClass], preparedCopyIn: { [`${mainClass}.class`]: { fileId: clsId } }, cleanupIds: [clsId] };
    }

    async _preparePython(code, mainName, lang) {
        const srcName = mainName || 'main.py';
        const fileId = await this.cacheSingleFile(srcName, code);
        const interp = lang === 'pypy' ? '/usr/bin/pypy3' : '/usr/bin/python3';
        return { runArgs: [interp, srcName], preparedCopyIn: { [srcName]: { fileId } }, cleanupIds: [fileId] };
    }

    async prepareChecker(checkerSourceText, testlibPath = '/lib/testlib', srcName = 'chk.cc') {
        const outName = 'chk';
        const res = await this.runOne({
            args: ['/usr/bin/g++', srcName, '-O2', '-pipe', '-std=gnu++17', '-I', testlibPath, '-o', outName],
            env: ['PATH=/usr/bin:/bin'],
            files: [{ content: '' }, { name: 'stdout', max: 65536 }, { name: 'stderr', max: 65536 }],
            copyIn: { [srcName]: { content: checkerSourceText } },
            copyOut: ['stdout', 'stderr'],
            copyOutCached: [outName],
            cpuLimit: 30e9,
            memoryLimit: 512 << 20,
            procLimit: 50,
        });
        if (res.status !== 'Accepted') throw new Error(`checker compile failed: ${res.files?.stderr || res.error || res.status}`);
        const checkerId = res.fileIds[outName];
        return { checkerId, cleanup: () => this.deleteFile(checkerId) };
    }

    async prepareInteractor() {
        throw new Error('interactive problems are not supported by the Modal local subprocess backend');
    }

    async getCheckerBin(checkerSourceText, testlibPath = '/lib/testlib', srcName = 'chk.cc') {
        const { checkerId, cleanup } = await this.prepareChecker(checkerSourceText, testlibPath, srcName);
        const checkerBin = await this.getFileContent(checkerId);
        cleanup();
        return checkerBin;
    }

    async getInteractorBin() {
        throw new Error('interactive problems are not supported by the Modal local subprocess backend');
    }

    async copyInBin(binPath) {
        const binId = await this.copyInFile(binPath);
        if (!binId) throw new Error('Failed to copy in binary');
        return { binId, cleanup: () => this.deleteFile(binId) };
    }
}
