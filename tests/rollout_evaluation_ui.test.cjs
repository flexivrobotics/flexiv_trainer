// Run with: node --test tests/rollout_evaluation_ui.test.cjs
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../src/flexivtrainer/web/app.js'), 'utf8');
const startup = 'init().catch((error) => showToast(error.message, true));';
assert.equal(source.split(startup).length, 2);

function harness() {
    const elements = new Map();
    const makeElement = (id = '') => ({
        id, innerHTML: '', textContent: '', value: '', checked: false, dataset: {}, style: {},
        classList: { toggle() {}, remove() {}, add() {} },
        querySelector() { return null; }, querySelectorAll() { return []; },
        appendChild() {}, setAttribute() {},
    });
    const element = (id) => {
        if (!elements.has(id)) elements.set(id, makeElement(id));
        return elements.get(id);
    };
    const context = vm.createContext({
        assert,
        HTMLInputElement: class {},
        console: { warn() {} },
        document: { getElementById: element, createElement: () => makeElement() },
    });
    const run = (code) => vm.runInContext(code, context);
    run(source.replace(startup, ''));
    run('escapeHtml = value => String(value); showToast = () => {};');
    return { run, element };
}

const BLIND_TRIAL = `{
    status: 'running', checkpoint_path: null, checkpoint_label: 'B',
    trial: {session: 'cmp', trial: 4, arm: 'B', condition: 'rim', description: 'on the rim', blind: true},
    recording: {episode: '20261015_101010_t004', frames: 12, dropped_frames: 0, saving: false, error: null},
    awaiting_verdict: null, send_twist_default: false,
}`;

test('a blind trial shows its arm and condition, never a checkpoint path', () => {
    const { run, element } = harness();
    run(`state.rolloutStatus = ${BLIND_TRIAL}; renderRolloutEvaluation();`);
    const html = element('rollout-evaluation').innerHTML;
    assert.match(html, /Running trial 4/);
    assert.match(html, /arm <strong>B<\/strong>/);
    assert.match(html, /on the rim/);
    assert.match(html, /12 frames/);
    assert.doesNotMatch(html, /pretrained_model|checkpoints\//);
    // The server default seeds the feedforward checkbox once.
    run('assert.equal(state.rolloutSendTwist, false);');
});

test('a failure verdict needs a failure mode and posts the right payload', async () => {
    const { run, element } = harness();
    run(`
        state.rolloutStatus = {status: 'idle', awaiting_verdict: {
            episode: 'e7', trial: null, failure_modes: ['hovered', 'stalled'], frames: 9,
        }};
        state.rolloutVerdictOutcome = 'failure';
        renderRolloutEvaluation();
    `);
    const html = element('rollout-evaluation').innerHTML;
    assert.match(html, /How did <code>e7<\/code> go\?/);
    assert.match(html, /<option value="stalled">stalled<\/option>/);
    run(`
        globalThis.posted = [];
        api = async (path, request) => {
            if (request && request.method === 'POST') posted.push([path, JSON.parse(request.body)]);
            if (path === '/rollout/evaluations') return {sessions: []};
            return {status: 'idle', awaiting_verdict: null};
        };
    `);
    await run('saveRolloutVerdict()');
    run('assert.equal(posted.length, 0, "no failure mode chosen, nothing posted");');
    run(`state.rolloutVerdictMode = 'stalled'; state.rolloutVerdictNote = 'on the rim';`);
    await run('saveRolloutVerdict()');
    run(`
        assert.equal(posted.length, 1);
        assert.equal(posted[0][0], '/rollout/verdict');
        assert.deepEqual(posted[0][1], {outcome: 'failure', note: 'on the rim', failure_mode: 'stalled'});
        assert.equal(state.rolloutVerdictOutcome, '');
    `);
});

test('a trial left without a verdict by a restart is judged by name', async () => {
    const { run } = harness();
    run(`
        state.rolloutStatus = {status: 'idle', awaiting_verdict: null};
        state.rolloutEvaluationName = 'cmp';
        state.rolloutEvaluation = {name: 'cmp', next: null, failure_modes: ['hovered'],
            awaiting_verdict: [{trial: 2, episode: 'e2'}], summary: {arms: {}}};
        state.rolloutVerdictOutcome = 'success';
        globalThis.posted = [];
        api = async (path, request) => {
            if (request && request.method === 'POST') posted.push([path, JSON.parse(request.body)]);
            if (path === '/rollout/evaluations') return {sessions: [{name: 'cmp', total: 4, judged: 1}]};
            if (path.startsWith('/rollout/evaluations/')) return state.rolloutEvaluation;
            return {status: 'idle', awaiting_verdict: null};
        };
    `);
    await run('saveRolloutVerdict()');
    run(`assert.deepEqual(posted[0][1], {outcome: 'success', note: '', session: 'cmp', episode: 'e2'});`);
});

test('a single rollout carries the recording and feedforward choices', async () => {
    const { run } = harness();
    run(`
        state.rolloutCheckpointSource = 'local';
        state.rolloutCheckpointPath = '/runs/a/checkpoints/010000';
        state.rolloutRecord = true;
        state.rolloutSendTwist = false;
        globalThis.posted = [];
        api = async (path, request) => {
            if (request && request.method === 'POST') posted.push([path, JSON.parse(request.body)]);
            return {status: 'running'};
        };
        refreshTeleopStatus = async () => {};
        refreshSummary = async () => {};
        renderRollout = () => {};
    `);
    await run('startRolloutRun()');
    run(`
        assert.equal(posted[0][0], '/rollout/start');
        assert.equal(posted[0][1].record, true);
        assert.equal(posted[0][1].send_twist, false);
    `);
    run('state.rolloutSendTwist = null; posted.length = 0;');
    await run('startRolloutRun()');
    run('assert.ok(!("send_twist" in posted[0][1]), "unknown default: leave it to the server");');
});

test('the summary shows each arm with its interval', () => {
    const { run, element } = harness();
    run(`
        state.rolloutStatus = {status: 'idle', awaiting_verdict: null};
        state.rolloutEvaluationName = 'cmp';
        state.rolloutEvaluationSessions = [{name: 'cmp', total: 8, judged: 8}];
        state.rolloutEvaluation = {name: 'cmp', next: null, awaiting_verdict: [], failure_modes: [],
            summary: {arms: {A: {successes: 3, trials: 4, rate: 0.75, ci95: [0.3, 0.95], unknown: 0}}}};
        renderRolloutEvaluation();
    `);
    const html = element('rollout-evaluation').innerHTML;
    assert.match(html, /3\/4 · 75% \(30%–95%\)/);
    assert.match(html, /Every trial of this session has a verdict/);
});

test('checks are shown as agreement over judged trials, not per pending trial', () => {
    const { run, element } = harness();
    run(`
        state.rolloutStatus = {status: 'idle', awaiting_verdict: {
            episode: 'e9', trial: {session: 'cmp', trial: 9, arm: 'A', condition: 'home', blind: true},
            failure_modes: ['stalled'], frames: 30, stop_reason: 'done'}};
        state.rolloutEvaluationName = 'cmp';
        state.rolloutEvaluationSessions = [{name: 'cmp', total: 10, judged: 9}];
        state.rolloutEvaluation = {name: 'cmp', next: null, awaiting_verdict: [{trial: 9, episode: 'e9'}],
            failure_modes: ['stalled'], summary: {arms: {}, checks: {placed: {
                agree: 8, trials: 9, rate: 8 / 9, ci95: [0.56, 0.98], passed_but_failure: 1,
                failed_but_success: 0, unscored: 0}}}};
        renderRolloutEvaluation();
    `);
    const html = element('rollout-evaluation').innerHTML;
    assert.match(html, /placed/);
    assert.match(html, /8\/9 · 89% \(56%–98%\)/);
    assert.match(html, /How did <code>e9<\/code> go\?/);
});
