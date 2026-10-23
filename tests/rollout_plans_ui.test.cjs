// Run with: node --test tests/rollout_plans_ui.test.cjs
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
        id, innerHTML: '', textContent: '', value: '', dataset: {}, style: {},
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
    run(`
        escapeHtml = value => String(value);
        showToast = () => {};
        renderRollout = () => renderRolloutPlans();
        globalThis.posted = [];
        globalThis.reply = {status: 'running'};
        api = async (path, request) => {
            if (path === '/rollout/plans') return {plans: globalThis.plans};
            posted.push([path, request && request.body ? JSON.parse(request.body) : null]);
            return globalThis.reply;
        };
        globalThis.plans = [
            {name: 'pick_place', kind: 'chain', about: 'pick, place, release', steps: ['pick', 'place', 'release']},
            {name: 'insert', kind: 'practice', checkpoint: 'training/insert', task: '', reward: 'seated',
                episodes: 0, learner: '10.0.0.2:50051'},
            {name: 'broken', kind: null, error: 'a chain plan needs at least one step'},
        ];
    `);
    const html = () => element('rollout-plans').innerHTML;
    return { run, element, html };
}

test('plans are offered by kind, an unusable one says why, and one starts by name', async () => {
    const { run, element, html } = harness();
    run(`state.rolloutStatus = {status: 'idle'};`);
    await run('refreshRolloutPlans()');
    assert.match(html(), /<optgroup label="Chains"><option value="pick_place">pick_place<\/option><\/optgroup>/);
    assert.match(html(), /<optgroup label="Practice"><option value="insert">insert<\/option><\/optgroup>/);
    assert.doesNotMatch(html(), /<option value="broken"/);
    assert.match(html(), /broken: a chain plan needs at least one step/);
    assert.match(html(), /id="rollout-plan-start" type="button" disabled|id="rollout-plan-start" disabled/);
    run(`state.rolloutPlanName = 'pick_place'; renderRolloutPlans();`);
    assert.match(html(), /Steps: pick → place → release/);
    assert.match(html(), />Start chain</);
    await element('rollout-plan-start').onclick();
    run(`assert.deepEqual(posted, [['/rollout/plans/pick_place/start', null]]);`);
    run(`state.rolloutStatus = {status: 'idle'}; state.rolloutPlanName = 'insert'; renderRolloutPlans();`);
    assert.match(html(), /Practises <code>training\/insert<\/code> · reward <strong>seated<\/strong> · until stopped/);
    assert.match(html(), />Start practice</);
});

test('with no plans the panel says where to write one', async () => {
    const { run, html } = harness();
    run(`state.rolloutStatus = {status: 'idle'}; globalThis.plans = [];`);
    await run('refreshRolloutPlans()');
    assert.match(html(), /No plans yet\. Write a chain or practice plan at <code>rollouts\/plans\/&lt;name&gt;\.json<\/code>/);
    assert.doesNotMatch(html(), /rollout-plan-start/);
});

const WAITING_CHAIN = `{
    state: 'waiting for the operator', step: 'insert', step_index: 1, steps: ['approach', 'insert', 'lift'],
    try: 3, tries: 3, waiting_for_operator: 'insert: 3 tries failed', record: '/s/rollouts/chains/20261023_010203.json',
    result: null,
    recent: [
        {t: 4.0, step: 'approach', event: 'try', attempt: 1, outcome: {passed: true, reason: 'check ready passed', seconds: 4.0}},
        {t: 25.2, step: 'insert', event: 'try', attempt: 3, outcome: {passed: false, reason: 'time limit', seconds: 20.0}},
        {t: 25.3, step: 'insert', event: 'operator called', why: '3 tries failed'},
    ],
}`;

test('a chain waiting for the operator asks to retry or skip, and shows its steps and events', async () => {
    const { run, element, html } = harness();
    run(`state.rolloutStatus = {status: 'running', chain: ${WAITING_CHAIN}}; renderRolloutPlans();`);
    const shown = html();
    assert.match(shown, /<strong>Chain<\/strong> · Waiting for you/);
    assert.match(shown, /rollout-chain-step rollout-chain-step--done">approach</);
    assert.match(shown, /rollout-chain-step rollout-chain-step--current">insert</);
    assert.match(shown, /rollout-chain-step">lift</);
    assert.match(shown, /You are needed<\/strong> at insert: 3 tries failed/);
    // Newest first; a failed try stands out.
    assert.match(shown, /25\.3 s · insert · operator called: 3 tries failed<\/li>\s*<li class="rollout-chain-event--failed">25\.2 s · insert · try 3: time limit \(20\.0 s\)/);
    assert.match(shown, /Record: <code>20261023_010203\.json<\/code>/);
    assert.match(shown, /id="rollout-plan-stop"/);
    // While it runs, the panel is about the chain, not the plans.
    assert.doesNotMatch(shown, /id="rollout-plan"|rollout-plan-start/);
    await element('rollout-chain-retry').onclick();
    await element('rollout-chain-skip').onclick();
    run(`assert.deepEqual(posted, [
        ['/rollout/chain/continue', {action: 'retry'}],
        ['/rollout/chain/continue', {action: 'skip'}],
    ]);`);
});

test('a finished chain shows its result and offers no stop', () => {
    const { run, html } = harness();
    run(`
        const chain = ${WAITING_CHAIN};
        state.rolloutStatus = {status: 'idle', chain: {...chain, result: 'done', step_index: 3,
            waiting_for_operator: null, state: 'done'}};
        renderRolloutPlans();
    `);
    assert.match(html(), /<strong>Chain<\/strong> · Done/);
    assert.doesNotMatch(html(), /You are needed/);
    assert.doesNotMatch(html(), /rollout-plan-stop/);
    assert.equal((html().match(/rollout-chain-step--done/g) || []).length, 3);
});

const PRACTICE = `{
    state: 'waiting for the operator', learner: '10.0.0.2:50051', reward: 'seated',
    episodes: 4, successes: 1, steps: 200, intervention_steps: 50, weights_loaded: 3,
    recent: [
        {episode: 3, steps: 100, success: false, interventions: 0, seconds: 10.0},
        {episode: 4, steps: 40, success: true, interventions: 12, seconds: 4.2},
    ],
}`;

test('practice shows its totals and asks the operator to go on after a success', async () => {
    const { run, element, html } = harness();
    run(`state.rolloutStatus = {status: 'running', practice: ${PRACTICE}}; renderRolloutPlans();`);
    const shown = html();
    assert.match(shown, /<strong>Practice<\/strong> · waiting for the operator · reward <strong>seated<\/strong>/);
    assert.match(shown, /4 episodes, 1 success \(25%\) · 200 steps, 25% under the operator · new weights 3 times/);
    assert.match(shown, /<td>4<\/td><td>Success<\/td><td>40<\/td><td>12<\/td><td>4\.2 s<\/td>[\s\S]*<td>3<\/td><td>Cut short<\/td>/);
    assert.match(shown, /Practice waits for you/);
    await element('rollout-practice-continue').onclick();
    run(`assert.deepEqual(posted, [['/rollout/practice/continue', null]]);`);
});

test('a stopped practice keeps its totals in view', () => {
    const { run, html } = harness();
    run(`state.rolloutStatus = {status: 'running', practice: ${PRACTICE}}; renderRolloutPlans();`);
    run(`state.rolloutStatus = {status: 'idle', practice: null}; renderRolloutPlans();`);
    const shown = html();
    assert.match(shown, /<strong>Practice<\/strong> · ended/);
    assert.match(shown, /4 episodes, 1 success/);
    assert.doesNotMatch(shown, /Practice waits for you/);
    assert.doesNotMatch(shown, /rollout-plan-stop/);
});
