// Copyright 2026 Flexiv Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Run with: node --test tests/rollout_takeover_ui.test.cjs
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
    run('escapeHtml = value => String(value); showToast = () => {}; renderRollout = () => {};');
    return { run, element };
}

const POLICY = `{status: 'running', takeover: {
    available: true, state: 'policy', control: 'policy', error: null, count: 0,
    last_switch: null, action: 'take_over', reason: null, trigger_channel: null}}`;

test('under the policy the operator is offered the takeover', () => {
    const { run, element } = harness();
    run(`state.rolloutStatus = ${POLICY}; renderRolloutTakeover();`);
    const html = element('rollout-takeover').innerHTML;
    assert.match(html, /In control: <strong>Policy<\/strong>/);
    assert.match(html, />Take over<\/button>/);
    assert.doesNotMatch(html, /disabled/);
    assert.doesNotMatch(html, /Last switch/);
});

test('an unavailable takeover is disabled and says why', () => {
    const { run, element } = harness();
    run(`
        state.rolloutStatus = {status: 'running', takeover: {
            available: false, state: null, control: 'policy', error: null, count: 0,
            last_switch: null, action: null, trigger_channel: null,
            reason: 'Teleoperation was not connected when the rollout started'}};
        renderRolloutTakeover();
    `);
    const html = element('rollout-takeover').innerHTML;
    assert.match(html, /disabled/);
    assert.match(html, /Teleoperation was not connected when the rollout started/);
    assert.match(html, />Take over<\/button>/);
});

test('under the operator it shows the hand-back and the switch time, and posts it', async () => {
    const { run, element } = harness();
    run(`
        state.rolloutStatus = {status: 'running', takeover: {
            available: true, state: 'operator', control: 'operator', error: null, count: 1,
            last_switch: {to: 'operator', s: 0.8412}, action: 'hand_back', reason: null,
            trigger_channel: 3}};
        renderRolloutTakeover();
    `);
    const html = element('rollout-takeover').innerHTML;
    assert.match(html, /In control: <strong>Operator \(leader arm\)<\/strong>/);
    assert.match(html, />Hand back<\/button>/);
    assert.match(html, /Last switch: to the operator in 0\.84 s/);
    assert.match(html, /Leader input 3 also switches/);
    run(`
        globalThis.posted = [];
        api = async (path, request) => {
            posted.push([path, request && request.method]);
            return {status: 'running', takeover: {...state.rolloutStatus.takeover,
                state: 'switching_to_policy', control: 'switching', action: null,
                reason: 'A switch is under way'}};
        };
    `);
    await run("byId('rollout-takeover-action').onclick()");
    run(`
        assert.deepEqual(posted, [['/rollout/hand-back', 'POST']]);
        assert.equal(state.rolloutStatus.takeover.state, 'switching_to_policy');
        assert.equal(state.rolloutTakeoverBusy, false);
    `);
});

test('it re-renders when control changes and not otherwise', () => {
    const { run, element } = harness();
    run(`state.rolloutStatus = ${POLICY}; renderRolloutTakeover();`);
    element('rollout-takeover').innerHTML = 'unchanged';
    run('renderRolloutTakeover();');
    assert.equal(element('rollout-takeover').innerHTML, 'unchanged');
    run(`
        state.rolloutStatus = {status: 'running', takeover: {...state.rolloutStatus.takeover,
            state: 'switching_to_operator', control: 'switching', action: null,
            reason: 'A switch is under way'}};
        renderRolloutTakeover();
    `);
    const html = element('rollout-takeover').innerHTML;
    assert.match(html, /Switching to the operator/);
    assert.match(html, /disabled/);
    assert.match(html, /A switch is under way/);
});

test('a failed switch shows the error and still offers the hand-back', () => {
    const { run, element } = harness();
    run(`
        state.rolloutStatus = {status: 'running', takeover: {
            available: true, state: 'error', control: 'switching', count: 1,
            error: 'RuntimeError: Init failed', action: 'hand_back', reason: null,
            last_switch: {to: 'operator', s: 0.12, error: 'RuntimeError: Init failed'},
            trigger_channel: null}};
        renderRolloutTakeover();
    `);
    const html = element('rollout-takeover').innerHTML;
    assert.match(html, /Nobody: the last switch failed/);
    assert.match(html, /RuntimeError: Init failed/);
    assert.match(html, /to the operator failed after 0\.12 s/);
    assert.match(html, />Hand back<\/button>/);
    assert.doesNotMatch(html, /disabled/);
});

test('after the rollout only the last switch is left', () => {
    const { run, element } = harness();
    run(`
        state.rolloutStatus = {status: 'idle', takeover: {
            available: false, state: null, control: null, error: null, count: 0,
            last_switch: {to: 'policy', s: 0.5}, action: null,
            reason: 'No rollout is running', trigger_channel: null}};
        renderRolloutTakeover();
    `);
    const html = element('rollout-takeover').innerHTML;
    assert.match(html, /Last switch: to the policy in 0\.50 s/);
    assert.doesNotMatch(html, /button/);
});
