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

// Run with: node --test tests/rollout_stiffness_ui.test.cjs
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

test('the page states the stiffness the rollout applies and its source', () => {
    const { run, element } = harness();
    run(`state.rolloutStatus = {status: 'running', stiffness: {
        applied: [2600, 2600, 2900, 150, 150, 150], source: 'checkpoint',
        trained: [2600, 2600, 2900, 150, 150, 150], warning: null}};
        renderRolloutStiffness();`);
    const html = element('rollout-stiffness').innerHTML;
    assert.match(html, /Stiffness: <strong>\[2600, 2600, 2900, 150, 150, 150\]<\/strong> from the checkpoint/);
    assert.match(html, /Trained at: \[2600, 2600, 2900, 150, 150, 150\]/);
    assert.doesNotMatch(html, /rollout-warning/);
});

test('a setting that differs from the training stiffness is shown as a warning', () => {
    const { run, element } = harness();
    run(`state.rolloutStatus = {status: 'running', stiffness: {
        applied: [5000, 5000, 5000, 300, 300, 300], source: 'setting',
        trained: [2600, 2600, 2900, 150, 150, 150],
        warning: 'Trained at stiffness [2600, 2600, 2900, 150, 150, 150], running at [5000, 5000, 5000, 300, 300, 300]'}};
        renderRolloutStiffness();`);
    const html = element('rollout-stiffness').innerHTML;
    assert.match(html, /from ROLLOUT__CARTESIAN_STIFFNESS/);
    assert.match(html, /class="rollout-verdict rollout-warning">Trained at stiffness \[2600/);
});

test('the controller default and an unrecorded checkpoint read as such', () => {
    const { run, element } = harness();
    run(`state.rolloutStatus = {status: 'running', stiffness: {
        applied: null, source: 'controller default', trained: 'not recorded',
        warning: 'Running at the controller default stiffness'}};
        renderRolloutStiffness();`);
    const html = element('rollout-stiffness').innerHTML;
    assert.match(html, /Stiffness: <strong>controller default<\/strong> from the controller default/);
    assert.match(html, /Trained at: not recorded/);
    assert.match(html, /rollout-warning/);
});

test('without a rollout the panel is empty', () => {
    const { run, element } = harness();
    run(`state.rolloutStatus = {status: 'idle', stiffness: null}; renderRolloutStiffness();`);
    assert.equal(element('rollout-stiffness').innerHTML, '');
});

test('the full render places the stiffness panel in the rollout control', () => {
    const { run } = harness();
    const body = run('renderRollout.toString()');
    assert.match(body, /id="rollout-stiffness"/);
    assert.equal(body.split('renderRolloutStiffness();').length, 3, 'rendered on both paths');
});
