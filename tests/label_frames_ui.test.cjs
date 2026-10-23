// Run with: node --test tests/label_frames_ui.test.cjs
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
        id, innerHTML: '', textContent: '', value: '', dataset: {}, style: {}, children: [],
        classList: { toggle() {}, remove() {}, add() {} },
        querySelector() { return null; }, querySelectorAll() { return []; },
        appendChild(child) { this.children.push(child); },
        insertAdjacentHTML(_, html) { this.children.push({ innerHTML: html }); },
        setAttribute() {},
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
        renderDatasetPreviewBlock = (id, preview) => { globalThis.shown = [id, preview.episode_index]; };
        _showPreviewLoadingOverlay = () => {};
        state.processingMode = 'labels';
        state.processingStep = 2;
        state.episodes = [
            {name: 'e1', path: '/s/episodes/job/e1', job: 'job'},
            {name: 'demos', path: '/s/merged/demos', job: null},
        ];
        globalThis.calls = [];
        globalThis.views = {
            '/s/episodes/job/e1': {path: '/s/episodes/job/e1', episodes: [{index: 0, num_frames: 30}],
                checks: {seated: {episodes: {'0': 12}, summary: {episodes: 1, reached: 1, never: 0}}}},
            '/s/merged/demos': {path: '/s/merged/demos',
                episodes: [{index: 0, num_frames: 50}, {index: 1, num_frames: 60}], checks: {}},
        };
        api = async (path, request) => {
            const method = request?.method || 'GET';
            calls.push([method, path, request?.body ? JSON.parse(request.body) : null]);
            if (path.startsWith('/datasets/check-labels?') && method === 'GET') {
                return views[decodeURIComponent(path.split('path=')[1])];
            }
            if (path.startsWith('/datasets/preview?')) {
                return {path: 'p', num_frames: 60, episode_index: Number(path.split('episode_index=')[1])};
            }
            if (path.startsWith('/datasets/series?')) return {series: {}};
            return globalThis.reply;
        };
    `);
    const rows = () => element('label-episode-picker').children.map((child) => child.innerHTML).join('\n');
    return { run, element, rows };
}

test('the label step reads every loaded dataset and names a check already used', async () => {
    const { run, rows } = harness();
    run('renderProcessing();');
    await run('_labelViewsLoading');
    run(`
        assert.deepEqual(calls.map(c => c[1]), [
            '/datasets/check-labels?path=%2Fs%2Fepisodes%2Fjob%2Fe1',
            '/datasets/check-labels?path=%2Fs%2Fmerged%2Fdemos',
        ]);
        assert.equal(state.labelCheck, 'seated');
    `);
    const listed = rows();
    // A recording is one row under its job; a merged dataset heads its episodes.
    assert.match(listed, /job/);
    assert.match(listed, /e1<\/span><\/div>\s*<span class="label-badge label-badge--holds"[^>]*>from 13</);
    assert.match(listed, /demos/);
    assert.match(listed, /Episode 1<\/span><\/div>\s*<span class="label-badge label-badge--none"/);
});

test('an episode is labelled at the frame shown, and the badge follows', async () => {
    const { run, element, rows } = harness();
    run(`state.labelViews = {...views}; state.labelCheck = 'seated'; renderProcessing();`);
    await run(`selectLabelEpisode('/s/merged/demos', 1)`);
    run(`
        assert.deepEqual(shown, ['label-preview-block', 1]);
        assert.ok(calls.some(c => c[1] === '/datasets/preview?path=%2Fs%2Fmerged%2Fdemos&episode_index=1'));
        assert.equal(state.labelFrame, 0, 'an unlabelled episode opens at its start');
    `);
    assert.match(element('label-status').innerHTML, /This episode: not labelled/);
    run(`
        state.labelFrame = 41;
        globalThis.reply = {...views['/s/merged/demos'], checks: {seated: {
            episodes: {'1': 41}, summary: {episodes: 1, reached: 1, never: 0}}}};
    `);
    await element('label-holds').onclick();
    run(`
        const put = calls.at(-1);
        assert.equal(put[0], 'PUT');
        assert.equal(put[1], '/datasets/check-labels');
        assert.deepEqual(put[2], {path: '/s/merged/demos', check: 'seated', episode: 1, first_frame: 41});
    `);
    assert.match(rows(), /label-badge--holds"[^>]*>from 42</);
    const status = element('label-status').innerHTML;
    assert.match(status, /holds from frame 42/);
    assert.match(status, /2 episodes labelled: 2 reach it, 0 never/);
    assert.match(status, /Go to frame 42/);
});

test('never and clear send the right requests', async () => {
    const { run, element } = harness();
    run(`state.labelViews = {...views}; state.labelCheck = 'seated'; renderProcessing();`);
    await run(`selectLabelEpisode('/s/merged/demos', 0)`);
    run(`globalThis.reply = {...views['/s/merged/demos'], checks: {seated: {
        episodes: {'0': null}, summary: {episodes: 1, reached: 0, never: 1}}}};`);
    await element('label-never').onclick();
    run(`assert.deepEqual(calls.at(-1)[2], {path: '/s/merged/demos', check: 'seated', episode: 0, first_frame: null});`);
    assert.match(element('label-status').innerHTML, /never holds/);
    run(`globalThis.reply = views['/s/merged/demos'];`);
    await element('label-clear').onclick();
    run(`
        assert.deepEqual(calls.at(-1).slice(0, 2), ['DELETE',
            '/datasets/check-labels?path=%2Fs%2Fmerged%2Fdemos&check=seated&episode=0']);
    `);
    assert.match(element('label-status').innerHTML, /not labelled/);
});

test('a labelled episode opens at its label', async () => {
    const { run, element } = harness();
    run(`state.labelViews = {...views}; state.labelCheck = 'seated'; renderProcessing();`);
    await run(`selectLabelEpisode('/s/episodes/job/e1', 0)`);
    run('assert.equal(state.labelFrame, 12);');
    assert.match(element('label-status').innerHTML, /holds from frame 13/);
});

test('a bad check name saves nothing', async () => {
    const { run, element } = harness();
    run(`state.labelViews = {...views}; state.labelCheck = '../x'; renderProcessing();`);
    await run(`selectLabelEpisode('/s/merged/demos', 0)`);
    const status = element('label-status').innerHTML;
    assert.match(status, /A check name is 1-64 letters/);
    assert.match(status, /id="label-holds" type="button" disabled/);
    await run('saveLabel(3)');
    run(`assert.ok(!calls.some(c => c[0] !== 'GET'), 'nothing written');`);
});

test('a slow preview of an episode no longer picked is dropped', async () => {
    const { run } = harness();
    run(`
        state.labelViews = {...views}; state.labelCheck = 'seated'; renderProcessing();
        // Episode 0's preview answers only after episode 1's.
        const answer = api;
        globalThis.release = null;
        const held = new Promise((resolve) => { release = resolve; });
        api = async (path, request) => {
            if (path.endsWith('episode_index=0')) await held;
            return answer(path, request);
        };
    `);
    const first = run(`selectLabelEpisode('/s/merged/demos', 0)`);
    await run(`selectLabelEpisode('/s/merged/demos', 1)`);
    run('release();');
    await first;
    run(`
        assert.equal(state.labelPreview.episode_index, 1);
        assert.deepEqual(state.labelTarget, {path: '/s/merged/demos', episode: 1});
    `);
});
