// Run with: node --test tests/dataset_preview.test.cjs
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
    const warnings = [];
    const makeElement = (id = '') => ({
        id, innerHTML: '', textContent: '', dataset: {}, style: {},
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
        console: { warn: (...args) => warnings.push(args) },
        document: { getElementById: element, createElement: () => makeElement() },
    });
    const run = (code) => vm.runInContext(code, context);
    run(source.replace(startup, ''));
    // These tests exercise rendering/navigation, not HTML entity conversion.
    run('escapeHtml = value => String(value);');
    return { run, element, warnings };
}

for (const frames of [500, 15000, 50000]) {
    test(`renders all ${frames} frames across twelve velocity channels`, () => {
        const { run } = harness();
        run(`
            const frames = ${frames};
            const axes = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz'];
            const keys = ['observation.state', 'action'].flatMap(section =>
                axes.map(axis => section + '.left_arm.tcp_twist.' + axis));
            const group = buildDatasetPlotGroups(keys)[0];
            const series = Object.fromEntries(keys.map(key => [key, Array(frames).fill(0.5)]));
            const scope = {stateEnabled: Array(6).fill(true), actionEnabled: Array(6).fill(true)};
            const svg = _buildDatasetPlotSvg(series, group, frames, 0, scope, 30);
            assert.equal((svg.match(/<path /g) || []).length, 12);
            assert.equal((svg.match(/[ML]\\d+\\.\\d+,/g) || []).length, frames * 12);
            assert.ok(!svg.includes('NaN'));
        `);
    });
}

test('range preserves invalid-value filtering, padding, and constant/empty fallbacks', () => {
    const { run } = harness();
    run(`
        const group = {stateKeys: ['state', 'missing'], actionKeys: ['action'],
            stateColors: ['red'], actionColors: ['blue']};
        const scope = {stateEnabled: [true], actionEnabled: [true]};
        for (const [series, high, low] of [
            [{state: [null, NaN, -2], action: [Infinity, -Infinity, 8]}, '9.2', '-3.2'],
            [{state: [4], action: [4]}, '5.0', '3.0'],
            [{state: [0], action: [0]}, '1.0', '-1.0'],
            [{state: [null, NaN], action: [Infinity]}, '1.0', '-1.0'],
        ]) {
            const svg = _buildDatasetPlotSvg(series, group, 3, 0, scope, 30);
            assert.ok(svg.includes('>' + high + '</text>'));
            assert.ok(svg.includes('>' + low + '</text>'));
            assert.ok(!svg.includes('NaN'));
        }
    `);
});

test('episode preview failure leaves Back and Merge usable', async () => {
    const { run, element } = harness();
    run(`
        state.processingMode = 'episodes';
        state.processingStep = 2;
        state.episodes = [{path: '/synthetic', name: 'synthetic'}];
        state.selectedEpisodes = ['/synthetic'];
        state.preview = {path: '/synthetic', num_frames: 15000};
        _renderDatasetPreviewBlock = () => {
            assert.equal(typeof byId('training-prev-step').onclick, 'function');
            assert.equal(typeof byId('training-merge').onclick, 'function');
            throw new Error('injected chart failure');
        };
        renderProcessing();
        renderProcessing = () => {};
        _animateWizardStep = () => {};
        byId('training-prev-step').onclick();
        assert.equal(state.processingStep, 1);
        _showMergeModal = () => {};
        _pollMergeProgress = async () => {};
        api = async (path, request) => {
            assert.equal(path, '/datasets/merge');
            assert.deepEqual(JSON.parse(request.body).episode_paths, ['/synthetic']);
        };
    `);
    assert.match(element('episode-preview-block').innerHTML, /WARNING/);
    await run("byId('training-merge').onclick()");
    run('assert.equal(state.merging, true);');
});

for (const mode of ['new', 'fine_tune', 'processing']) {
    test(`${mode}: navigation is bound before a failing preview and still works`, async () => {
        const { run, element, warnings } = harness();
        const processing = mode === 'processing';
        const fineTune = mode === 'fine_tune';
        const back = processing ? 'merge-prev' : 'training-prev-dataset';
        const next = processing ? 'merge-next' : 'training-flow-next';
        const block = processing ? 'merged-preview-block' : 'merged-dataset-preview-block';
        run(`
            const preview = {path: '/synthetic', num_frames: 15000, numeric_keys: []};
            state.mergedPreview = preview;
            state.mergedPath = preview.path;
            state.processingMode = 'episodes';
            state.processingStep = 3;
            state.trainingMode = '${mode}';
            state.trainingStep = ${fineTune ? 3 : 2};
            state.mergedDatasetPreview = preview;
            state.mergedDatasetPlaying = true;
            state.mergedPlaying = true;
            _renderDatasetPreviewBlock = () => {
                assert.equal(typeof byId('${back}').onclick, 'function');
                assert.equal(typeof byId('${next}').onclick, 'function');
                throw new Error('injected chart failure');
            };
            ${processing ? 'renderProcessing' : 'renderTraining'}();
        `);
        assert.match(element(block).innerHTML, /WARNING: Dataset preview failed/);
        assert.match(element(block).innerHTML, /injected chart failure/);
        assert.equal(warnings.length, 1);
        run(`assert.equal(state.${processing ? 'mergedPlaying' : 'mergedDatasetPlaying'}, false);`);
        // Keep the real click handlers, but isolate their next-page rendering.
        run(`
            renderProcessing = () => {};
            renderTraining = () => {};
            _animateWizardStep = () => {};
            resetTrainingRunViewState = () => {};
            setActiveView = view => { state.activeView = view; };
            bootstrapTraining = async () => {};
            byId('${back}').onclick();
            assert.equal(state.${processing ? 'processingStep' : 'trainingStep'}, ${processing || fineTune ? 2 : 1});
            byId('${next}').onclick();
            ${processing
                ? "assert.equal(state.activeView, 'training'); assert.equal(state.mergedDatasetPath, '/synthetic');"
                : `assert.equal(state.trainingStep, ${fineTune ? 4 : 3});`}
        `);
        await Promise.resolve();
    });
}
