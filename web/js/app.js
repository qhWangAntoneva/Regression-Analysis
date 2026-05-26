/**
 * Regression Analysis — Web App
 *
 * A Pyodide-based static web application for running OLS regression
 * analysis in the browser. No server-side Python required.
 *
 * Architecture:
 *   - Pyodide loads Python runtime in-browser (WebAssembly)
 *   - bridge.py runs inside Pyodide, providing:
 *       parse_file, run_regression, compute_diagnostics,
 *       generate_diagnostic_charts, generate_coefficient_chart,
 *       export_csv, export_excel
 *   - GALLERY_DATA is pre-computed (loaded from gallery_data.js)
 *   - Plotly.js renders chart JSON in the browser
 */

'use strict';

// =========================================================================
// Global State
// =========================================================================

const STATE = {
    pyodide: null,             // Pyodide runtime instance
    pyodideReady: false,       // Whether Pyodide has finished loading
    data: null,                // Raw data as [[headers], [row1], ...]
    columns: null,             // [{name, dtype, col_type, ...}]
    nRows: 0,
    nCols: 0,
    result: null,              // Regression result JSON
    diagnostics: null,         // Diagnostics JSON
    charts: null,              // Diagnostic chart JSON
    coefChart: null,           // Coefficient chart JSON
    galleryLoaded: false,      // Whether a gallery scenario is active
    currentFile: null,         // {name, size} of uploaded file
    modelHistory: [],          // [{name, spec, result}] for multi-model comparison
    scatterCharts: {},         // {varName: chartSpec} cached scatter charts
    filterEnabled: false,      // Whether a data filter is active
    filterConditions: null,    // {col, type, min, max, values} filter spec
};

// =========================================================================
// Initialization
// =========================================================================

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initUpload();
    initModelForm();
    initGallery();
    initExport();
    initPyodide();
});

// =========================================================================
// Pyodide Loading
// =========================================================================

async function initPyodide() {
    const progressContainer = document.getElementById('pyodide-progress-container');
    const statusEl = document.getElementById('pyodide-status');

    // Helper to update progress bar and text
    function updatePyodideProgress(percent, statusText) {
        const fill = document.getElementById('pyodide-progress-fill');
        const text = document.getElementById('pyodide-progress-text');
        if (fill) fill.style.width = percent + '%';
        if (text) text.textContent = statusText;
    }

    try {
        // Stage 1: Downloading Pyodide core (0-40%)
        updatePyodideProgress(5, 'Downloading Pyodide core...');

        const pyodide = await loadPyodide({
            indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.27.5/full/',
        });

        updatePyodideProgress(40, 'Pyodide core loaded. Installing packages...');

        // Stage 2: Installing packages (40-70%)
        await pyodide.loadPackage(['numpy', 'pandas', 'statsmodels', 'scipy', 'openpyxl']);

        updatePyodideProgress(70, 'Packages installed. Importing modules...');

        // Stage 3: Loading bridge module (70-95%)
        const bridgeCode = await fetch('py/bridge.py').then(r => r.text());
        pyodide.runPython(bridgeCode);

        updatePyodideProgress(95, 'Bridge loaded. Finalizing...');

        // Stage 4: Ready (95-100%)
        STATE.pyodide = pyodide;
        STATE.pyodideReady = true;

        updatePyodideProgress(100, 'Ready');
        progressContainer.classList.add('ready');
        statusEl.classList.remove('hidden');

        console.log('[Pyodide] Ready with numpy, pandas, statsmodels, scipy, openpyxl');
    } catch (err) {
        console.error('[Pyodide] Failed to initialize:', err);
        updatePyodideProgress(0, 'Error loading Pyodide');
        statusEl.textContent = 'Pyodide: Error';
        statusEl.className = 'status-badge error';
        statusEl.classList.remove('hidden');
        progressContainer.classList.add('hidden');
        showError('data-error', 'Failed to load Python runtime (Pyodide). Please check your internet connection and reload the page.');
    }
}

// =========================================================================
// Tab Navigation
// =========================================================================

function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.disabled) return;
            const tabId = btn.dataset.tab;

            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            tabContents.forEach(c => c.classList.remove('active'));
            const tabEl = document.getElementById(`tab-${tabId}`);
            tabEl.classList.add('active');

            // Resize charts when switching to diagnostics or results tab
            if (tabId === 'diagnostics') {
                setTimeout(() => {
                    resizeAllCharts();
                    if (!document.getElementById('chart-residual-fitted')._fullLayout) {
                        renderDiagnosticChartsIfNeeded();
                    }
                }, 100);
            }
            if (tabId === 'results') {
                setTimeout(() => {
                    const coefEl = document.getElementById('coef-chart');
                    if (coefEl && coefEl._fullLayout) Plotly.Plots.resize(coefEl);
                }, 100);
            }
        });
    });
}

function resizeAllCharts() {
    // Query all Plotly chart containers in the DOM (avoids hardcoded list going stale)
    const chartContainers = document.querySelectorAll('.chart-container[id]');
    chartContainers.forEach(el => {
        if (el._fullLayout) Plotly.Plots.resize(el);
    });
}

function renderDiagnosticChartsIfNeeded() {
    if (STATE.charts) renderDiagnosticCharts();
}

function switchToTab(tabId) {
    const btn = document.getElementById(`tab-btn-${tabId}`);
    if (btn && !btn.disabled) {
        btn.click();
        document.getElementById(`tab-${tabId}`).scrollIntoView({ behavior: 'smooth' });
    }
}

function enableTab(tabId) {
    const btn = document.getElementById(`tab-btn-${tabId}`);
    if (btn) btn.disabled = false;
}

// =========================================================================
// Upload
// =========================================================================

function initUpload() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const removeBtn = document.getElementById('btn-remove-file');

    // Click to browse
    uploadArea.addEventListener('click', () => fileInput.click());

    // File selected via browse
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag & drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Remove file
    removeBtn.addEventListener('click', () => {
        clearData();
    });
}

async function handleFile(file) {
    if (!STATE.pyodideReady) {
        showError('data-error', 'Pyodide is still loading. Please wait and try again.');
        return;
    }

    // Size check
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
        showError('data-error', `File is ${(file.size / 1024 / 1024).toFixed(1)}MB. Maximum recommended size is 50MB.`);
        return;
    }

    clearError('data-error');
    showProgress('upload-progress', true);
    const uploadArea = document.getElementById('upload-area');
    uploadArea.classList.add('hidden');

    try {
        // Read file as base64
        const contentB64 = await readFileAsBase64(file);

        // Call Python bridge to parse
        const pyodide = STATE.pyodide;
        const resultJson = pyodide.runPython(`
            parse_file(${JSON.stringify(file.name)}, ${JSON.stringify(contentB64)})
        `);
        const result = JSON.parse(resultJson);

        if (!result.success) {
            showError('data-error', result.error || 'Failed to parse file.');
            uploadArea.classList.remove('hidden');
            return;
        }

        // Store data
        STATE.data = result.data;
        STATE.columns = result.columns;
        STATE.nRows = result.n_rows;
        STATE.nCols = result.n_cols;
        STATE.currentFile = { name: file.name, size: file.size };
        STATE.galleryLoaded = false;
        STATE.result = null;
        STATE.diagnostics = null;
        STATE.charts = null;
        STATE.coefChart = null;

        // Update UI
        showFileInfo(file.name, file.size);
        renderDataPreview();
        populateVariableSelectors();
        enableTab('model');
        disableTabs('results', 'diagnostics', 'export');

        console.log('[Upload] Parsed', STATE.nRows, 'rows x', STATE.nCols, 'cols');
    } catch (err) {
        console.error('[Upload] Error:', err);
        showError('data-error', `Error parsing file: ${err.message}`);
        uploadArea.classList.remove('hidden');
    } finally {
        showProgress('upload-progress', false);
    }
}

function readFileAsBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            // Extract base64 content (remove data:...;base64, prefix)
            const b64 = reader.result.split(',')[1];
            resolve(b64);
        };
        reader.onerror = () => reject(new Error('Failed to read file.'));
        reader.readAsDataURL(file);
    });
}

function showFileInfo(name, size) {
    const fi = document.getElementById('file-info');
    fi.classList.remove('hidden');
    document.getElementById('file-name').textContent = name;
    document.getElementById('file-size').textContent = formatBytes(size);
}

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function clearData() {
    STATE.data = null;
    STATE.columns = null;
    STATE.nRows = 0;
    STATE.nCols = 0;
    STATE.result = null;
    STATE.diagnostics = null;
    STATE.charts = null;
    STATE.coefChart = null;
    STATE.galleryLoaded = false;
    STATE.currentFile = null;
    STATE.modelHistory = [];
    STATE.scatterCharts = {};
    STATE.compareChart = null;

    document.getElementById('file-info').classList.add('hidden');
    document.getElementById('upload-area').classList.remove('hidden');
    document.getElementById('data-preview-container').classList.add('hidden');
    document.getElementById('data-error').classList.add('hidden');
    clearElement('indep-var-list');
    document.getElementById('indep-var-list').innerHTML = '<p class="empty-hint">Upload data or load a gallery sample first.</p>';
    document.getElementById('dep-var-select').innerHTML = '<option value="">-- Select dependent variable --</option>';
    document.getElementById('btn-run-regression').disabled = true;
    clearElement('interaction-list');
    disableTabs('model', 'results', 'diagnostics', 'export');
    clearResults();
}

function clearResults() {
    document.getElementById('results-content').classList.add('hidden');
    document.getElementById('no-results-message').classList.remove('hidden');
    document.getElementById('diag-content').classList.add('hidden');
    document.getElementById('no-diag-message').classList.remove('hidden');
    document.getElementById('export-content').classList.add('hidden');
    document.getElementById('no-export-message').classList.remove('hidden');
    document.getElementById('compare-chart-section').classList.add('hidden');
    document.getElementById('visualizations-section').classList.add('hidden');
    document.getElementById('roc-chart-section').classList.add('hidden');
    document.getElementById('or-chart-section').classList.add('hidden');
    clearElement('visualizations-grid');
    clearElement('model-compare-list');
}

// =========================================================================
// Data Preview
// =========================================================================

function renderDataPreview() {
    const container = document.getElementById('data-preview-container');
    container.classList.remove('hidden');

    const data = STATE.data;
    if (!data || data.length < 2) return;

    // Data table (first 20 rows)
    const headers = data[0];
    const rows = data.slice(1, Math.min(data.length, 21));
    const tableEl = document.getElementById('data-table');
    let html = '<thead><tr>';
    html += '<th>#</th>';
    headers.forEach(h => { html += `<th>${escapeHtml(String(h))}</th>`; });
    html += '</tr></thead><tbody>';
    rows.forEach((row, i) => {
        html += '<tr>';
        html += `<td class="numeric">${i + 1}</td>`;
        row.forEach(val => {
            html += `<td>${escapeHtml(formatValue(val))}</td>`;
        });
        html += '</tr>';
    });
    if (data.length > 21) {
        html += `<tr><td colspan="${headers.length + 1}" style="text-align:center;color:var(--color-text-muted);">... ${data.length - 21} more rows</td></tr>`;
    }
    html += '</tbody>';
    tableEl.innerHTML = html;

    // Variable info table
    const viTable = document.getElementById('varinfo-table');
    let viHtml = '<thead><tr><th>Column</th><th>Type</th><th>Inferred</th><th>Unique</th><th>Missing</th><th>Missing Rate</th></tr></thead><tbody>';
    STATE.columns.forEach(c => {
        viHtml += '<tr>';
        viHtml += `<td><strong>${escapeHtml(c.name)}</strong></td>`;
        viHtml += `<td>${escapeHtml(c.dtype)}</td>`;
        viHtml += `<td>${escapeHtml(c.col_type)}</td>`;
        viHtml += `<td class="numeric">${c.n_unique}</td>`;
        viHtml += `<td class="numeric">${c.n_missing}</td>`;
        const rateColor = c.missing_rate > 0.2 ? 'color:var(--color-danger);font-weight:600' : '';
        viHtml += `<td class="numeric" style="${rateColor}">${(c.missing_rate * 100).toFixed(1)}%</td>`;
        viHtml += '</tr>';
    });
    viHtml += '</tbody>';
    viTable.innerHTML = viHtml;
}

// =========================================================================
// Variable Selectors
// =========================================================================

function populateVariableSelectors() {
    const columns = STATE.columns;
    if (!columns) return;

    // Dependent variable dropdown
    const dvSelect = document.getElementById('dep-var-select');
    dvSelect.innerHTML = '<option value="">-- Select dependent variable --</option>';
    columns.forEach(c => {
        // Suggest numeric columns for DV, plus categorical columns with exactly 2 unique values
        // (string-encoded binary variables like "Yes"/"No", "Male"/"Female")
        if (c.col_type === 'numeric' || (c.col_type === 'categorical' && c.n_unique === 2)) {
            dvSelect.innerHTML += `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)}</option>`;
        }
    });

    // Independent variable checkboxes (with transform controls for numeric vars)
    const ivList = document.getElementById('indep-var-list');
    ivList.innerHTML = '';
    columns.forEach(c => {
        if (c.col_type === 'id') {
            ivList.innerHTML += `
                <label style="opacity:0.5">
                    <input type="checkbox" value="${escapeHtml(c.name)}" disabled>
                    <span class="var-id">${escapeHtml(c.name)}</span>
                    <span style="font-size:0.7rem;color:var(--color-text-muted)">(ID)</span>
                </label>`;
        } else if (c.col_type === 'numeric') {
            const cssClass = 'var-numeric';
            ivList.innerHTML += `
                <div class="var-item">
                    <label>
                        <input type="checkbox" value="${escapeHtml(c.name)}">
                        <span class="${cssClass}">${escapeHtml(c.name)}</span>
                    </label>
                    <select class="form-select form-select-sm var-transform" data-var="${escapeHtml(c.name)}">
                        <option value="">--</option>
                        <option value="log">Log</option>
                        <option value="standardize">Z</option>
                        <option value="center">Center</option>
                        <option value="square">Sq</option>
                    </select>
                </div>`;
        } else {
            const cssClass = 'var-categorical';
            ivList.innerHTML += `
                <label>
                    <input type="checkbox" value="${escapeHtml(c.name)}">
                    <span class="${cssClass}">${escapeHtml(c.name)}</span>
                </label>`;
        }
    });

    if (columns.length === 0) {
        ivList.innerHTML = '<p class="empty-hint">No variables found in data.</p>';
    }

    // Populate interaction term dropdowns
    populateInteractionDropdowns();

    // Populate MixedLM group variable and Panel entity/time selectors
    // All columns are eligible (including categorical and ID columns)
    const allColsOptions = columns.map(c =>
        `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)} (${c.col_type})</option>`
    ).join('');
    const dropdownsToPopulate = [
        'opt-group-var', 'opt-entity-var', 'opt-time-var',
    ];
    dropdownsToPopulate.forEach(id => {
        const sel = document.getElementById(id);
        if (sel) sel.innerHTML = '<option value="">-- Select --</option>' + allColsOptions;
    });

    // Populate filter column dropdown
    populateFilterUI();
}

// =========================================================================
// Model Form
// =========================================================================

function initModelForm() {
    const dvSelect = document.getElementById('dep-var-select');
    const runBtn = document.getElementById('btn-run-regression');
    const modelTypeSelect = document.getElementById('opt-model-type');

    // Enable run button when DV and at least one IV are selected
    dvSelect.addEventListener('change', checkRunButton);
    document.getElementById('indep-var-list').addEventListener('change', checkRunButton);

    // Model type change: show/hide relevant options
    if (modelTypeSelect) {
        modelTypeSelect.addEventListener('change', onModelTypeChange);
        onModelTypeChange(); // Initialize state
    }

    runBtn.addEventListener('click', runRegression);

    // MixedLM / Panel selector listeners
    const groupVarSelect = document.getElementById('opt-group-var');
    const entityVarSelect = document.getElementById('opt-entity-var');
    const timeVarSelect = document.getElementById('opt-time-var');
    if (groupVarSelect) groupVarSelect.addEventListener('change', checkRunButton);
    if (entityVarSelect) entityVarSelect.addEventListener('change', checkRunButton);
    if (timeVarSelect) timeVarSelect.addEventListener('change', checkRunButton);

    // Interaction term controls
    initInteractions();

    // Compare models button
    document.getElementById('btn-save-model').addEventListener('click', saveModelForComparison);
    document.getElementById('btn-compare-models').addEventListener('click', compareModels);
    document.getElementById('btn-clear-compare').addEventListener('click', clearModelHistory);

    // Data filter controls
    initDataFilter();
}

function onModelTypeChange() {
    const modelType = document.getElementById('opt-model-type').value;
    const isMLE = ['logit', 'probit', 'poisson', 'negbin'].includes(modelType);
    const isMixedLM = modelType === 'mixedlm';
    const isPanel = modelType === 'panel';
    const isPanelML = isMixedLM || isPanel;

    // MLE models use MLE, no HC covariance types. Also hide for MixedLM / Panel.
    const covSelect = document.getElementById('opt-cov');
    if (covSelect) {
        covSelect.disabled = isMLE || isPanelML;
        if (isMLE || isPanelML) covSelect.value = 'nonrobust';
    }

    // Show/hide MixedLM group variable selector
    const mixedlmControls = document.getElementById('mixedlm-controls');
    if (mixedlmControls) {
        mixedlmControls.classList.toggle('hidden', !isMixedLM);
    }

    // Show/hide Panel entity / time / model selectors
    const panelControls = document.getElementById('panel-controls');
    if (panelControls) {
        panelControls.classList.toggle('hidden', !isPanel);
    }

    // Re-validate run button (extra required fields for mixedlm/panel)
    checkRunButton();
}

function checkRunButton() {
    const dv = document.getElementById('dep-var-select').value;
    const checked = document.querySelectorAll('#indep-var-list input[type="checkbox"]:checked');
    const modelType = document.getElementById('opt-model-type').value;

    let canRun = dv && checked.length > 0;

    // MixedLM: require group_var
    if (modelType === 'mixedlm') {
        const groupVar = document.getElementById('opt-group-var').value;
        if (!groupVar) canRun = false;
    }

    // Panel: require entity_var and time_var
    if (modelType === 'panel') {
        const entityVar = document.getElementById('opt-entity-var').value;
        const timeVar = document.getElementById('opt-time-var').value;
        if (!entityVar || !timeVar) canRun = false;
    }

    document.getElementById('btn-run-regression').disabled = !canRun;
}

function getSelectedIVs() {
    const checked = document.querySelectorAll('#indep-var-list input[type="checkbox"]:checked');
    return Array.from(checked).map(cb => cb.value);
}

// =========================================================================
// Interaction Terms UI
// =========================================================================

function initInteractions() {
    document.getElementById('btn-add-interaction').addEventListener('click', addInteraction);
}

function populateInteractionDropdowns() {
    const columns = STATE.columns || [];
    const numericVars = columns.filter(c => c.col_type === 'numeric');

    // Also include binary categorical variables (exactly 2 unique values)
    // These can be safely multiplied as 0/1 after encoding
    const binaryCategoricalVars = [];
    if (STATE.data && STATE.data.length > 1) {
        const headerRow = STATE.data[0];
        for (const col of columns) {
            if (col.col_type === 'categorical' || col.col_type === 'binary') {
                const colIdx = headerRow.indexOf(col.name);
                if (colIdx >= 0) {
                    const values = new Set();
                    for (let i = 1; i < STATE.data.length; i++) {
                        const v = STATE.data[i][colIdx];
                        if (v != null && v !== '') values.add(v);
                    }
                    if (values.size === 2) {
                        binaryCategoricalVars.push(col);
                    }
                }
            }
        }
    }

    const allVars = [...numericVars, ...binaryCategoricalVars];
    const options = allVars.map(c => `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)}</option>`).join('');

    const sel1 = document.getElementById('interaction-var1');
    const sel2 = document.getElementById('interaction-var2');
    if (sel1) sel1.innerHTML = '<option value="">-- Select --</option>' + options;
    if (sel2) sel2.innerHTML = '<option value="">-- Select --</option>' + options;
}

function addInteraction() {
    const sel1 = document.getElementById('interaction-var1');
    const sel2 = document.getElementById('interaction-var2');
    const v1 = sel1.value;
    const v2 = sel2.value;

    if (!v1 || !v2) {
        showError('model-error', 'Select two variables for the interaction term.');
        return;
    }
    if (v1 === v2) {
        showError('model-error', 'Interaction requires two different variables.');
        return;
    }

    const list = document.getElementById('interaction-list');
    // Check for duplicates
    const existing = list.querySelectorAll('.interaction-tag');
    for (const tag of existing) {
        if (tag.dataset.pair === `${v1},${v2}` || tag.dataset.pair === `${v2},${v1}`) {
            showError('model-error', 'This interaction term already exists.');
            return;
        }
    }

    const tag = document.createElement('div');
    tag.className = 'interaction-tag';
    tag.dataset.pair = `${v1},${v2}`;
    tag.innerHTML = `<span>${escapeHtml(v1)} &times; ${escapeHtml(v2)}</span>
        <button class="btn-interaction-remove" onclick="this.parentElement.remove()">&times;</button>`;
    list.appendChild(tag);

    clearError('model-error');
    // Reset dropdowns
    sel1.value = '';
    sel2.value = '';
}

function getTransforms() {
    const selects = document.querySelectorAll('.var-transform');
    const transforms = {};
    selects.forEach(sel => {
        if (sel.value) {
            transforms[sel.dataset.var] = sel.value;
        }
    });
    return Object.keys(transforms).length > 0 ? transforms : null;
}

function getInteractions() {
    const tags = document.querySelectorAll('#interaction-list .interaction-tag');
    const pairs = [];
    tags.forEach(tag => {
        const [v1, v2] = tag.dataset.pair.split(',');
        pairs.push([v1, v2]);
    });
    return pairs.length > 0 ? pairs : null;
}

// =========================================================================
// Data Filter
// =========================================================================

function initDataFilter() {
    const filterSection = document.getElementById('data-filter-section');
    const filterColSelect = document.getElementById('filter-col-select');
    const btnApply = document.getElementById('btn-apply-filter');
    const btnClear = document.getElementById('btn-clear-filter');

    filterColSelect.addEventListener('change', onFilterColumnChange);
    btnApply.addEventListener('click', applyDataFilter);
    btnClear.addEventListener('click', clearDataFilter);
}

function populateFilterUI() {
    const filterSection = document.getElementById('data-filter-section');
    if (!STATE.columns || STATE.columns.length === 0) {
        filterSection.classList.add('hidden');
        return;
    }
    filterSection.classList.remove('hidden');

    const filterColSelect = document.getElementById('filter-col-select');
    const currentVal = filterColSelect.value;
    filterColSelect.innerHTML = '<option value="">-- Select column to filter --</option>';
    STATE.columns.forEach(c => {
        if (c.col_type !== 'id') {
            filterColSelect.innerHTML += `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)} (${c.col_type})</option>`;
        }
    });
    if (currentVal && STATE.columns.some(c => c.name === currentVal)) {
        filterColSelect.value = currentVal;
    }
}

function onFilterColumnChange() {
    const colName = document.getElementById('filter-col-select').value;
    const numControls = document.getElementById('filter-numeric-controls');
    const catControls = document.getElementById('filter-cat-controls');
    numControls.classList.add('hidden');
    catControls.classList.add('hidden');

    if (!colName || !STATE.columns) return;

    const colMeta = STATE.columns.find(c => c.name === colName);
    if (!colMeta) return;

    if (colMeta.col_type === 'numeric') {
        numControls.classList.remove('hidden');
    } else if (colMeta.col_type === 'categorical') {
        catControls.classList.remove('hidden');
        // Populate checkboxes from data
        const colIdx = STATE.data[0].indexOf(colName);
        const uniqueVals = [...new Set(STATE.data.slice(1).map(r => r[colIdx]).filter(v => v != null))].sort();
        const checkboxesDiv = document.getElementById('filter-cat-checkboxes');
        checkboxesDiv.innerHTML = uniqueVals.map(v => `
            <label>
                <input type="checkbox" value="${escapeHtml(String(v))}" checked> ${escapeHtml(String(v))}
            </label>
        `).join('');
    }
}

function applyDataFilter() {
    const colName = document.getElementById('filter-col-select').value;
    if (!colName) {
        showError('model-error', 'Select a column to filter.');
        return;
    }
    const colMeta = STATE.columns.find(c => c.name === colName);
    if (!colMeta) return;

    if (colMeta.col_type === 'numeric') {
        const minVal = parseFloat(document.getElementById('filter-num-min').value);
        const maxVal = parseFloat(document.getElementById('filter-num-max').value);
        if (isNaN(minVal) && isNaN(maxVal)) {
            showError('model-error', 'Enter at least one range value.');
            return;
        }
        STATE.filterEnabled = true;
        STATE.filterConditions = { col: colName, type: 'numeric', min: isNaN(minVal) ? null : minVal, max: isNaN(maxVal) ? null : maxVal };
    } else {
        const checked = document.querySelectorAll('#filter-cat-checkboxes input[type="checkbox"]:checked');
        STATE.filterEnabled = true;
        STATE.filterConditions = { col: colName, type: 'categorical', values: Array.from(checked).map(cb => cb.value) };
    }
    clearError('model-error');
    document.getElementById('btn-apply-filter').textContent = 'Filter Applied';
    setTimeout(() => { document.getElementById('btn-apply-filter').textContent = 'Apply Filter'; }, 1500);
}

function clearDataFilter() {
    STATE.filterEnabled = false;
    STATE.filterConditions = null;
    document.getElementById('filter-col-select').value = '';
    document.getElementById('filter-numeric-controls').classList.add('hidden');
    document.getElementById('filter-cat-controls').classList.add('hidden');
    document.getElementById('filter-num-min').value = '';
    document.getElementById('filter-num-max').value = '';
    clearError('model-error');
}

// =========================================================================
// Run Regression
// =========================================================================

async function runRegression() {
    if (!STATE.pyodideReady || !STATE.data) return;

    clearError('model-error');
    showProgress('regression-progress', true);
    document.getElementById('btn-run-regression').disabled = true;

    try {
        const pyodide = STATE.pyodide;
        const dataJson = JSON.stringify({ data: STATE.data, columns: STATE.columns });
        const modelType = document.getElementById('opt-model-type').value;
        const spec = {
            dep_var: document.getElementById('dep-var-select').value,
            indep_vars: getSelectedIVs(),
            has_intercept: document.getElementById('opt-intercept').value === 'true',
            alpha: parseFloat(document.getElementById('opt-alpha').value),
            cov_type: document.getElementById('opt-cov').value,
            missing_strategy: document.getElementById('opt-missing').value,
            model_type: modelType,
        };

        // MixedLM: pass group variable
        if (modelType === 'mixedlm') {
            spec.group_var = document.getElementById('opt-group-var').value;
        }

        // Panel: pass entity, time, and panel model type
        if (modelType === 'panel') {
            spec.entity_var = document.getElementById('opt-entity-var').value;
            spec.time_var = document.getElementById('opt-time-var').value;
            spec.panel_model = document.getElementById('opt-panel-model').value;
        }

        // Collect transforms and interactions from UI
        const transforms = getTransforms();
        const interactions = getInteractions();
        if (transforms) spec.transforms = transforms;
        if (interactions) spec.interactions = interactions;

        // Pass data filter conditions
        if (STATE.filterEnabled && STATE.filterConditions) {
            spec.filter = STATE.filterConditions;
        }

        // Serialize spec for model history and bridge
        const specJson = JSON.stringify(spec);

        // Run regression
        const resultJson = pyodide.runPython(`
            run_regression(${JSON.stringify(dataJson)}, ${JSON.stringify(specJson)})
        `);
        const result = JSON.parse(resultJson);

        if (!result.success) {
            showError('model-error', result.error || 'Regression failed.');
            return;
        }

        STATE.result = result;

        console.log('[Regression] Success. R-squared:', result.r_squared, 'N:', result.n_obs);

        // Enable result tabs
        enableTab('results');
        enableTab('diagnostics');
        enableTab('export');

        // Switch to results
        switchToTab('results');
        renderResults(result);

        // After showing results, compute diagnostics and charts in parallel
        const parallelTasks = [
            generateAndRenderCharts(resultJson),
            generateCoefficientChart(resultJson),
        ];

        // Diagnostics (OLS/Panel/MixedLM only — VIF/residual tests don't apply to MLE)
        const isMLE = ['logit', 'probit', 'poisson', 'negbin'].includes(result.model_type);
        if (!isMLE) {
            parallelTasks.push(computeAndRenderDiagnostics(dataJson, resultJson));
        }

        // Scatter plots: available for all model types
        parallelTasks.push(generateAllScatterCharts(dataJson, result));

        // Logit/Probit-specific tasks (ROC is available for both binary choice models)
        if (result.model_type === 'logit' || result.model_type === 'probit') {
            parallelTasks.push(generateROCChart(resultJson));
        }
        // OR chart only for logit
        if (result.model_type === 'logit') {
            parallelTasks.push(generateORChart(resultJson));
        }

        await Promise.all(parallelTasks);

    } catch (err) {
        console.error('[Regression] Error:', err);
        showError('model-error', `Regression error: ${err.message}`);
    } finally {
        showProgress('regression-progress', false);
        document.getElementById('btn-run-regression').disabled = false;
    }
}

// =========================================================================
// Render Results
// =========================================================================

function renderResults(result) {
    document.getElementById('no-results-message').classList.add('hidden');
    document.getElementById('results-content').classList.remove('hidden');

    // Specification
    document.getElementById('result-spec-display').textContent =
        `Model: ${result.specification || 'OLS'} | N = ${result.n_obs} | SE: ${result.se_type || 'nonrobust'}`;

    // Model statistics grid
    renderStatsGrid(result);

    // Coefficient table
    renderCoefficientTable(result.coefficients || [], result.variable_labels || {});

    // ANOVA table (if diagnostics available)
    if (STATE.diagnostics && STATE.diagnostics.anova) {
        renderAnovaTable(STATE.diagnostics.anova);
    }

    // Summary text
    renderSummaryText(result);
}

function renderStatsGrid(result) {
    const grid = document.getElementById('model-stats-grid');
    const mt = result.model_type;
    const isLogit = mt === 'logit';
    const isMLE = ['logit', 'probit', 'poisson', 'negbin'].includes(mt);
    const isCount = ['poisson', 'negbin'].includes(mt);
    const isPanel = mt === 'panel';
    const isMixedLM = mt === 'mixedlm';
    const isOLS = mt === 'OLS' || !isMLE && !isPanel && !isMixedLM;
    const stats = [];

    if (isMLE) {
        stats.push(
            { label: 'Pseudo R-squared', value: fmtNum(result.pseudo_r_squared, '.6f') },
            { label: 'Log-Likelihood', value: fmtNum(result.log_likelihood, '.2f') },
        );
        if (result.llr != null) {
            stats.push({
                label: 'LR chi2',
                value: `${fmtNum(result.llr, '.4f')} (p=${fmtPvalue(result.llr_pvalue)})`,
            });
        }
        if (isCount && result.dispersion != null) {
            stats.push({ label: 'Dispersion', value: fmtNum(result.dispersion, '.4f') });
        }
        if (isLogit) {
            stats.push({ label: 'Method', value: 'Logit (MLE)' });
        } else if (mt === 'probit') {
            stats.push({ label: 'Method', value: 'Probit (MLE)' });
        } else if (mt === 'poisson') {
            stats.push({ label: 'Method', value: 'Poisson (MLE)' });
        } else if (mt === 'negbin') {
            stats.push({ label: 'Method', value: 'NegativeBinomial (MLE)' });
        }
    } else if (isPanel) {
        if (result.within_r_squared != null) {
            stats.push({ label: 'Within R²', value: fmtNum(result.within_r_squared, '.6f') });
        }
        if (result.between_r_squared != null) {
            stats.push({ label: 'Between R²', value: fmtNum(result.between_r_squared, '.6f') });
        }
        if (result.overall_r_squared != null) {
            stats.push({ label: 'Overall R²', value: fmtNum(result.overall_r_squared, '.6f') });
        }
        if (result.f_statistic) {
            stats.push({
                label: 'F-statistic',
                value: `${fmtNum(result.f_statistic[0], '.4f')} (p=${fmtPvalue(result.f_statistic[1])})`,
            });
        }
        stats.push(
            { label: 'Entities', value: result.entity_count || 'N/A' },
            { label: 'Periods', value: result.time_count || 'N/A' },
        );
    } else if (isMixedLM) {
        stats.push(
            { label: 'R-squared', value: fmtNum(result.r_squared, '.6f') },
            { label: 'Adj R-squared', value: fmtNum(result.adj_r_squared, '.6f') },
            { label: 'RMSE', value: fmtNum(result.rmse, '.4f') },
            { label: 'Groups', value: result.group_count || 'N/A' },
        );
        if (result.re_var) {
            Object.keys(result.re_var).forEach(k => {
                stats.push({ label: `RE: ${k}`, value: fmtNum(result.re_var[k], '.4f') });
            });
        }
    } else {
        // OLS (default)
        stats.push(
            { label: 'R-squared', value: fmtNum(result.r_squared, '.6f') },
            { label: 'Adj R-squared', value: fmtNum(result.adj_r_squared, '.6f') },
            { label: 'RMSE', value: fmtNum(result.rmse, '.4f') },
        );
        if (result.f_statistic) {
            stats.push({
                label: 'F-statistic',
                value: `${fmtNum(result.f_statistic[0], '.4f')} (p=${fmtPvalue(result.f_statistic[1])})`,
            });
        }
    }

    stats.push(
        { label: 'AIC', value: fmtNum(result.aic, '.2f') },
        { label: 'BIC', value: fmtNum(result.bic, '.2f') },
        { label: 'N', value: result.n_obs },
    );
    if (isOLS || isPanel || isMixedLM) {
        stats.push({ label: 'Log-Likelihood', value: fmtNum(result.log_likelihood, '.2f') });
    }

    grid.innerHTML = stats.map(s => `
        <div class="stat-card">
            <div class="stat-value">${s.value}</div>
            <div class="stat-label">${s.label}</div>
        </div>
    `).join('');
}

function renderCoefficientTable(coefs, variableLabels) {
    variableLabels = variableLabels || {};
    const mt = STATE.result ? STATE.result.model_type : '';
    const isLogit = mt === 'logit';
    const isMLE = ['logit', 'probit', 'poisson', 'negbin'].includes(mt);
    const isCount = ['poisson', 'negbin'].includes(mt);
    const statLabel = isMLE ? 'z-value' : 't-value';
    const statField = isMLE ? 'z_stat' : 't_stat';

    // Update table header
    const thead = document.querySelector('#coef-table thead tr');
    let headerHTML = '<th>Variable</th><th>Coefficient</th><th>Std. Error</th>';
    headerHTML += `<th>${statLabel}</th>`;
    if (isLogit) headerHTML += '<th>Odds Ratio</th>';
    if (isCount) headerHTML += '<th>IRR</th>';
    headerHTML += '<th>p-value</th><th>95% CI Low</th><th>95% CI High</th><th>Sig.</th>';
    thead.innerHTML = headerHTML;

    const tbody = document.querySelector('#coef-table tbody');
    tbody.innerHTML = coefs.map(c => {
        const pClass = c.pvalue < 0.05 ? 'p-significant' : (c.pvalue < 0.1 ? 'p-marginal' : '');
        const statVal = c[statField] != null ? c[statField] : (c.t_stat || 0);
        const displayName = variableLabels[c.name] || c.name;
        let rowHTML = `<tr>
            <td><strong>${escapeHtml(displayName)}</strong></td>
            <td class="numeric">${fmtNum(c.coef, '.6f')}</td>
            <td class="numeric">${fmtNum(c.se, '.6f')}</td>
            <td class="numeric">${fmtNum(statVal, '.4f')}</td>`;
        if (isLogit) {
            // Only logit gets odds_ratio in the result dict
            rowHTML += `<td class="numeric">${fmtNum(c.odds_ratio, '.4f')}</td>`;
        }
        if (isCount) {
            // Count models get IRR
            rowHTML += `<td class="numeric">${fmtNum(c.irr, '.4f')}</td>`;
        }
        rowHTML += `<td class="numeric ${pClass}">${fmtPvalue(c.pvalue)}</td>
            <td class="numeric">${fmtNum(c.ci_lower, '.6f')}</td>
            <td class="numeric">${fmtNum(c.ci_upper, '.6f')}</td>
            <td style="color:var(--color-danger);font-weight:600">${c.significance || ''}</td>
        </tr>`;
        return rowHTML;
    }).join('');

    // Update export table visibility
    updateExportState();
}

function renderAnovaTable(anova) {
    const tbody = document.querySelector('#anova-table tbody');
    const rows = ['explained', 'residual', 'total'];
    tbody.innerHTML = rows.map(key => {
        const row = anova[key];
        if (!row) return '';
        return `<tr>
            <td>${row.source || key}</td>
            <td class="numeric">${fmtNum(row.SS, '.6f')}</td>
            <td class="numeric">${row.df}</td>
            <td class="numeric">${fmtNum(row.MS, '.4f')}</td>
            <td class="numeric">${fmtNum(row.F, '.4f')}</td>
            <td class="numeric">${fmtPvalue(row.p_value)}</td>
        </tr>`;
    }).join('');
}

function renderSummaryText(result) {
    const el = document.getElementById('summary-text');
    const mt = result.model_type;
    const isMLE = ['logit', 'probit', 'poisson', 'negbin'].includes(mt);
    const isLogit = mt === 'logit';
    const isCount = ['poisson', 'negbin'].includes(mt);
    const isPanel = mt === 'panel';
    const isMixedLM = mt === 'mixedlm';
    let text = '';
    const methodLabel = mt === 'logit' ? 'Logit' : mt === 'probit' ? 'Probit' :
        mt === 'poisson' ? 'Poisson' : mt === 'negbin' ? 'NegativeBinomial' :
        mt === 'mixedlm' ? 'MixedLM' : mt === 'panel' ? 'Panel' : 'OLS';
    text += `${methodLabel} Regression: ${result.specification || 'Unspecified'}\n\n`;

    if (isMLE) {
        text += `Pseudo R-squared = ${result.pseudo_r_squared != null ? result.pseudo_r_squared.toFixed(4) : 'N/A'}.\n`;
        if (result.llr != null) {
            const llrP = result.llr_pvalue != null ? result.llr_pvalue : 1;
            const sigLabel = llrP < 0.001 ? '<0.001' : llrP < 0.05 ? '<0.05' : llrP < 0.1 ? '<0.10' : '>=0.10';
            text += `Overall model: LR chi2 = ${result.llr.toFixed(4)}, p ${sigLabel}.\n`;
        }
        if (isCount && result.dispersion != null) {
            text += `Dispersion = ${result.dispersion.toFixed(4)}.\n`;
        }
    } else if (isPanel) {
        if (result.within_r_squared != null) text += `Within R-squared = ${result.within_r_squared.toFixed(4)}.\n`;
        if (result.between_r_squared != null) text += `Between R-squared = ${result.between_r_squared.toFixed(4)}.\n`;
        if (result.overall_r_squared != null) text += `Overall R-squared = ${result.overall_r_squared.toFixed(4)}.\n`;
        if (result.f_statistic) {
            const fv = result.f_statistic[0], fp = result.f_statistic[1];
            const sigLabel = fp < 0.001 ? '<0.001' : fp < 0.05 ? '<0.05' : fp < 0.1 ? '<0.10' : '>=0.10';
            text += `Overall model: F = ${fv.toFixed(4)}, p ${sigLabel}.\n`;
        }
        text += `Entities = ${result.entity_count || 'N/A'}, Periods = ${result.time_count || 'N/A'}.\n`;
    } else if (isMixedLM) {
        text += `R-squared = ${result.r_squared != null ? result.r_squared.toFixed(4) : 'N/A'}`;
        if (result.adj_r_squared != null) text += `, Adj R-squared = ${result.adj_r_squared.toFixed(4)}`;
        text += `.\nRMSE = ${result.rmse != null ? result.rmse.toFixed(4) : 'N/A'}.\n`;
        text += `Groups = ${result.group_count || 'N/A'}.\n`;
    } else {
        if (result.f_statistic) {
            const df1 = result.n_params - 1;
            const df2 = result.df_resid;
            const fv = result.f_statistic[0];
            const fp = result.f_statistic[1];
            const sigLabel = fp < 0.001 ? '<0.001' : fp < 0.05 ? `<0.05` : fp < 0.1 ? `<0.10` : `>=0.10`;
            text += `Overall model: F(${df1},${df2}) = ${fv.toFixed(4)}, p ${sigLabel}.\n`;
        }
        text += `R-squared = ${result.r_squared != null ? result.r_squared.toFixed(4) : 'N/A'}`;
        if (result.adj_r_squared != null) text += `, Adj R-squared = ${result.adj_r_squared.toFixed(4)}`;
        text += `.\nRMSE = ${result.rmse != null ? result.rmse.toFixed(4) : 'N/A'}.\n`;
    }
    text += `Log-Likelihood = ${result.log_likelihood != null ? result.log_likelihood.toFixed(2) : 'N/A'}.\n`;
    text += `AIC = ${result.aic != null ? result.aic.toFixed(2) : 'N/A'}, BIC = ${result.bic != null ? result.bic.toFixed(2) : 'N/A'}.\n`;
    text += `N = ${result.n_obs}.\n\n`;

    text += `${methodLabel} Coefficients:\n`;
    (result.coefficients || []).forEach(c => {
        const statVal = isMLE ? (c.z_stat != null ? c.z_stat : c.t_stat) : c.t_stat;
        const coefStr = c.coef != null ? c.coef.toFixed(6).padStart(12) : '         N/A';
        const seStr = c.se != null ? c.se.toFixed(6) : 'N/A';
        const statStr = statVal != null ? statVal.toFixed(4) : 'N/A';
        let coefLine = `  ${c.name.padEnd(20)} ${coefStr} (SE: ${seStr}, ${isMLE ? 'z' : 't'}=${statStr}, p=${fmtPvalue(c.pvalue)}) ${c.significance}`;
        if (isLogit && c.odds_ratio != null) {
            coefLine += ` OR=${c.odds_ratio.toFixed(4)}`;
        }
        if (isCount && c.irr != null) {
            coefLine += ` IRR=${c.irr.toFixed(4)}`;
        }
        text += coefLine + '\n';
    });

    el.textContent = text;
}

// =========================================================================
// Diagnostics
// =========================================================================

async function computeAndRenderDiagnostics(dataJson, resultJson) {
    try {
        const pyodide = STATE.pyodide;
        const diagJson = pyodide.runPython(`
            compute_diagnostics(${JSON.stringify(dataJson)}, ${JSON.stringify(resultJson)})
        `);
        const diag = JSON.parse(diagJson);
        if (diag.success) {
            STATE.diagnostics = diag;
            // Re-render ANOVA if results tab is active
            if (document.getElementById('tab-results').classList.contains('active') && diag.anova) {
                renderAnovaTable(diag.anova);
            }
            if (document.getElementById('tab-diagnostics').classList.contains('active')) {
                renderDiagnostics();
            }
        }
    } catch (err) {
        console.error('[Diagnostics] Computation error:', err);
    }
}

function renderDiagnostics() {
    document.getElementById('no-diag-message').classList.add('hidden');
    document.getElementById('diag-content').classList.remove('hidden');

    if (!STATE.diagnostics) return;

    const diag = STATE.diagnostics;

    // Residual diagnostic table
    if (diag.residual_tests) {
        const rt = diag.residual_tests;
        const tbody = document.querySelector('#resid-diag-table tbody');
        let rows = '';
        if (rt.shapiro_normal) {
            rows += `<tr>
                <td>Shapiro-Wilk (Normality)</td>
                <td class="numeric">${rt.shapiro_stat != null ? rt.shapiro_stat.toFixed(6) : 'N/A'}</td>
                <td class="numeric">${rt.shapiro_pvalue != null ? fmtPvalue(rt.shapiro_pvalue) : 'N/A'}</td>
                <td>${rt.shapiro_normal}</td>
            </tr>`;
        }
        if (rt.dw_autocorrelation) {
            rows += `<tr>
                <td>Durbin-Watson (Autocorrelation)</td>
                <td class="numeric">${rt.dw_stat != null ? rt.dw_stat.toFixed(4) : 'N/A'}</td>
                <td>--</td>
                <td>${rt.dw_autocorrelation}</td>
            </tr>`;
        }
        tbody.innerHTML = rows;
    }

    // VIF table
    if (diag.vif && diag.vif.length > 0) {
        const tbody = document.querySelector('#vif-table tbody');
        tbody.innerHTML = diag.vif.map(v => {
            let color = '';
            if (v.diagnosis === 'High') color = 'color:var(--color-danger);font-weight:600';
            else if (v.diagnosis === 'Moderate') color = 'color:var(--color-warning);font-weight:600';
            return `<tr>
                <td>${escapeHtml(v.variable)}</td>
                <td class="numeric" style="${color}">${v.vif.toFixed(4)}</td>
                <td>${v.diagnosis}</td>
            </tr>`;
        }).join('');
    }
}

// =========================================================================
// Charts
// =========================================================================

async function generateAndRenderCharts(resultJson) {
    try {
        const pyodide = STATE.pyodide;
        const chartsJson = pyodide.runPython(`
            generate_diagnostic_charts(${JSON.stringify(resultJson)})
        `);
        const charts = JSON.parse(chartsJson);
        if (charts.success && charts.charts) {
            STATE.charts = charts.charts;
            // Only render immediately if diagnostics tab is visible
            if (document.getElementById('tab-diagnostics').classList.contains('active')) {
                renderDiagnosticCharts();
            }
        }
    } catch (err) {
        console.error('[Charts] Generation error:', err);
    }
}

async function generateCoefficientChart(resultJson) {
    try {
        const pyodide = STATE.pyodide;
        const chartJson = pyodide.runPython(`
            generate_coefficient_chart(${JSON.stringify(resultJson)})
        `);
        const chart = JSON.parse(chartJson);
        if (chart.success && chart.chart) {
            STATE.coefChart = chart.chart;
            renderCoefficientChart();
        }
    } catch (err) {
        console.error('[Charts] Coefficient chart error:', err);
    }
}

function renderDiagnosticCharts() {
    if (!STATE.charts) return;
    const charts = STATE.charts;

    const config = { responsive: true, displayModeBar: true, displaylogo: false };

    if (charts.residual_fitted) {
        Plotly.newPlot('chart-residual-fitted', charts.residual_fitted.data, charts.residual_fitted.layout, config);
    }
    if (charts.qq) {
        Plotly.newPlot('chart-qq', charts.qq.data, charts.qq.layout, config);
    }
    if (charts.scale_location) {
        Plotly.newPlot('chart-scale-location', charts.scale_location.data, charts.scale_location.layout, config);
    }
    if (charts.cooks_distance) {
        Plotly.newPlot('chart-cooks', charts.cooks_distance.data, charts.cooks_distance.layout, config);
    }
}

function renderCoefficientChart() {
    if (!STATE.coefChart) return;
    const config = { responsive: true, displayModeBar: true, displaylogo: false };
    Plotly.newPlot('coef-chart', STATE.coefChart.data, STATE.coefChart.layout, config);
}

// =========================================================================
// Multi-Model Comparison
// =========================================================================

function saveModelForComparison() {
    if (!STATE.result) {
        showError('model-error', 'Run a regression first before saving for comparison.');
        return;
    }
    const dv = document.getElementById('dep-var-select').value;
    const iva = getSelectedIVs().join(', ');
    const name = document.getElementById('model-compare-name').value.trim() ||
                 `Model ${STATE.modelHistory.length + 1}: ${dv} ~ ${iva}`;

    const existingNames = STATE.modelHistory.map(m => m.name);
    let uniqueName = name;
    let suffix = 1;
    while (existingNames.includes(uniqueName)) {
        suffix++;
        uniqueName = `${name} (${suffix})`;
    }

    STATE.modelHistory.push({
        name: uniqueName,
        spec: null,  // spec not needed for comparison chart
        result: STATE.result,
    });
    updateModelHistoryUI();
    clearError('model-error');
}

function updateModelHistoryUI() {
    const list = document.getElementById('model-compare-list');
    if (!list) return;
    const section = document.getElementById('model-compare-section');
    if (section) section.classList.remove('hidden');
    list.innerHTML = STATE.modelHistory.map((m, i) => `
        <div class="compare-model-item">
            <span>${escapeHtml(m.name)}</span>
            <button class="btn btn-sm btn-outline" onclick="removeModelFromHistory(${i})">&times;</button>
        </div>
    `).join('');

    // Enable/disable compare button
    const btn = document.getElementById('btn-compare-models');
    if (btn) btn.disabled = STATE.modelHistory.length < 2;
}

function removeModelFromHistory(index) {
    STATE.modelHistory.splice(index, 1);
    updateModelHistoryUI();
    if (STATE.modelHistory.length < 2) {
        const section = document.getElementById('compare-chart-section');
        if (section) section.classList.add('hidden');
    }
}

function clearModelHistory() {
    STATE.modelHistory = [];
    updateModelHistoryUI();
    const section = document.getElementById('compare-chart-section');
    if (section) section.classList.add('hidden');
    const listSection = document.getElementById('model-compare-list');
    if (listSection) listSection.innerHTML = '';
}

async function compareModels() {
    if (STATE.modelHistory.length < 2) {
        showError('model-error', 'Need at least 2 models to compare.');
        return;
    }
    if (!STATE.pyodideReady || !STATE.pyodide) {
        showError('model-error', 'Pyodide is not ready.');
        return;
    }

    try {
        const modelsJson = STATE.modelHistory.map(m => ({
            name: m.name,
            result: m.result,
        }));

        const pyodide = STATE.pyodide;
        const chartJson = pyodide.runPython(`
            compare_models(${JSON.stringify(JSON.stringify(modelsJson))})
        `);
        const chart = JSON.parse(chartJson);

        if (chart.success && chart.chart) {
            STATE.compareChart = chart.chart;
            renderComparisonChart();
        } else {
            showError('model-error', chart.error || 'Comparison failed.');
        }
    } catch (err) {
        console.error('[Compare] Error:', err);
        showError('model-error', `Comparison error: ${err.message}`);
    }
}

function renderComparisonChart() {
    if (!STATE.compareChart) return;
    const section = document.getElementById('compare-chart-section');
    section.classList.remove('hidden');
    const config = { responsive: true, displayModeBar: true, displaylogo: false };
    Plotly.newPlot('compare-chart', STATE.compareChart.data, STATE.compareChart.layout, config);
}

// =========================================================================
// Scatterplot Visualizations
// =========================================================================

async function generateAllScatterCharts(dataJson, result) {
    if (!STATE.pyodideReady || !STATE.pyodide || !result) return;

    const indepVars = result.indep_vars || [];
    const depVar = result.dep_var;
    // Only generate for the original independent variables (skip interaction terms)
    const origVars = indepVars.filter(v => !v.includes('_x_'));
    if (origVars.length === 0) return;

    STATE.scatterCharts = {};
    const vizSection = document.getElementById('visualizations-section');
    const vizGrid = document.getElementById('visualizations-grid');

    // Show loading state
    if (vizGrid) vizGrid.innerHTML = '<div class="empty-hint" style="grid-column:1/-1">Generating scatter plots...</div>';

    const pyodide = STATE.pyodide;
    for (const xVar of origVars) {
        try {
            const chartJson = pyodide.runPython(`
                generate_scatter_chart(${JSON.stringify(dataJson)}, ${JSON.stringify(xVar)}, ${JSON.stringify(depVar)})
            `);
            const chart = JSON.parse(chartJson);
            if (chart.success && chart.chart) {
                STATE.scatterCharts[xVar] = chart.chart;
            }
        } catch (err) {
            console.error(`[Scatter] Error for ${xVar}:`, err);
        }
    }

    renderScatterCharts();
}

function renderScatterCharts() {
    const vizSection = document.getElementById('visualizations-section');
    const vizGrid = document.getElementById('visualizations-grid');
    if (!vizSection || !vizGrid) return;

    const chartKeys = Object.keys(STATE.scatterCharts);
    if (chartKeys.length === 0) {
        vizSection.classList.add('hidden');
        return;
    }

    vizSection.classList.remove('hidden');
    vizGrid.innerHTML = '';

    const config = { responsive: true, displayModeBar: true, displaylogo: false };

    chartKeys.forEach(xVar => {
        const cardDiv = document.createElement('div');
        cardDiv.className = 'viz-card';
        const chartDiv = document.createElement('div');
        chartDiv.className = 'chart-container';
        chartDiv.style.minHeight = '300px';
        cardDiv.appendChild(chartDiv);
        vizGrid.appendChild(cardDiv);

        const chart = STATE.scatterCharts[xVar];
        Plotly.newPlot(chartDiv, chart.data, chart.layout, config);
    });
}

// =========================================================================
// Logit-specific Charts: ROC and OR Forest Plot
// =========================================================================

async function generateROCChart(resultJson) {
    try {
        const pyodide = STATE.pyodide;
        const chartJson = pyodide.runPython(`
            generate_roc_chart(${JSON.stringify(resultJson)})
        `);
        const chart = JSON.parse(chartJson);
        if (chart.success && chart.chart) {
            STATE.rocChart = chart.chart;
            STATE.rocAUC = chart.auc;
            renderROCChart();
        } else {
            console.warn('[ROC] Chart generation failed:', chart.error);
        }
    } catch (err) {
        console.error('[ROC] Error:', err);
    }
}

async function generateORChart(resultJson) {
    try {
        const pyodide = STATE.pyodide;
        const chartJson = pyodide.runPython(`
            generate_or_chart(${JSON.stringify(resultJson)})
        `);
        const chart = JSON.parse(chartJson);
        if (chart.success && chart.chart) {
            STATE.orChart = chart.chart;
            renderORChart();
        } else {
            console.warn('[OR] Chart generation failed:', chart.error);
        }
    } catch (err) {
        console.error('[OR] Error:', err);
    }
}

function renderROCChart() {
    if (!STATE.rocChart) return;
    const section = document.getElementById('roc-chart-section');
    section.classList.remove('hidden');
    const config = { responsive: true, displayModeBar: true, displaylogo: false };
    Plotly.newPlot('roc-chart', STATE.rocChart.data, STATE.rocChart.layout, config);
}

function renderORChart() {
    if (!STATE.orChart) return;
    const section = document.getElementById('or-chart-section');
    section.classList.remove('hidden');
    const config = { responsive: true, displayModeBar: true, displaylogo: false };
    Plotly.newPlot('or-chart', STATE.orChart.data, STATE.orChart.layout, config);
}

// =========================================================================
// Gallery (Pre-computed Scenarios)
// =========================================================================

function initGallery() {
    const grid = document.getElementById('gallery-grid');
    if (typeof GALLERY_DATA === 'undefined' || !GALLERY_DATA.length) {
        grid.innerHTML = '<p class="empty-hint">Gallery data not available.</p>';
        return;
    }

    grid.innerHTML = GALLERY_DATA.map(item => `
        <div class="gallery-card" data-gallery-id="${item.id}">
            <div class="card-title">${item.persona_icon || ''} ${escapeHtml(item.title)}</div>
            <div class="card-persona">${escapeHtml(item.persona)}</div>
            <div class="card-desc">${escapeHtml(item.description)}</div>
            <div class="card-tags">
                ${(item.tags || []).map(t => `<span class="tag">${escapeHtml(t)}</span>`).join('')}
            </div>
            <div class="card-meta">
                <span>N = ${item.n_obs}</span>
                <span>DV: ${item.dep_var}</span>
            </div>
            <button class="btn btn-primary btn-sm" style="margin-top:0.75rem;width:100%">Load &amp; View Results</button>
        </div>
    `).join('');

    // Click handlers
    grid.querySelectorAll('.gallery-card').forEach(card => {
        const btn = card.querySelector('button');
        btn.addEventListener('click', (e) => {
            e.stopPropagation(); // prevent double-load
            const id = card.dataset.galleryId;
            loadGalleryItem(id);
        });
    });
}

function loadGalleryItem(id) {
    const item = GALLERY_DATA.find(g => g.id === id);
    if (!item) return;

    // Populate state from gallery data
    STATE.data = item.data;
    STATE.columns = item.columns.map(name => ({
        name: name,
        dtype: item.column_types[name] === 'numeric' ? 'float64' : 'object',
        col_type: item.column_types[name] || 'numeric',
        n_unique: 0,
        n_missing: 0,
        missing_rate: 0,
    }));
    STATE.nRows = item.n_obs;
    STATE.nCols = item.columns.length;
    STATE.galleryLoaded = true;
    STATE.currentFile = null;
    STATE.result = item.result;
    STATE.diagnostics = null;
    STATE.charts = null;
    STATE.coefChart = null;
    STATE.modelHistory = [];
    STATE.compareChart = null;
    STATE.scatterCharts = {};
    STATE.rocChart = null;
    STATE.orChart = null;

    // Hide file info, update upload area
    document.getElementById('file-info').classList.add('hidden');
    document.getElementById('upload-area').classList.add('hidden');
    document.getElementById('data-error').classList.add('hidden');

    // Clear stale model comparison UI and chart sections
    clearModelHistory();
    document.getElementById('roc-chart-section').classList.add('hidden');
    document.getElementById('or-chart-section').classList.add('hidden');
    document.getElementById('visualizations-section').classList.add('hidden');

    // Render data preview
    renderDataPreview();

    // Populate variable selectors and pre-select
    populateVariableSelectors();
    document.getElementById('dep-var-select').value = item.dep_var;
    const checkboxes = document.querySelectorAll('#indep-var-list input[type="checkbox"]');
    checkboxes.forEach(cb => {
        cb.checked = (item.indep_vars || []).includes(cb.value);
    });
    checkRunButton();

    // Enable all tabs
    enableTab('model');
    enableTab('results');
    enableTab('diagnostics');
    enableTab('export');

    // Generate coefficient chart from pre-computed result
    renderResults(item.result);
    STATE.diagnostics = computeDiagnosticsFromResult(item.result, item.data);
    renderDiagnostics();
    renderAnovaTable(STATE.diagnostics.anova);
    generateCoefficientChartFromResult(item.result);

    // Show gallery info in model spec page
    const uploadArea = document.getElementById('upload-area');
    uploadArea.classList.add('hidden');

    // Switch to results
    switchToTab('results');

    // Generate scatter charts for Gallery items
    if (STATE.pyodideReady) {
        const dataJson = JSON.stringify({ data: STATE.data, columns: STATE.columns });
        generateAllScatterCharts(dataJson, item.result);
    }

    console.log('[Gallery] Loaded:', item.title, 'N:', item.n_obs);
}

function computeDiagnosticsFromResult(result, data) {
    // Compute basic diagnostics from the pre-computed result
    const residuals = result.residuals || [];
    const n = result.n_obs || 0;
    const p = result.n_params || 1;

    // Approximate residual diagnostics (full computation needs Pyodide)
    let anova = {};
    if (result.rmse && result.df_resid) {
        const ss_resid = result.rmse ** 2 * result.df_resid;
        const df_expl = result.n_params - 1;
        const df_total = result.n_obs - 1;
        let ss_total = ss_resid;
        if (result.r_squared != null && result.r_squared < 1 && result.r_squared >= 0) {
            ss_total = ss_resid / (1 - result.r_squared);
        }
        const ss_expl = ss_total - ss_resid;
        const ms_expl = df_expl > 0 ? ss_expl / df_expl : NaN;
        const ms_resid = result.df_resid > 0 ? ss_resid / result.df_resid : NaN;
        let f_val = NaN, f_p = NaN;
        if (ms_resid > 0 && ms_expl > 0) {
            f_val = ms_expl / ms_resid;
        }
        if (result.f_statistic) {
            f_val = result.f_statistic[0];
            f_p = result.f_statistic[1];
        }
        anova = {
            explained: { source: 'Regression', SS: roundNum(ss_expl, 6), df: df_expl, MS: roundNum(ms_expl, 6), F: roundNum(f_val, 6), p_value: roundNum(f_p, 6) },
            residual: { source: 'Residual', SS: roundNum(ss_resid, 6), df: result.df_resid, MS: roundNum(ms_resid, 6) },
            total: { source: 'Total', SS: roundNum(ss_total, 6), df: df_total },
        };
    }

    // Simple VIF (placeholder - full computation needs Pyodide)
    let vif = null;

    // Residual diagnostics
    let residual_tests = {};
    if (residuals.length > 0 && STATE.pyodideReady) {
        // Compute actual Shapiro-Wilk and Durbin-Watson via Pyodide
        try {
            const pyodide = STATE.pyodide;
            const residualListJson = JSON.stringify(residuals);
            const testsJson = pyodide.runPython(`
import json
import numpy as np

residuals = np.array(json.loads('''${residualListJson}'''), dtype=float)

result = {}
# Shapiro-Wilk
if len(residuals) >= 3:
    try:
        from scipy import stats
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        result["shapiro_stat"] = round(float(shapiro_stat), 6)
        result["shapiro_pvalue"] = float(shapiro_p)
        result["shapiro_normal"] = "Yes" if shapiro_p > 0.05 else "No"
    except Exception:
        result["shapiro_normal"] = "Error"
else:
    result["shapiro_normal"] = "Insufficient data"

# Durbin-Watson
if len(residuals) >= 2:
    try:
        diff_sum = np.sum(np.diff(residuals) ** 2)
        total_sum = np.sum(residuals ** 2)
        if total_sum > 0:
            dw = float(diff_sum / total_sum)
            result["dw_stat"] = round(dw, 4)
            if dw < 1.0:
                result["dw_autocorrelation"] = "Positive (strong)"
            elif dw > 3.0:
                result["dw_autocorrelation"] = "Negative (strong)"
            elif dw < 1.5:
                result["dw_autocorrelation"] = "Positive (mild)"
            elif dw > 2.5:
                result["dw_autocorrelation"] = "Negative (mild)"
            else:
                result["dw_autocorrelation"] = "None"
        else:
            result["dw_autocorrelation"] = "N/A (zero variance)"
    except Exception:
        result["dw_autocorrelation"] = "Error"
else:
    result["dw_autocorrelation"] = "Insufficient data"

json.dumps(result)
            `);
            residual_tests = JSON.parse(testsJson);
        } catch (err) {
            console.error('[Diagnostics] Pyodide computation error:', err);
            residual_tests = {
                shapiro_normal: "Error computing diagnostics",
                dw_autocorrelation: "Error computing diagnostics",
            };
        }
    } else if (residuals.length > 0 && !STATE.pyodideReady) {
        residual_tests = {
            shapiro_normal: "N/A (Pyodide not loaded)",
            dw_autocorrelation: "N/A (Pyodide not loaded)",
        };
    } else {
        residual_tests = {
            shapiro_normal: "N/A (no residuals)",
            dw_autocorrelation: "N/A (no residuals)",
        };
    }

    return {
        success: true,
        vif: vif,
        residual_tests: residual_tests,
        anova: anova,
    };
}

function generateCoefficientChartFromResult(result) {
    // Build coefficient chart JSON from pre-computed result
    const coefs = result.coefficients || [];
    const nonIntercept = coefs.filter(c => c.name !== 'Intercept')
        .sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef));
    const intercept = coefs.find(c => c.name === 'Intercept');
    const sorted = intercept ? [intercept, ...nonIntercept] : nonIntercept;

    const n = sorted.length;
    const names = sorted.map(c => c.name);
    const est = sorted.map(c => c.coef || 0);
    const ciLow = sorted.map(c => c.ci_lower || 0);
    const ciHigh = sorted.map(c => c.ci_upper || 0);

    const traces = [];
    for (let i = 0; i < n; i++) {
        traces.push({
            type: 'scatter',
            x: [ciLow[i], ciHigh[i]],
            y: [n - 1 - i, n - 1 - i],
            mode: 'lines',
            line: { color: '#1f77b4', width: 2 },
            showlegend: false,
            hoverinfo: 'none',
        });
    }
    traces.push({
        type: 'scatter',
        x: est,
        y: Array.from({ length: n }, (_, i) => n - 1 - i),
        mode: 'markers+text',
        marker: { color: '#1f77b4', size: 10, symbol: 'circle' },
        text: sorted.map(c => (c.pvalue < 0.01 ? '***' : c.pvalue < 0.05 ? '**' : c.pvalue < 0.1 ? '*' : '')),
        textposition: 'middle right',
        textfont: { size: 12, color: 'red' },
        name: 'Coefficient',
        showlegend: false,
    });

    const layout = {
        title: { text: 'Coefficient Estimates (Dot-Whisker)', x: 0.5 },
        xaxis: { title: 'Coefficient Estimate' },
        yaxis: { tickvals: Array.from({ length: n }, (_, i) => i), ticktext: [...names].reverse(), title: '' },
        template: 'plotly_white',
        height: Math.max(300, n * 40),
        annotations: [{
            xref: 'paper', yref: 'paper', x: 1, y: -0.08,
            text: '*** p<0.01, ** p<0.05, * p<0.1',
            showarrow: false, font: { size: 10, color: 'gray' }, xanchor: 'right',
        }],
    };

    STATE.coefChart = { data: traces, layout: layout };
    renderCoefficientChart();
}

// =========================================================================
// Export
// =========================================================================

function initExport() {
    document.getElementById('btn-export-csv').addEventListener('click', () => exportFormat('csv'));
    document.getElementById('btn-export-excel').addEventListener('click', () => exportFormat('excel'));
    document.getElementById('btn-export-text').addEventListener('click', () => exportText());
    document.getElementById('btn-export-charts').addEventListener('click', () => exportCharts());
}

function updateExportState() {
    if (STATE.result) {
        document.getElementById('no-export-message').classList.add('hidden');
        document.getElementById('export-content').classList.remove('hidden');
    }
}

async function exportFormat(format) {
    if (!STATE.result) return;

    try {
        showError('export-error', '');

        const resultStr = JSON.stringify(STATE.result);
        const pyodide = STATE.pyodide;

        if (format === 'csv') {
            let csvResult;
            if (pyodide && STATE.pyodideReady && !STATE.galleryLoaded) {
                csvResult = JSON.parse(pyodide.runPython(`export_csv(${JSON.stringify(resultStr)})`));
            } else {
                // Generate CSV from result directly
                csvResult = { success: true, csv: generateCSVFromResult(STATE.result) };
            }
            if (csvResult.success) {
                let csv = csvResult.csv;
                if (STATE.galleryLoaded) {
                    csv = '# Note: This result is from a pre-computed Gallery sample. Some statistics may be approximate.\n' + csv;
                }
                downloadBlob(csv, 'regression_results.csv', 'text/csv');
            }
        } else if (format === 'excel') {
            if (pyodide && STATE.pyodideReady && !STATE.galleryLoaded) {
                const excelResult = JSON.parse(pyodide.runPython(`export_excel(${JSON.stringify(resultStr)})`));
                if (excelResult.success) {
                    const byteChars = atob(excelResult.excel_b64);
                    const bytes = new Uint8Array(byteChars.length);
                    for (let i = 0; i < byteChars.length; i++) bytes[i] = byteChars.charCodeAt(i);
                    downloadBlob(new Blob([bytes]), excelResult.filename || 'regression_results.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
                } else {
                    showError('export-error', 'Excel export failed. Downloading CSV instead.');
                    const csvResult2 = JSON.parse(pyodide.runPython(`export_csv(${JSON.stringify(resultStr)})`));
                    if (csvResult2.success) {
                        downloadBlob(csvResult2.csv, 'regression_results.csv', 'text/csv');
                    }
                }
            } else {
                showError('export-error', 'Excel export requires Pyodide runtime. Downloading CSV instead.');
                let csv = generateCSVFromResult(STATE.result);
                if (STATE.galleryLoaded) {
                    csv = '# Note: This result is from a pre-computed Gallery sample. Some statistics may be approximate.\n' + csv;
                }
                downloadBlob(csv, 'regression_results.csv', 'text/csv');
            }
        }
    } catch (err) {
        console.error('[Export] Error:', err);
        alert('Export failed: ' + err.message);
    }
}

function generateCSVFromResult(result) {
    const mt = result.model_type || '';
    const isMLE = ['logit', 'probit', 'poisson', 'negbin'].includes(mt);
    const isLogit = mt === 'logit';
    const isCount = ['poisson', 'negbin'].includes(mt);
    const statLabel = isMLE ? 'z-value' : 't-value';
    const statField = isMLE ? 'z_stat' : 't_stat';
    const extraHeader = isLogit ? ',Odds Ratio' : (isCount ? ',IRR' : '');
    const extraField = isLogit ? 'odds_ratio' : (isCount ? 'irr' : '');

    let lines = [`Variable,Coefficient,Std.Err.,${statLabel}${extraHeader},p-value,CI(95%) Low,CI(95%) High,Significance`];
    (result.coefficients || []).forEach(c => {
        const statVal = c[statField] != null ? c[statField] : (c.t_stat || 0);
        const extraVal = extraField ? `,${c[extraField] || ''}` : '';
        lines.push(`"${c.name}",${c.coef},${c.se},${statVal}${extraVal},${c.pvalue},${c.ci_lower},${c.ci_upper},${c.significance}`);
    });
    lines.push('');
    lines.push('# Model Summary');
    lines.push(`# Model Type,${mt.toUpperCase()}`);
    if (isMLE) {
        lines.push(`# Pseudo R-squared,${result.pseudo_r_squared}`);
        lines.push(`# LR chi2,${result.llr}`);
        lines.push(`# LR p-value,${result.llr_pvalue}`);
        if (isCount && result.dispersion != null) lines.push(`# Dispersion,${result.dispersion}`);
    } else {
        lines.push(`# R-squared,${result.r_squared}`);
        lines.push(`# Adj R-squared,${result.adj_r_squared}`);
        lines.push(`# RMSE,${result.rmse}`);
    }
    lines.push(`# Log-Likelihood,${result.log_likelihood}`);
    lines.push(`# AIC,${result.aic}`);
    lines.push(`# BIC,${result.bic}`);
    lines.push(`# N,${result.n_obs}`);
    lines.push(`# Specification,"${result.specification}"`);
    return lines.join('\n');
}

function exportText() {
    if (!STATE.result) return;
    const el = document.getElementById('summary-text');
    const text = el.textContent || 'No summary available.';
    downloadBlob(text, 'regression_summary.txt', 'text/plain');
}

function exportCharts() {
    // Lazy-render any unrendered chart DIVs (user may not have visited Diagnostics tab)
    if (STATE.charts && !document.getElementById('chart-residual-fitted')._fullLayout) {
        renderDiagnosticCharts();
    }
    if (STATE.coefChart && !document.getElementById('coef-chart')._fullLayout) {
        renderCoefficientChart();
    }
    if (STATE.rocChart && !document.getElementById('roc-chart')._fullLayout) {
        renderROCChart();
    }
    if (STATE.orChart && !document.getElementById('or-chart')._fullLayout) {
        renderORChart();
    }
    // Build list of all chart containers that have been rendered
    const chartIds = ['chart-residual-fitted', 'chart-qq', 'chart-scale-location', 'chart-cooks',
                      'coef-chart', 'roc-chart', 'or-chart'];
    let exported = 0;
    chartIds.forEach(id => {
        const el = document.getElementById(id);
        if (el && el._fullLayout) {
            Plotly.downloadImage(el, { format: 'png', width: 800, height: 500, filename: id });
            exported++;
        }
    });
    if (exported === 0) {
        if (!STATE.charts && !STATE.coefChart) {
            alert('No charts available to export. Run a regression first to generate charts.');
        } else {
            alert('No charts available to export. Please visit the Diagnostics and Results tabs first to render the charts, then try exporting again.');
        }
    }
}

function downloadBlob(content, filename, mimeType) {
    const blob = content instanceof Blob ? content : new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// =========================================================================
// Utility Functions
// =========================================================================

function escapeHtml(str) {
    if (str == null) return '';
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function formatValue(val) {
    if (val == null) return 'NA';
    if (typeof val === 'number') return val.toPrecision(6);
    return String(val);
}

function fmtNum(val, fmt) {
    if (val == null) return 'N/A';
    if (typeof val !== 'number' || isNaN(val)) return 'N/A';
    // Apply the format pattern
    if (fmt.startsWith('.')) {
        const decimals = parseInt(fmt.substring(1));
        return val.toFixed(decimals);
    }
    return val.toString();
}

function fmtPvalue(p) {
    if (p == null || isNaN(p)) return 'N/A';
    if (p < 0.0001) return '<0.0001';
    return p.toFixed(4);
}

function roundNum(val, decimals) {
    if (val == null || isNaN(val)) return val;
    return parseFloat(val.toFixed(decimals));
}

function showError(id, message) {
    const el = document.getElementById(id);
    if (el) {
        el.textContent = message;
        el.classList.remove('hidden');
    }
}

function clearError(id) {
    const el = document.getElementById(id);
    if (el) el.classList.add('hidden');
}

function showProgress(id, show) {
    const el = document.getElementById(id);
    if (el) {
        if (show) el.classList.remove('hidden');
        else el.classList.add('hidden');
    }
}

function clearElement(id) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = '';
}

function disableTabs(...ids) {
    ids.forEach(id => {
        const btn = document.getElementById(`tab-btn-${id}`);
        if (btn) btn.disabled = true;
    });
}
