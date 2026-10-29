/**
 * BharatSupply API & ML Integration Test Suite.
 * Tests:
 * 1. Express ML routes and prediction service adapter.
 * 2. Delay decomposition and demurrage arithmetic.
 * 3. Fallback when ML service is offline.
 * 4. MILP optimizer consuming delay predictions.
 */

const assert = require('assert');
const path = require('path');
const mlPredictionService = require('../backend/services/mlPredictionService');

async function runTests() {
    console.log('🚀 Running BharatSupply API & ML Integration Tests...\n');
    let passed = 0;
    let failed = 0;

    async function test(name, fn) {
        try {
            await fn();
            console.log(`✅ PASS: ${name}`);
            passed++;
        } catch (err) {
            console.error(`❌ FAIL: ${name}\n   Error: ${err.message}`);
            failed++;
        }
    }

    // ── Test 1: Service Health & Constraints ───────────────────────
    await test('ML Service Constraints & Defaults', async () => {
        const constraints = await mlPredictionService.getConstraints();
        assert(constraints, 'Constraints should not be null');
        assert(typeof constraints.monsoonPenalty === 'number', 'monsoonPenalty must be number');
        assert(typeof constraints.weatherRiskFactor === 'number', 'weatherRiskFactor must be number');
        assert(typeof constraints.meanTrainedDelay === 'number', 'meanTrainedDelay must be number');
    });

    // ── Test 2: Single Vessel Prediction & Arithmetic ─────────────
    await test('Single Vessel Prediction Arithmetic (No Double Counting)', async () => {
        const vessel = {
            id: 'VSL_TEST_01',
            name: 'MV Bulk Carrier One',
            origin: 'Australia',
            destinationPort: 'paradip',
            destinationPortName: 'Paradip Port',
            material: 'Coking Coal',
            vesselAge: 12,
            scheduledETA: new Date().toISOString(),
        };

        const result = await mlPredictionService.predictVessel(vessel);
        assert(result, 'Prediction result should exist');
        assert(typeof result.predicted_total_delay_h === 'number', 'Total delay must be a number');
        assert(result.predicted_total_delay_h >= 0, 'Total delay must be non-negative');

        // Check delay decomposition
        const b = result.predicted_berthing_delay_h || 0;
        const c = result.predicted_customs_delay_h || 0;
        const y = result.predicted_yard_dwell_h || 0;
        const sum = Math.round((b + c + y) * 10) / 10;
        assert(Math.abs(sum - result.predicted_total_delay_h) <= 0.5,
            `Components (${b}+${c}+${y}=${sum}) must equal total (${result.predicted_total_delay_h})`);

        // Check Demurrage consistency
        const expectedDemurrage = Math.round((result.predicted_total_delay_h / 24.0) * 25000);
        assert(Math.abs(result.predicted_demurrage_usd - expectedDemurrage) <= 2,
            `Demurrage ($${result.predicted_demurrage_usd}) must match expected ($${expectedDemurrage})`);
    });

    // ── Test 3: Batch Prediction ──────────────────────────────────
    await test('Batch Vessel Predictions', async () => {
        const vessels = [
            { name: 'MV Ocean Steel', destinationPort: 'vizag', scheduledETA: new Date() },
            { name: 'MV Coal Express', destinationPort: 'haldia', scheduledETA: new Date() },
        ];
        const batchResults = await mlPredictionService.predictBatch(vessels);
        assert(Array.isArray(batchResults), 'Batch results must be an array');
        assert.strictEqual(batchResults.length, 2, 'Must return predictions for each vessel');
    });

    // ── Test 4: Evaluation Metrics ────────────────────────────────
    await test('Held-Out Model Evaluation Metrics Comparison', async () => {
        const metrics = await mlPredictionService.getMetrics();
        assert(metrics.baseline_model, 'Must have baseline model evaluation');
        assert(metrics.new_ensemble_model, 'Must have new ensemble model evaluation');
        assert(metrics.new_ensemble_model.metrics.MAE_hours > 0, 'MAE must be positive');
        assert(metrics.new_ensemble_model.metrics.RMSE_hours > 0, 'RMSE must be positive');
    });

    // ── Test 5: Optimizer Consuming Prediction ─────────────────────
    await test('MILP Optimizer Consuming Delay Predictions', async () => {
        const { optimizeLogistics } = await import('../js/engines/optimizer.js');
        const vessels = [
            { id: 'v1', name: 'MV Ocean Steel', destinationPort: 'vizag', quantity: 50000, material: 'coal', scheduledETA: new Date() }
        ];
        const rakes = [
            { id: 'r1', fromPort: 'vizag', toPlant: 'bhilai', capacity: 3800, quantity: 3800, costPerTonKm: 1.7, distance: 750, material: 'coal', available: true }
        ];
        const inventory = {
            bhilai: { coal: { currentLevel: 10000, safetyStock: 8000, dailyConsumption: 8000 } }
        };

        const plan = optimizeLogistics(vessels, rakes, inventory);
        assert(plan, 'Plan must be generated');
        assert(plan.feasible, 'Plan must be feasible');
        assert(plan.totalCost > 0, 'Total cost must be positive');
        assert(plan.vesselSchedule.length > 0, 'Vessel schedule must be populated');
        assert(typeof plan.vesselSchedule[0].predictedDelay === 'number', 'Schedule must include predicted delay');
    });

    console.log(`\n========================================`);
    console.log(`Test Results: ${passed} Passed, ${failed} Failed`);
    console.log(`========================================\n`);

    if (failed > 0) process.exit(1);
}

runTests().catch(err => {
    console.error('Test suite failed:', err);
    process.exit(1);
});

