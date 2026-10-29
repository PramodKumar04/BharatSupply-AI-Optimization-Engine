const express = require('express');
const router = express.Router();
const auth = require('../middleware/auth');
const {
    saveModel,
    loadModel,
    predictSingle,
    predictBatch,
    getMetrics,
    getConstraints,
    trainModel,
    healthCheck,
} = require('../controllers/mlController');

// Legacy routes
router.post('/save', auth, saveModel);
router.get('/load', auth, loadModel);

// Health check
router.get('/health', healthCheck);

// Model evaluation metrics & dynamic constraints
router.get('/metrics', getMetrics);
router.get('/constraints', getConstraints);

// Prediction endpoints
router.post('/predict', predictSingle);
router.post('/predict-batch', predictBatch);

// Training endpoint
router.post('/train', auth, trainModel);

module.exports = router;
